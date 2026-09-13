#include "recurrent_restorer.hpp"
#include "attention_mask_descriptor.hpp"
#include "flow_warp.hpp"
#include "gelu.hpp"
#include "spynet.hpp"

#include <allocator.h>
#include <gpu.h>
#include <net.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace rvf
{
namespace
{
void check(int result, const char *operation)
{
    if (result != 0)
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(result) +
                                 ")");
}
} // namespace

class RecurrentRestorer::Impl
{
  public:
    SpyNet flow;
    ncnn::Net core;
    FlowWarpLayer warp{FlowWarpPadding::Zeros};
    // Allocators outlive every VkMat held by this instance.
    std::unique_ptr<ncnn::VkBlobAllocator> blobs;
    std::unique_ptr<ncnn::VkStagingAllocator> staging;
    ncnn::Option opt;
    ncnn::Mat previous_cpu, state_cpu;
    ncnn::VkMat previous_gpu, state_gpu;
    int width = 0, height = 0;

    Impl(const std::filesystem::path &directory, bool vulkan, int device_index, int threads)
        : flow((directory / "spynet.param").string(), (directory / "spynet.bin").string(), vulkan,
               device_index, threads)
    {
        core.opt = flow.option();
        core.opt.use_winograd23_convolution = false;
        core.opt.use_winograd43_convolution = false;
        core.opt.use_winograd63_convolution = false;
        opt = core.opt;
        core.register_custom_layer("archs.realviformer_arch.AttentionMaskDescriptor",
                                   create_attention_mask_descriptor_layer);
        core.register_custom_layer("GELU", create_gelu_layer);
        if (vulkan)
        {
            core.set_vulkan_device(flow.device());
            warp.vkdev = flow.device();
            blobs = std::make_unique<ncnn::VkBlobAllocator>(flow.device());
            staging = std::make_unique<ncnn::VkStagingAllocator>(flow.device());
            opt.blob_vkallocator = blobs.get();
            opt.workspace_vkallocator = blobs.get();
            opt.staging_vkallocator = staging.get();
        }
        check(warp.create_pipeline(opt), "recurrent warp pipeline");
        check(core.load_param((directory / "frame_core.param").string().c_str()),
              "frame core parameter loading");
        check(core.load_model((directory / "frame_core.bin").string().c_str()),
              "frame core weight loading");
        if (vulkan)
            for (const ncnn::Layer *layer : core.layers())
                if (layer->type != "Input" && !layer->support_vulkan)
                    throw std::runtime_error("frame core has a CPU-only operator: " + layer->type);
    }

    ~Impl() { warp.destroy_pipeline(opt); }

    void validate(const ncnn::Mat &frame) const
    {
        if (frame.empty() || frame.dims != 3 || frame.c != 3 || frame.elempack != 1 ||
            frame.elemsize != sizeof(float) || frame.w < 4 || frame.h < 4 || frame.w > 16384 ||
            frame.h > 16384 || frame.w % 4 || frame.h % 4)
            throw std::invalid_argument(
                "restorer requires planar FP32 RGB with dimensions divisible by four");
        if (frame.w > 32 && frame.h <= 32)
            throw std::invalid_argument("upstream SPyNet requires height > 32 when width > 32");
        if (width != 0 && (frame.w != width || frame.h != height))
            throw std::invalid_argument(
                "frame dimensions changed; reset the recurrent state first");
    }

    ncnn::Mat process_cpu(const ncnn::Mat &frame, ncnn::Mat *state_readback)
    {
        ncnn::Mat aligned;
        if (previous_cpu.empty())
        {
            aligned.create(frame.w, frame.h, 48, sizeof(float), 1);
            if (aligned.empty())
                throw std::runtime_error("initial state allocation failed");
            aligned.fill(0.f);
        }
        else
        {
            const ncnn::Mat displacement = flow.forward(frame, previous_cpu);
            std::vector<ncnn::Mat> warped(1);
            check(warp.forward({state_cpu, displacement}, warped, opt), "recurrent CPU warp");
            aligned = warped[0];
        }
        ncnn::Extractor extractor = core.create_extractor();
        check(extractor.input("in0", frame), "current frame input");
        check(extractor.input("in1", aligned), "aligned state input");
        ncnn::Mat output, next_state;
        check(extractor.extract("out0", output), "frame reconstruction");
        check(extractor.extract("out1", next_state), "next recurrent state");
        // Own the previous frame so callers can reuse their input buffer.
        ncnn::Mat owned_frame = frame.clone();
        if (owned_frame.empty())
            throw std::runtime_error("previous frame allocation failed");
        state_cpu = next_state;
        previous_cpu = owned_frame;
        if (state_readback)
            *state_readback = state_cpu.clone();
        return output;
    }

    ncnn::Mat process_gpu(const ncnn::Mat &frame, ncnn::Mat *state_readback)
    {
        ncnn::VkCompute command(flow.device());
        ncnn::VkMat current, aligned;
        command.record_upload(frame, current, opt);
        if (current.empty())
            throw std::runtime_error("frame upload allocation failed");
        if (previous_gpu.empty())
        {
            ncnn::Mat zero(frame.w, frame.h, 48, sizeof(float), 1);
            if (zero.empty())
                throw std::runtime_error("initial state allocation failed");
            zero.fill(0.f);
            // record_clone preserves pack1, unlike record_upload's auto-packing.
            command.record_clone(zero, aligned, opt);
            if (aligned.empty())
                throw std::runtime_error("initial GPU state allocation failed");
        }
        else
        {
            const ncnn::VkMat displacement = flow.record(current, previous_gpu, command, opt);
            std::vector<ncnn::VkMat> warped(1);
            check(warp.forward({state_gpu, displacement}, warped, command, opt),
                  "recurrent Vulkan warp");
            aligned = warped[0];
        }
        ncnn::Extractor extractor = core.create_extractor();
        extractor.set_blob_vkallocator(blobs.get());
        extractor.set_workspace_vkallocator(blobs.get());
        extractor.set_staging_vkallocator(staging.get());
        check(extractor.input("in0", current), "current GPU frame input");
        check(extractor.input("in1", aligned), "aligned GPU state input");
        ncnn::VkMat output_gpu, packed_state, next_state;
        check(extractor.extract("out0", output_gpu, command), "Vulkan frame reconstruction");
        check(extractor.extract("out1", packed_state, command), "Vulkan recurrent state");
        flow.device()->convert_packing(packed_state, next_state, 1, command, opt);
        if (next_state.empty() || next_state.c != 48 || next_state.w != frame.w ||
            next_state.h != frame.h || next_state.elempack != 1)
            throw std::runtime_error("invalid recurrent feature state");
        ncnn::Mat output;
        command.record_download(output_gpu, output, opt);
        if (state_readback)
            command.record_download(next_state, *state_readback, opt);
        check(command.submit_and_wait(), "Vulkan frame submission");
        previous_gpu = current;
        state_gpu = next_state;
        return output;
    }
};

RecurrentRestorer::RecurrentRestorer(const std::filesystem::path &model_directory, bool use_vulkan,
                                     int device_index, int cpu_threads)
    : impl_(std::make_unique<Impl>(model_directory, use_vulkan, device_index, cpu_threads))
{
}
RecurrentRestorer::~RecurrentRestorer() = default;

ncnn::Mat RecurrentRestorer::process(const ncnn::Mat &frame, ncnn::Mat *state_readback)
{
    impl_->validate(frame);
    ncnn::Mat output = impl_->flow.device() ? impl_->process_gpu(frame, state_readback)
                                            : impl_->process_cpu(frame, state_readback);
    impl_->width = frame.w;
    impl_->height = frame.h;
    return output;
}

void RecurrentRestorer::reset()
{
    impl_->previous_cpu.release();
    impl_->state_cpu.release();
    impl_->previous_gpu.release();
    impl_->state_gpu.release();
    impl_->width = impl_->height = 0;
}

} // namespace rvf
