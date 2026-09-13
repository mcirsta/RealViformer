#include "spynet.hpp"
#include "flow_warp.hpp"

#include <gpu.h>
#include <pipeline.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace rvf
{
namespace
{

enum Operation
{
    NormalizeResize,
    Pool,
    UpsampleFlow,
    ResizeFlow,
    Concat,
    Add,
    Zero
};

// All auxiliary SPyNet operations share one FP32/pack1 pipeline. The flow
// upsampler replicates a trailing row/column AFTER a 2x align_corners resize,
// which differs from directly resizing to an odd target size.
constexpr char shader[] = R"glsl(
#version 450
layout(binding=0) readonly buffer A { float a[]; };
layout(binding=1) readonly buffer B { float b[]; };
layout(binding=2) readonly buffer C { float c[]; };
layout(binding=3) writeonly buffer D { float d[]; };
layout(push_constant) uniform parameter {
    int op; int iw; int ih; int acs; int bcs; int ccs;
    int ow; int oh; int channels; int ocs;
} p;

float sample_a(int ch, int x, int y) {
    return a[ch*p.acs + clamp(y,0,p.ih-1)*p.iw + clamp(x,0,p.iw-1)];
}
float resize_a(int ch, int x, int y) {
    float sx, sy;
    if (p.op == 2) {
        sx = float(min(x,2*p.iw-1))*float(p.iw-1)/float(2*p.iw-1);
        sy = float(min(y,2*p.ih-1))*float(p.ih-1)/float(2*p.ih-1);
    } else {
        sx = max((float(x)+0.5)*(float(p.iw)/float(p.ow))-0.5,0.0);
        sy = max((float(y)+0.5)*(float(p.ih)/float(p.oh))-0.5,0.0);
    }
    int x0=int(floor(sx)), y0=int(floor(sy));
    float dx=sx-float(x0), dy=sy-float(y0);
    return mix(mix(sample_a(ch,x0,y0),sample_a(ch,x0+1,y0),dx),
               mix(sample_a(ch,x0,y0+1),sample_a(ch,x0+1,y0+1),dx),dy);
}
void main() {
    int x=int(gl_GlobalInvocationID.x), y=int(gl_GlobalInvocationID.y);
    int ch=int(gl_GlobalInvocationID.z);
    if(x>=p.ow || y>=p.oh || ch>=p.channels) return;
    int pixel=y*p.ow+x;
    float v=0.0;
    if(p.op==0) {
        const float mean[3]=float[3](0.485,0.456,0.406);
        const float stddev[3]=float[3](0.229,0.224,0.225);
        v=(resize_a(ch,x,y)-mean[ch])/stddev[ch];
    } else if(p.op==1) {
        v=((sample_a(ch,x*2,y*2)+sample_a(ch,x*2+1,y*2))
           +sample_a(ch,x*2,y*2+1)+sample_a(ch,x*2+1,y*2+1))*0.25;
    } else if(p.op==2) {
        v=resize_a(ch,x,y)*2.0;
    } else if(p.op==3) {
        v=resize_a(ch,x,y)*(ch==0?float(p.ow)/float(p.iw):float(p.oh)/float(p.ih));
    } else if(p.op==4) {
        if(ch<3) v=a[ch*p.acs+pixel];
        else if(ch<6) v=b[(ch-3)*p.bcs+pixel];
        else v=c[(ch-6)*p.ccs+pixel];
    } else if(p.op==5) {
        v=a[ch*p.acs+pixel]+b[ch*p.bcs+pixel];
    }
    d[ch*p.ocs+pixel]=v;
}
)glsl";

void check(int result, const std::string &action)
{
    if (result != 0)
        throw std::runtime_error(action + " failed (" + std::to_string(result) + ")");
}

template <class Tensor> void validate_rgb_pair(const Tensor &reference, const Tensor &support)
{
    auto valid = [](const Tensor &tensor)
    {
        return !tensor.empty() && tensor.n == 1 && tensor.dims == 3 && tensor.c == 3 && tensor.elempack == 1 &&
               tensor.elemsize == sizeof(float) && tensor.w > 0 && tensor.h > 0 &&
               tensor.w <= 16384 && tensor.h <= 16384;
    };
    if (!valid(reference) || !valid(support) || reference.w != support.w ||
        reference.h != support.h)
        throw std::invalid_argument(
            "SPyNet requires matching planar FP32 RGB inputs (1..16384 pixels per axis)");
    const int width = (reference.w + 31) / 32 * 32;
    const int height = (reference.h + 31) / 32 * 32;
    // Preserve the upstream width-dependent pyramid depth. Its first flow
    // tensor must have a nonzero height; reject unsupported short-wide pairs.
    if (width > 32 && height < 64)
        throw std::invalid_argument("upstream SPyNet requires height > 32 when width > 32");
}

float sample(const ncnn::Mat &tensor, int channel, int x, int y)
{
    return tensor.channel(channel).row(
        std::clamp(y, 0, tensor.h - 1))[std::clamp(x, 0, tensor.w - 1)];
}

float resize(const ncnn::Mat &input, int channel, int x, int y, int width, int height,
             bool upsample)
{
    const float sx =
        upsample ? float(std::min(x, 2 * input.w - 1)) * float(input.w - 1) / float(2 * input.w - 1)
                 : std::max((x + 0.5f) * (float(input.w) / width) - 0.5f, 0.f);
    const float sy =
        upsample ? float(std::min(y, 2 * input.h - 1)) * float(input.h - 1) / float(2 * input.h - 1)
                 : std::max((y + 0.5f) * (float(input.h) / height) - 0.5f, 0.f);
    const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
    const float dx = sx - x0, dy = sy - y0;
    const float top =
        sample(input, channel, x0, y0) * (1.f - dx) + sample(input, channel, x0 + 1, y0) * dx;
    const float bottom = sample(input, channel, x0, y0 + 1) * (1.f - dx) +
                         sample(input, channel, x0 + 1, y0 + 1) * dx;
    return top * (1.f - dy) + bottom * dy;
}

} // namespace

class SpyNet::Impl
{
  public:
    ncnn::Net network;
    FlowWarpLayer warp{FlowWarpPadding::Border};
    std::unique_ptr<ncnn::Pipeline> pipeline;
    const ncnn::VulkanDevice *vkdev = nullptr;

    ~Impl() { warp.destroy_pipeline(network.opt); }

    ncnn::Mat operation(Operation op, const ncnn::Mat &a, const ncnn::Mat &b, const ncnn::Mat &c,
                        int width, int height, int channels, ncnn::VkCompute *,
                        const ncnn::Option &opt) const
    {
        ncnn::Mat out(width, height, channels, sizeof(float), 1, opt.blob_allocator);
        if (out.empty())
            throw std::runtime_error("SPyNet CPU allocation failed");
        constexpr float mean[] = {0.485f, 0.456f, 0.406f};
        constexpr float stddev[] = {0.229f, 0.224f, 0.225f};
#pragma omp parallel for num_threads(opt.num_threads)
        for (int ch = 0; ch < channels; ++ch)
        {
            for (int y = 0; y < height; ++y)
            {
                float *row = out.channel(ch).row(y);
                for (int x = 0; x < width; ++x)
                {
                    float v = 0.f;
                    switch (op)
                    {
                    case NormalizeResize:
                        v = (resize(a, ch, x, y, width, height, false) - mean[ch]) / stddev[ch];
                        break;
                    case Pool:
                        v = (sample(a, ch, 2 * x, 2 * y) + sample(a, ch, 2 * x + 1, 2 * y) +
                             sample(a, ch, 2 * x, 2 * y + 1) +
                             sample(a, ch, 2 * x + 1, 2 * y + 1)) *
                            0.25f;
                        break;
                    case UpsampleFlow:
                        v = resize(a, ch, x, y, width, height, true) * 2.f;
                        break;
                    case ResizeFlow:
                        v = resize(a, ch, x, y, width, height, false) *
                            (ch == 0 ? float(width) / a.w : float(height) / a.h);
                        break;
                    case Concat:
                        v = ch < 3   ? sample(a, ch, x, y)
                            : ch < 6 ? sample(b, ch - 3, x, y)
                                     : sample(c, ch - 6, x, y);
                        break;
                    case Add:
                        v = sample(a, ch, x, y) + sample(b, ch, x, y);
                        break;
                    case Zero:
                        break;
                    }
                    row[x] = v;
                }
            }
        }
        return out;
    }

    ncnn::VkMat operation(Operation op, const ncnn::VkMat &a, const ncnn::VkMat &b,
                          const ncnn::VkMat &c, int width, int height, int channels,
                          ncnn::VkCompute *command, const ncnn::Option &opt) const
    {
        ncnn::VkMat out(width, height, channels, sizeof(float), 1, opt.blob_vkallocator);
        if (out.empty())
            throw std::runtime_error("SPyNet Vulkan allocation failed");
        std::vector<ncnn::vk_constant_type> params(10);
        params[0].i = op;
        params[1].i = a.w;
        params[2].i = a.h;
        params[3].i = static_cast<int>(a.cstep);
        params[4].i = static_cast<int>(b.cstep);
        params[5].i = static_cast<int>(c.cstep);
        params[6].i = width;
        params[7].i = height;
        params[8].i = channels;
        params[9].i = static_cast<int>(out.cstep);
        command->record_pipeline(pipeline.get(), {a, b, c, out}, params, out);
        return out;
    }

    template <class Tensor>
    Tensor run(const Tensor &reference, const Tensor &support, ncnn::VkCompute *command,
               const ncnn::Option &opt) const
    {
        validate_rgb_pair(reference, support);
        const int width = (reference.w + 31) / 32 * 32;
        const int height = (reference.h + 31) / 32 * 32;
        const int depth = width > 32 ? 5 : 4;
        const Tensor empty;
        auto op = [&](Operation kind, const Tensor &a, int w, int h, int channels,
                      const Tensor &b = Tensor(), const Tensor &c = Tensor())
        { return operation(kind, a, b, c, w, h, channels, command, opt); };
        std::vector<Tensor> refs{op(NormalizeResize, reference, width, height, 3)};
        std::vector<Tensor> supps{op(NormalizeResize, support, width, height, 3)};
        for (int i = 0; i < depth; ++i)
        {
            refs.push_back(op(Pool, refs.back(), refs.back().w / 2, refs.back().h / 2, 3));
            supps.push_back(op(Pool, supps.back(), supps.back().w / 2, supps.back().h / 2, 3));
        }
        Tensor flow = op(Zero, empty, refs.back().w / 2, refs.back().h / 2, 2);
        for (int level = 0; level <= depth; ++level)
        {
            const Tensor &ref = refs[depth - level];
            const Tensor &supp = supps[depth - level];
            Tensor up = op(UpsampleFlow, flow, ref.w, ref.h, 2);
            std::vector<Tensor> warped(1);
            if constexpr (std::is_same_v<Tensor, ncnn::VkMat>)
                check(warp.forward({supp, up}, warped, *command, opt), "SPyNet Vulkan warp");
            else
                check(warp.forward({supp, up}, warped, opt), "SPyNet CPU warp");
            Tensor input = op(Concat, ref, ref.w, ref.h, 8, warped[0], up);
            ncnn::Extractor extractor = network.create_extractor();
            if constexpr (std::is_same_v<Tensor, ncnn::VkMat>)
            {
                extractor.set_blob_vkallocator(opt.blob_vkallocator);
                extractor.set_workspace_vkallocator(opt.workspace_vkallocator);
                extractor.set_staging_vkallocator(opt.staging_vkallocator);
            }
            const std::string input_name = "in" + std::to_string(level);
            const std::string output_name = "out" + std::to_string(level);
            check(extractor.input(input_name.c_str(), input), "SPyNet stage input");
            Tensor residual;
            if constexpr (std::is_same_v<Tensor, ncnn::VkMat>)
                check(extractor.extract(output_name.c_str(), residual, *command),
                      "SPyNet Vulkan stage");
            else
                check(extractor.extract(output_name.c_str(), residual), "SPyNet CPU stage");
            if (residual.w != ref.w || residual.h != ref.h || residual.c != 2 ||
                residual.elempack != 1 || residual.elemsize != sizeof(float))
                throw std::runtime_error("SPyNet stage output must be planar FP32 flow");
            flow = op(Add, residual, ref.w, ref.h, 2, up);
        }
        return op(ResizeFlow, flow, reference.w, reference.h, 2);
    }
};

SpyNet::SpyNet(const std::string &parameters, const std::string &weights, bool use_vulkan,
               int device_index, int cpu_threads)
    : impl_(std::make_unique<Impl>())
{
    ncnn::Option &opt = impl_->network.opt;
    opt.use_vulkan_compute = use_vulkan;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_fp16_uniform = false;
    opt.use_bf16_storage = false;
    opt.use_bf16_packed = false;
    opt.use_winograd_convolution = false;
    opt.num_threads = std::max(cpu_threads, 1);
    if (use_vulkan)
    {
        if (device_index < 0 || device_index >= ncnn::get_gpu_count())
            throw std::invalid_argument("invalid Vulkan compute device index");
        impl_->vkdev = ncnn::get_gpu_device(device_index);
        impl_->network.set_vulkan_device(impl_->vkdev);
        impl_->warp.vkdev = impl_->vkdev;
        std::vector<std::uint32_t> spirv;
        check(ncnn::compile_spirv_module(shader, sizeof(shader) - 1, opt, spirv),
              "SPyNet shader compilation");
        impl_->pipeline = std::make_unique<ncnn::Pipeline>(impl_->vkdev);
        impl_->pipeline->set_local_size_xyz(8, 8, 1);
        check(impl_->pipeline->create(spirv.data(), spirv.size() * sizeof(std::uint32_t), {}),
              "SPyNet pipeline");
    }
    check(impl_->warp.create_pipeline(opt), "SPyNet warp pipeline");
    check(impl_->network.load_param(parameters.c_str()), "SPyNet parameter loading");
    check(impl_->network.load_model(weights.c_str()), "SPyNet weight loading");
    if (use_vulkan)
        for (const ncnn::Layer *layer : impl_->network.layers())
            if (layer->type != "Input" && !layer->support_vulkan)
                throw std::runtime_error("SPyNet graph contains a CPU-only operator: " +
                                         layer->type);
}

SpyNet::~SpyNet() = default;

ncnn::Mat SpyNet::forward(const ncnn::Mat &reference, const ncnn::Mat &support) const
{
    if (impl_->vkdev)
        throw std::logic_error("use record() for a Vulkan SPyNet instance");
    return impl_->run(reference, support, nullptr, impl_->network.opt);
}

ncnn::VkMat SpyNet::record(const ncnn::VkMat &reference, const ncnn::VkMat &support,
                           ncnn::VkCompute &command, const ncnn::Option &option) const
{
    if (!impl_->vkdev || !option.blob_vkallocator || !option.workspace_vkallocator ||
        !option.staging_vkallocator || option.use_fp16_storage || option.use_fp16_packed ||
        option.use_fp16_arithmetic || option.use_bf16_storage || option.use_bf16_packed)
        throw std::invalid_argument("SPyNet record requires Vulkan allocators and FP32 options");
    return impl_->run(reference, support, &command, option);
}

const ncnn::Option &SpyNet::option() const { return impl_->network.opt; }
const ncnn::VulkanDevice *SpyNet::device() const { return impl_->vkdev; }

} // namespace rvf
