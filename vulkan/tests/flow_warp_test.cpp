#include "flow_warp.hpp"

#include <allocator.h>
#include <command.h>
#include <gpu.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class GpuInstance
{
public:
    explicit GpuInstance(bool enabled) : enabled_(enabled)
    {
        if (enabled_)
            ncnn::create_gpu_instance();
    }

    ~GpuInstance()
    {
        if (enabled_)
            ncnn::destroy_gpu_instance();
    }

private:
    bool enabled_;
};

ncnn::Mat make_feature(int width, int height, int channels)
{
    ncnn::Mat result(width, height, channels, sizeof(float), 1);
    for (int channel = 0; channel < channels; ++channel)
    {
        for (int y = 0; y < height; ++y)
        {
            float* row = result.channel(channel).row(y);
            for (int x = 0; x < width; ++x)
                row[x] = std::sin((channel + 1) * 0.31f + x * 0.17f - y * 0.23f);
        }
    }
    return result;
}

ncnn::Mat make_flow(int width, int height)
{
    ncnn::Mat result(width, height, 2, sizeof(float), 1);
    float* flow_x = result.channel(0);
    float* flow_y = result.channel(1);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const int pixel = y * width + x;
            flow_x[pixel] = 1.35f * std::sin(x * 0.41f + y * 0.07f) - 0.2f;
            flow_y[pixel] = 1.15f * std::cos(y * 0.37f - x * 0.11f) + 0.1f;
        }
    }
    return result;
}

double maximum_error(const ncnn::Mat& left, const ncnn::Mat& right)
{
    if (left.w != right.w || left.h != right.h || left.c != right.c)
        throw std::runtime_error("output shape mismatch: expected " + std::to_string(left.w) + "x" +
            std::to_string(left.h) + "x" + std::to_string(left.c) + ", got " +
            std::to_string(right.w) + "x" + std::to_string(right.h) + "x" +
            std::to_string(right.c) + " pack" + std::to_string(right.elempack));
    double maximum = 0.0;
    for (int channel = 0; channel < left.c; ++channel)
    {
        for (int y = 0; y < left.h; ++y)
        {
            const float* a = left.channel(channel).row(y);
            const float* b = right.channel(channel).row(y);
            for (int x = 0; x < left.w; ++x)
            {
                if (!std::isfinite(a[x]) || !std::isfinite(b[x]))
                    throw std::runtime_error("nonfinite warp result");
                maximum = std::max(maximum, std::abs(static_cast<double>(a[x] - b[x])));
            }
        }
    }
    return maximum;
}

ncnn::Mat run(
    const ncnn::Mat& feature,
    const ncnn::Mat& flow,
    rvf::FlowWarpPadding padding,
    bool use_vulkan)
{
    rvf::FlowWarpLayer layer(padding);
    ncnn::Option option;
    option.num_threads = 1;
    option.use_packing_layout = false;
    option.use_vulkan_compute = use_vulkan;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_fp16_uniform = false;
    option.use_bf16_packed = false;
    option.use_bf16_storage = false;

    if (!use_vulkan)
    {
        std::vector<ncnn::Mat> outputs(1);
        if (layer.forward({feature, flow}, outputs, option) != 0)
            throw std::runtime_error("CPU flow warp failed");
        return outputs[0];
    }

    if (ncnn::get_gpu_count() == 0)
        throw std::runtime_error("ncnn did not find a Vulkan compute device");
    const ncnn::VulkanDevice* device = ncnn::get_gpu_device(0);
    layer.vkdev = device;
    ncnn::VkAllocator* blob_allocator = device->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = device->acquire_staging_allocator();
    option.blob_vkallocator = blob_allocator;
    option.workspace_vkallocator = blob_allocator;
    option.staging_vkallocator = staging_allocator;

    if (layer.create_pipeline(option) != 0)
        throw std::runtime_error("could not create flow-warp pipeline");
    ncnn::Mat output;
    {
        ncnn::VkCompute command(device);
        ncnn::VkMat feature_gpu;
        ncnn::VkMat flow_gpu;
        // Keep pack1 even when the fixture has four channels.
        command.record_clone(feature, feature_gpu, option);
        command.record_clone(flow, flow_gpu, option);
        std::vector<ncnn::VkMat> outputs(1);
        if (layer.forward({feature_gpu, flow_gpu}, outputs, command, option) != 0)
            throw std::runtime_error("Vulkan flow warp failed");
        command.record_download(outputs[0], output, option);
        if (command.submit_and_wait() != 0)
            throw std::runtime_error("Vulkan flow-warp dispatch failed");
    }
    layer.destroy_pipeline(option);
    device->reclaim_blob_allocator(blob_allocator);
    device->reclaim_staging_allocator(staging_allocator);
    return output;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 3)
    {
        std::cerr << "usage: " << argv[0] << " [--vulkan] [FIXTURE_ROOT]\n";
        return 2;
    }

    try
    {
        bool use_vulkan = false;
        std::filesystem::path fixture_root;
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--vulkan")
                use_vulkan = true;
            else if (fixture_root.empty())
                fixture_root = argv[i];
            else
                throw std::invalid_argument("unexpected argument");
        GpuInstance gpu_instance(use_vulkan);
        const ncnn::Mat feature = make_feature(19, 13, 5);
        const ncnn::Mat flow = make_flow(feature.w, feature.h);

        // Independent analytical checks also run without external fixtures.
        const ncnn::Mat singleton = make_feature(1, 1, 4);
        ncnn::Mat singleton_flow(1, 1, 2, sizeof(float), 1);
        singleton_flow.fill(1e30f);
        ncnn::Mat identity_flow(feature.w, feature.h, 2, sizeof(float), 1);
        identity_flow.fill(0.f);

        for (const rvf::FlowWarpPadding padding : {
                 rvf::FlowWarpPadding::Zeros,
                 rvf::FlowWarpPadding::Border})
        {
            if (maximum_error(singleton, run(singleton, singleton_flow, padding, use_vulkan)) > 1e-7 ||
                maximum_error(feature, run(feature, identity_flow, padding, use_vulkan)) > 3e-6)
                throw std::runtime_error("analytical singleton/identity warp check failed");
            const ncnn::Mat reference = run(feature, flow, padding, false);
            const ncnn::Mat actual = run(feature, flow, padding, use_vulkan);
            const double error = maximum_error(reference, actual);
            std::cout << (padding == rvf::FlowWarpPadding::Zeros ? "zeros" : "border")
                      << ": max error=" << error << '\n';
            const double tolerance = use_vulkan ? 2e-5 : 0.0;
            if (error > tolerance)
                throw std::runtime_error("flow-warp error exceeds tolerance");
        }
        std::cout << "PASS: bilinear flow warp agrees with the scalar reference\n";
        if (!fixture_root.empty())
        {
            int cases = 0;
            for (const auto &entry : std::filesystem::directory_iterator(fixture_root))
            {
                if (!entry.is_directory())
                    continue;
                int width = 0, height = 0, channels = 0;
                std::ifstream shape(entry.path() / "shape.txt");
                if (!(shape >> width >> height >> channels) || width < 1 || height < 1 ||
                    width > 640 || height > 480 || channels < 1 || channels > 48)
                    throw std::runtime_error("invalid warp fixture shape");
                auto load = [&](const std::string &name, int ch)
                {
                    std::ifstream file(entry.path() / (name + ".f32"), std::ios::binary | std::ios::ate);
                    const std::streamoff bytes = std::streamoff(width) * height * sizeof(float);
                    if (!file || file.tellg() != bytes * ch)
                        throw std::runtime_error("invalid warp fixture size");
                    file.seekg(0);
                    ncnn::Mat tensor(width, height, ch, sizeof(float), 1);
                    for (int c = 0; c < ch; ++c)
                        file.read(reinterpret_cast<char *>(static_cast<float *>(tensor.channel(c))), bytes);
                    if (!file)
                        throw std::runtime_error("could not read warp fixture");
                    return tensor;
                };
                for (const auto padding : {rvf::FlowWarpPadding::Zeros, rvf::FlowWarpPadding::Border})
                {
                    const std::string name = padding == rvf::FlowWarpPadding::Zeros ? "zeros" : "border";
                    const ncnn::Mat actual = run(load("feature", channels), load("flow", 2), padding, use_vulkan);
                    if (const char *dump = std::getenv("RVF_WARP_TEST_DUMP"))
                    {
                        const auto directory = std::filesystem::path(dump) / entry.path().filename();
                        std::filesystem::create_directories(directory);
                        std::ofstream file(directory / (name + ".f32"), std::ios::binary);
                        for (int c = 0; c < channels; ++c)
                            file.write(reinterpret_cast<const char *>(static_cast<const float *>(actual.channel(c))),
                                       std::streamsize(width) * height * sizeof(float));
                        if (!file)
                            throw std::runtime_error("could not write warp diagnostic");
                    }
                    const double error = maximum_error(load(name, channels), actual);
                    std::cout << entry.path().filename() << ' ' << name << ": PyTorch max=" << error << '\n';
                    if (error > 3e-6)
                        throw std::runtime_error("warp differs from PyTorch by more than 3e-6");
                }
                ++cases;
            }
            if (cases != 6)
                throw std::runtime_error("expected six independent warp fixtures");
            std::cout << "PASS: all six warp cases agree with PyTorch\n";
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
