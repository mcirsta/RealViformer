#include "last_axis_reduction.hpp"

#include <algorithm>
#include <allocator.h>
#include <chrono>
#include <cmath>
#include <gpu.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct GpuInstance
{
    bool enabled;
    explicit GpuInstance(bool enabled) : enabled(enabled)
    {
        if (enabled)
            ncnn::create_gpu_instance();
    }
    ~GpuInstance()
    {
        if (enabled)
            ncnn::destroy_gpu_instance();
    }
};

ncnn::ParamDict parameters(int axis, bool mean)
{
    ncnn::ParamDict pd;
    ncnn::Mat axes(1);
    static_cast<int *>(axes)[0] = axis;
    pd.set(0, mean ? 3 : 0);
    pd.set(1, 0);
    pd.set(2, 0.75f);
    pd.set(3, axes);
    pd.set(4, 1);
    pd.set(5, 1);
    return pd;
}

void test(int width, int rows, int channels, bool mean, bool vulkan)
{
    ncnn::Mat input(width, rows, channels, sizeof(float), 1);
    for (int c = 0; c < channels; ++c)
        for (int r = 0; r < rows; ++r)
            for (int x = 0; x < width; ++x)
                input.channel(c).row(r)[x] =
                    (1.f + 0.7f * std::sin(float(x % 137) + r * 0.13f + c)) / width;
    rvf::LastAxisReductionLayer layer;
    if (layer.load_param(parameters(2, mean)))
        throw std::runtime_error("parameter loading failed");
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = vulkan;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = opt.use_fp16_uniform =
        false;
    opt.use_bf16_packed = opt.use_bf16_storage = false;
    ncnn::Mat result;
    const auto start = std::chrono::steady_clock::now();
    if (vulkan)
    {
        if (ncnn::get_gpu_count() == 0)
            throw std::runtime_error("no Vulkan device");
        const auto *device = ncnn::get_gpu_device(0);
        ncnn::VkBlobAllocator blobs(device);
        ncnn::VkStagingAllocator staging(device);
        opt.blob_vkallocator = opt.workspace_vkallocator = &blobs;
        opt.staging_vkallocator = &staging;
        layer.vkdev = device;
        if (layer.create_pipeline(opt))
            throw std::runtime_error("pipeline creation failed");
        {
            ncnn::VkCompute cmd(device);
            ncnn::VkMat in, out;
            cmd.record_clone(input, in, opt);
            if (layer.forward(in, out, cmd, opt))
                throw std::runtime_error("GPU reduction failed");
            cmd.record_download(out, result, opt);
            if (cmd.submit_and_wait())
                throw std::runtime_error("GPU submission failed");
        }
        layer.destroy_pipeline(opt);
    }
    else if (layer.forward(input, result, opt))
        throw std::runtime_error("CPU reduction failed");
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (result.w != 1 || result.h != rows || result.c != channels || result.elempack != 1)
        throw std::runtime_error("reduction output shape mismatch");
    double maximum = 0.0;
    for (int c = 0; c < channels; ++c)
        for (int r = 0; r < rows; ++r)
        {
            double expected = 0.0;
            for (int x = 0; x < width; ++x)
                expected += input.channel(c).row(r)[x];
            expected *= mean ? 0.75 / width : 0.75;
            const float actual = result.channel(c).row(r)[0];
            if (!std::isfinite(actual))
                throw std::runtime_error("nonfinite reduction result");
            const double error = std::abs(actual - expected) / std::max(1e-12, std::abs(expected));
            maximum = std::max(maximum, error);
        }
    std::cout << width << 'x' << rows << 'x' << channels << (mean ? " mean" : " sum")
              << ": relative max=" << maximum << ", " << elapsed << " ms\n";
    if (maximum > (vulkan ? 2e-6 : 1e-7))
        throw std::runtime_error("reduction error exceeds tolerance");
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--vulkan"))
            throw std::invalid_argument("usage: rvf-reduction-test [--vulkan]");
        const bool vulkan = argc == 2;
        GpuInstance gpu(vulkan);
        for (bool mean : {false, true})
        {
            for (int width : {1, 3, 48, 65, 192, 2049, 307200})
                test(width, 3, 2, mean, vulkan);
            // Cross the guaranteed workgroup-count limit with a modest ~27 MB input.
            test(48, 70001, 2, mean, vulkan);
        }
        rvf::LastAxisReductionLayer invalid;
        auto pd = parameters(2, false);
        pd.set(0, 4);
        if (invalid.load_param(pd) == 0)
            throw std::runtime_error("unsupported op accepted");
        if (invalid.load_param(parameters(0, false)))
            throw std::runtime_error("valid axis failed loading");
        ncnn::Mat in(3, 2, 1, sizeof(float), 1), out;
        if (invalid.forward(in, out, ncnn::Option()) == 0)
            throw std::runtime_error("non-last axis accepted");
        std::cout << "PASS: reduction accuracy, large dispatch, and rejected unsupported modes\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
