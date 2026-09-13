#include "gelu.hpp"

#include <allocator.h>
#include <gpu.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
struct GpuInstance
{
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
    bool enabled;
};
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--vulkan"))
            throw std::invalid_argument("usage: rvf-gelu-test [--vulkan]");
        const bool gpu = argc == 2;
        GpuInstance instance(gpu);
        if (gpu && ncnn::get_gpu_count() == 0)
            throw std::runtime_error("no Vulkan device");
        const ncnn::VulkanDevice *device = gpu ? ncnn::get_gpu_device(0) : nullptr;
        for (int fast : {0, 1})
        {
            rvf::GeluLayer layer;
            ncnn::ParamDict params;
            params.set(0, fast);
            layer.load_param(params);
            layer.vkdev = device;
            ncnn::Option opt;
            opt.num_threads = 1;
            opt.use_vulkan_compute = gpu;
            opt.use_fp16_storage = opt.use_fp16_packed = opt.use_fp16_arithmetic = false;
            opt.use_fp16_uniform = false;
            ncnn::Mat input(257, 1, 5, sizeof(float), 1);
            for (int ch = 0; ch < input.c; ++ch)
                for (int x = 0; x < input.w; ++x)
                    input.channel(ch).row(0)[x] = -12.f + 24.f * (ch * input.w + x) / 1284.f;
            ncnn::Mat output = input.clone();
            if (layer.create_pipeline(opt) != 0)
                throw std::runtime_error("GELU pipeline failed");
            if (gpu)
            {
                ncnn::VkBlobAllocator blobs(device);
                ncnn::VkStagingAllocator staging(device);
                opt.blob_vkallocator = opt.workspace_vkallocator = &blobs;
                opt.staging_vkallocator = &staging;
                ncnn::VkCompute command(device);
                ncnn::VkMat tensor;
                command.record_clone(input, tensor, opt);
                if (layer.forward_inplace(tensor, command, opt) != 0)
                    throw std::runtime_error("Vulkan GELU failed");
                command.record_download(tensor, output, opt);
                if (command.submit_and_wait() != 0)
                    throw std::runtime_error("GELU submission failed");
            }
            else if (layer.forward_inplace(output, opt) != 0)
                throw std::runtime_error("CPU GELU failed");
            double maximum = 0.0;
            for (int ch = 0; ch < input.c; ++ch)
                for (int x = 0; x < input.w; ++x)
                {
                    const double v = input.channel(ch).row(0)[x];
                    const double reference =
                        fast ? 0.5 * v *
                                   (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) *
                                                  (v + 0.044715 * v * v * v)))
                             : 0.5 * v * std::erfc(-v / std::sqrt(2.0));
                    const float actual = output.channel(ch).row(0)[x];
                    if (!std::isfinite(actual))
                        throw std::runtime_error("nonfinite GELU value");
                    maximum = std::max(maximum, std::abs(reference - actual));
                }
            std::cout << (gpu ? "Vulkan" : "CPU") << " GELU fast=" << fast
                      << " max error=" << maximum << '\n';
            if (maximum > 2e-6)
                throw std::runtime_error("GELU error exceeds tolerance");
        }
        std::cout << "PASS: GELU agrees with the double-precision reference\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
