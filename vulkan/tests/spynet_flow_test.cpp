#include "spynet.hpp"

#include <allocator.h>
#include <gpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

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

ncnn::Mat load(const fs::path &path, int width, int height, int channels)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff plane_bytes = static_cast<std::streamoff>(width) * height * sizeof(float);
    if (!file || file.tellg() != channels * plane_bytes)
        throw std::runtime_error("wrong fixture size: " + path.string());
    file.seekg(0);
    ncnn::Mat result(width, height, channels, sizeof(float), 1);
    for (int ch = 0; ch < channels; ++ch)
        file.read(reinterpret_cast<char *>(static_cast<float *>(result.channel(ch))), plane_bytes);
    if (!file)
        throw std::runtime_error("could not read " + path.string());
    return result;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc < 4)
            throw std::invalid_argument("usage: rvf-spynet-flow-test SPYNET.param SPYNET.bin "
                                        "FIXTURE [--vulkan] [--device INDEX]");
        bool vulkan = false;
        int device_index = 0;
        for (int i = 4; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--vulkan")
                vulkan = true;
            else if (arg == "--device" && i + 1 < argc)
                device_index = std::stoi(argv[++i]);
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        const fs::path fixture = argv[3];
        std::ifstream shape(fixture / "shape.txt");
        int width = 0, height = 0;
        if (!(shape >> width >> height) || width < 1 || height < 1 || width > 16384 ||
            height > 16384)
            throw std::runtime_error("invalid fixture dimensions");
        const ncnn::Mat ref = load(fixture / "ref.f32", width, height, 3);
        const ncnn::Mat supp = load(fixture / "supp.f32", width, height, 3);
        const ncnn::Mat expected = load(fixture / "flow.f32", width, height, 2);
        GpuInstance instance(vulkan);
        rvf::SpyNet network(argv[1], argv[2], vulkan, device_index);
        ncnn::Mat actual;
        const auto start = std::chrono::steady_clock::now();
        if (vulkan)
        {
            ncnn::VkBlobAllocator blobs(network.device());
            ncnn::VkStagingAllocator staging(network.device());
            ncnn::Option opt = network.option();
            opt.blob_vkallocator = &blobs;
            opt.workspace_vkallocator = &blobs;
            opt.staging_vkallocator = &staging;
            ncnn::VkCompute command(network.device());
            ncnn::VkMat ref_gpu, supp_gpu;
            command.record_upload(ref, ref_gpu, opt);
            command.record_upload(supp, supp_gpu, opt);
            const ncnn::VkMat flow = network.record(ref_gpu, supp_gpu, command, opt);
            command.record_download(flow, actual, opt);
            if (command.submit_and_wait() != 0)
                throw std::runtime_error("Vulkan submission failed");
        }
        else
            actual = network.forward(ref, supp);
        const double milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        if (actual.w != width || actual.h != height || actual.c != 2 || actual.elempack != 1 ||
            actual.elemsize != sizeof(float))
            throw std::runtime_error("flow output shape/type mismatch");
        double maximum = 0.0, sum = 0.0;
        for (int ch = 0; ch < 2; ++ch)
        {
            for (int y = 0; y < height; ++y)
            {
                const float *a = actual.channel(ch).row(y);
                const float *e = expected.channel(ch).row(y);
                for (int x = 0; x < width; ++x)
                {
                    if (!std::isfinite(a[x]) || !std::isfinite(e[x]))
                        throw std::runtime_error("nonfinite flow value");
                    const double error = std::abs(double(a[x]) - e[x]);
                    maximum = std::max(maximum, error);
                    sum += error;
                }
            }
        }
        std::cout << (vulkan ? "Vulkan" : "CPU") << " " << width << "x" << height << ": "
                  << milliseconds << " ms, max=" << maximum
                  << " mean=" << sum / (2.0 * width * height) << " pixels\n";
        if (maximum > (vulkan ? 2e-3 : 5e-4))
            throw std::runtime_error("flow error exceeds tolerance");
        std::cout << "PASS: complete SPyNet optical flow agrees with PyTorch\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
