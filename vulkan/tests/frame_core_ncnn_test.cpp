#include <net.h>

#include "attention_mask_descriptor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<float> read_floats(const fs::path& path, std::size_t count)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("could not open " + path.string());
    const auto bytes = stream.tellg();
    if (bytes != static_cast<std::streamoff>(count * sizeof(float)))
        throw std::runtime_error("unexpected byte count in " + path.string());
    std::vector<float> values(count);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!stream)
        throw std::runtime_error("could not read " + path.string());
    return values;
}

ncnn::Mat load_mat(const fs::path& path, int width, int height, int channels)
{
    const std::size_t count = static_cast<std::size_t>(width) * height * channels;
    const std::vector<float> values = read_floats(path, count);
    ncnn::Mat result(width, height, channels, sizeof(float), 1);
    std::memcpy(result.data, values.data(), count * sizeof(float));
    return result;
}

struct ErrorMetrics
{
    double maximum = 0.0;
    double mean = 0.0;
    double rmse = 0.0;
};

ErrorMetrics compare(const ncnn::Mat& actual, const std::vector<float>& expected)
{
    if (actual.elempack != 1 || actual.elemsize != sizeof(float))
        throw std::runtime_error("test output is not unpacked FP32");
    if (actual.total() != expected.size())
        throw std::runtime_error("test output element count does not match fixture");

    const float* values = static_cast<const float*>(actual.data);
    double sum = 0.0;
    double squared_sum = 0.0;
    double maximum = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        const double error = std::abs(static_cast<double>(values[index]) - expected[index]);
        maximum = std::max(maximum, error);
        sum += error;
        squared_sum += error * error;
    }
    return {maximum, sum / expected.size(), std::sqrt(squared_sum / expected.size())};
}

void print_metrics(const char* name, const ErrorMetrics& metrics)
{
    std::cout << name << ": max=" << metrics.maximum << " mean=" << metrics.mean
              << " rmse=" << metrics.rmse << '\n';
}

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

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5)
    {
        std::cerr << "usage: " << argv[0]
                  << " FRAME_CORE.param FRAME_CORE.bin FIXTURE_DIRECTORY [--vulkan]\n";
        return 2;
    }

    try
    {
        constexpr int width = 32;
        constexpr int height = 32;
        const fs::path fixture_directory = argv[3];
        const bool use_vulkan = argc == 5 && std::string(argv[4]) == "--vulkan";
        if (argc == 5 && !use_vulkan)
            throw std::runtime_error("the only supported optional argument is --vulkan");

        GpuInstance gpu_instance(use_vulkan);
        ncnn::Net network;
        network.register_custom_layer(
            "archs.realviformer_arch.AttentionMaskDescriptor",
            rvf::create_attention_mask_descriptor_layer);
        network.opt.use_vulkan_compute = use_vulkan;
        network.opt.use_packing_layout = false;
        network.opt.use_fp16_packed = false;
        network.opt.use_fp16_storage = false;
        network.opt.use_fp16_arithmetic = false;
        network.opt.use_bf16_storage = false;
        network.opt.use_winograd_convolution = false;
        network.opt.use_winograd23_convolution = false;
        network.opt.use_winograd43_convolution = false;
        network.opt.use_winograd63_convolution = false;
        network.opt.num_threads = 1;
        if (use_vulkan)
        {
            if (ncnn::get_gpu_count() == 0)
                throw std::runtime_error("ncnn did not find a Vulkan compute device");
            network.set_vulkan_device(0);
            std::cout << "backend: Vulkan GPU 0\n";
        }
        else
        {
            std::cout << "backend: CPU\n";
        }
        if (network.load_param(argv[1]) != 0)
            throw std::runtime_error("could not load ncnn parameter file");
        if (network.load_model(argv[2]) != 0)
            throw std::runtime_error("could not load ncnn model file");

        const ncnn::Mat current = load_mat(fixture_directory / "in0.f32", width, height, 3);
        const ncnn::Mat previous = load_mat(fixture_directory / "in1.f32", width, height, 48);

        ncnn::Extractor extractor = network.create_extractor();
        if (extractor.input("in0", current) != 0 || extractor.input("in1", previous) != 0)
            throw std::runtime_error("could not set ncnn inputs");
        ncnn::Mat output;
        ncnn::Mat state;
        const auto start = std::chrono::steady_clock::now();
        if (extractor.extract("out0", output) != 0 || extractor.extract("out1", state) != 0)
            throw std::runtime_error("ncnn inference failed");
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start);
        std::cout << "inference: " << elapsed.count() << " ms\n";

        const ErrorMetrics output_error = compare(
            output,
            read_floats(fixture_directory / "out0.f32", 3U * width * 4 * height * 4));
        const ErrorMetrics state_error = compare(
            state, read_floats(fixture_directory / "out1.f32", 48U * width * height));
        print_metrics("out0", output_error);
        print_metrics("out1", state_error);

        const double maximum_error = std::max(output_error.maximum, state_error.maximum);
        const double tolerance = use_vulkan ? 3e-3 : 2e-4;
        if (maximum_error > tolerance)
        {
            std::cerr << "FAIL: maximum error exceeds " << tolerance << '\n';
            return 1;
        }
        std::cout << "PASS: ncnn frame core agrees with PyTorch\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
