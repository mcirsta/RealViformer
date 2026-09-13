#include <net.h>

#include <algorithm>
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
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    for (int channel = 0; channel < channels; ++channel)
    {
        float* destination = result.channel(channel);
        std::memcpy(
            destination,
            values.data() + static_cast<std::size_t>(channel) * plane,
            plane * sizeof(float));
    }
    return result;
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
                  << " SPYNET.param SPYNET.bin FIXTURE_DIRECTORY [--vulkan]\n";
        return 2;
    }

    try
    {
        constexpr int width = 19;
        constexpr int height = 13;
        constexpr int input_channels = 8;
        constexpr int output_channels = 2;
        const fs::path fixture_directory = argv[3];
        const bool use_vulkan = argc == 5 && std::string(argv[4]) == "--vulkan";
        if (argc == 5 && !use_vulkan)
            throw std::runtime_error("the only supported optional argument is --vulkan");

        GpuInstance gpu_instance(use_vulkan);
        ncnn::Net network;
        network.opt.use_vulkan_compute = use_vulkan;
        network.opt.use_packing_layout = false;
        network.opt.use_fp16_packed = false;
        network.opt.use_fp16_storage = false;
        network.opt.use_fp16_arithmetic = false;
        network.opt.use_fp16_uniform = false;
        network.opt.use_bf16_packed = false;
        network.opt.use_bf16_storage = false;
        network.opt.use_winograd_convolution = false;
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

        ncnn::Extractor extractor = network.create_extractor();
        for (int level = 0; level < 6; ++level)
        {
            const std::string name = "in" + std::to_string(level);
            const ncnn::Mat input = load_mat(
                fixture_directory / (name + ".f32"),
                width,
                height,
                input_channels);
            if (extractor.input(name.c_str(), input) != 0)
                throw std::runtime_error("could not set " + name);
        }

        double maximum_error = 0.0;
        for (int level = 0; level < 6; ++level)
        {
            const std::string name = "out" + std::to_string(level);
            ncnn::Mat output;
            if (extractor.extract(name.c_str(), output) != 0)
                throw std::runtime_error("could not extract " + name);
            const std::vector<float> expected = read_floats(
                fixture_directory / (name + ".f32"),
                static_cast<std::size_t>(width) * height * output_channels);
            if (output.w != width || output.h != height
                || output.c != output_channels || output.elempack != 1)
                throw std::runtime_error(
                    name + " shape mismatch: got " + std::to_string(output.w)
                    + "x" + std::to_string(output.h) + "x"
                    + std::to_string(output.c) + " pack"
                    + std::to_string(output.elempack));
            double level_error = 0.0;
            for (int channel = 0; channel < output_channels; ++channel)
            {
                const ncnn::Mat actual_channel = output.channel(channel);
                for (int row = 0; row < height; ++row)
                {
                    const float* actual = actual_channel.row(row);
                    const std::size_t offset =
                        (static_cast<std::size_t>(channel) * height + row) * width;
                    for (int column = 0; column < width; ++column)
                    {
                        if (!std::isfinite(actual[column]) ||
                            !std::isfinite(expected[offset + column]))
                            throw std::runtime_error(name + " contains a nonfinite value");
                        level_error = std::max(
                            level_error,
                            std::abs(static_cast<double>(
                                actual[column] - expected[offset + column])));
                    }
                }
            }
            maximum_error = std::max(maximum_error, level_error);
            std::cout << name << ": max error=" << level_error << '\n';
        }

        const double tolerance = use_vulkan ? 2e-3 : 2e-4;
        if (maximum_error > tolerance)
            throw std::runtime_error("SPyNet stage error exceeds tolerance");
        std::cout << "PASS: all six SPyNet refinement stages agree with PyTorch\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
