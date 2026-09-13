#include "recurrent_restorer.hpp"

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

double compare(const ncnn::Mat &actual, const ncnn::Mat &expected)
{
    if (actual.w != expected.w || actual.h != expected.h || actual.c != expected.c ||
        actual.elempack != 1 || actual.elemsize != sizeof(float))
        throw std::runtime_error("output shape/type mismatch");
    double maximum = 0.0;
    for (int ch = 0; ch < actual.c; ++ch)
        for (int y = 0; y < actual.h; ++y)
            for (int x = 0; x < actual.w; ++x)
            {
                const float a = actual.channel(ch).row(y)[x], e = expected.channel(ch).row(y)[x];
                if (!std::isfinite(a) || !std::isfinite(e))
                    throw std::runtime_error("nonfinite output");
                maximum = std::max(maximum, std::abs(double(a) - e));
            }
    return maximum;
}

void save(const fs::path &path, const ncnn::Mat &tensor)
{
    std::ofstream file(path, std::ios::binary);
    for (int ch = 0; ch < tensor.c; ++ch)
        file.write(reinterpret_cast<const char *>(static_cast<const float *>(tensor.channel(ch))),
                   static_cast<std::streamsize>(tensor.w) * tensor.h * sizeof(float));
    if (!file)
        throw std::runtime_error("could not write " + path.string());
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc < 3)
            throw std::invalid_argument("usage: rvf-sequence-test MODEL_DIRECTORY FIXTURE "
                                        "[--vulkan] [--device INDEX] [--save OUTPUT_DIRECTORY]");
        bool vulkan = false;
        int device_index = 0;
        fs::path save_directory;
        for (int i = 3; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--vulkan")
                vulkan = true;
            else if (arg == "--device" && i + 1 < argc)
                device_index = std::stoi(argv[++i]);
            else if (arg == "--save" && i + 1 < argc)
                save_directory = argv[++i];
            else
                throw std::invalid_argument("unknown option: " + arg);
        }
        const fs::path fixture = argv[2];
        std::ifstream shape(fixture / "shape.txt");
        int width = 0, height = 0, frames = 0;
        // Full-frame memory enforcement is still pending. Keep this validation
        // executable bounded until the allocator enforces the live VRAM budget.
        if (!(shape >> width >> height >> frames) || width < 4 || height < 4 || width > 128 ||
            height > 128 || frames < 2 || frames > 100)
            throw std::runtime_error(
                "validation requires 4..128-pixel dimensions and 2..100 frames");
        if (!save_directory.empty())
            fs::create_directories(save_directory);
        GpuInstance instance(vulkan);
        rvf::RecurrentRestorer restorer(argv[1], vulkan, device_index);
        double maximum_output = 0.0, maximum_state = 0.0;
        for (int index = 0; index < frames; ++index)
        {
            const std::string suffix = std::to_string(index) + ".f32";
            ncnn::Mat input = load(fixture / ("in" + suffix), width, height, 3), state;
            const auto start = std::chrono::steady_clock::now();
            const ncnn::Mat output = restorer.process(input, &state);
            const double milliseconds =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            const double output_error =
                compare(output, load(fixture / ("out" + suffix), width * 4, height * 4, 3));
            const double state_error =
                compare(state, load(fixture / ("state" + suffix), width, height, 48));
            maximum_output = std::max(maximum_output, output_error);
            maximum_state = std::max(maximum_state, state_error);
            std::cout << (vulkan ? "Vulkan" : "CPU") << " frame " << index << ": " << milliseconds
                      << " ms, output max=" << output_error << " state max=" << state_error << '\n';
            if (!save_directory.empty())
                save(save_directory / ("out" + suffix), output);
            // Callers may reuse or modify their input and state-readback buffers.
            input.fill(-7.f);
            state.fill(-9.f);
        }
        // Reset must reproduce frame zero, independent of the prior sequence.
        restorer.reset();
        ncnn::Mat state;
        const ncnn::Mat replay =
            restorer.process(load(fixture / "in0.f32", width, height, 3), &state);
        const double reset_error =
            compare(replay, load(fixture / "out0.f32", width * 4, height * 4, 3));
        maximum_output = std::max(maximum_output, reset_error);
        maximum_state = std::max(maximum_state,
                                 compare(state, load(fixture / "state0.f32", width, height, 48)));
        std::cout << "reset output max=" << reset_error << '\n';
        if (maximum_output > 1e-4 || maximum_state > 2e-4)
            throw std::runtime_error("sequence error exceeds image/state tolerances (1e-4 / 2e-4)");
        std::cout << "PASS: full recurrent restoration and reset agree with PyTorch\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
