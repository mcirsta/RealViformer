#pragma once

#include <mat.h>

#include <filesystem>
#include <memory>

namespace rvf
{

// Streaming, untiled FP32 reference backend. Keeps only the previous RGB frame
// and 48-channel feature state between calls; reset() marks a scene boundary.
// The caller must keep the ncnn GPU instance alive for a Vulkan restorer.
// Vulkan inputs are capped at 128x128 until tiled reconstruction and actual
// allocator-budget enforcement are implemented. This is not a VRAM guarantee.
class RecurrentRestorer final
{
  public:
    RecurrentRestorer(const std::filesystem::path &model_directory, bool use_vulkan,
                      int device_index = 0, int cpu_threads = 1);
    ~RecurrentRestorer();
    RecurrentRestorer(const RecurrentRestorer &) = delete;
    RecurrentRestorer &operator=(const RecurrentRestorer &) = delete;

    // Planar RGB FP32 [0,1], with spatial dimensions divisible by four.
    // The returned x4 image is FP32 and may extend outside [0,1].
    // Optional state readback is intended for numerical verification only.
    ncnn::Mat process(const ncnn::Mat &frame, ncnn::Mat *state_readback = nullptr);
    void reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rvf
