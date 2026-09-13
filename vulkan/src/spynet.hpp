#pragma once

#include <command.h>
#include <net.h>

#include <memory>
#include <string>

namespace rvf
{

// The caller owns the ncnn GPU instance, allocators, and command submission.
// RGB inputs are planar, unpacked FP32 in [0, 1]. Flow is in source pixels.
class SpyNet final
{
  public:
    SpyNet(const std::string &parameters, const std::string &weights, bool use_vulkan,
           int device_index = 0, int cpu_threads = 1);
    ~SpyNet();
    SpyNet(const SpyNet &) = delete;
    SpyNet &operator=(const SpyNet &) = delete;

    ncnn::Mat forward(const ncnn::Mat &reference, const ncnn::Mat &support) const;
    ncnn::VkMat record(const ncnn::VkMat &reference, const ncnn::VkMat &support,
                       ncnn::VkCompute &command, const ncnn::Option &option) const;
    const ncnn::Option &option() const;
    const ncnn::VulkanDevice *device() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rvf
