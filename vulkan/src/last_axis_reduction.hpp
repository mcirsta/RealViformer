#pragma once

#include <layer.h>

namespace rvf
{
// FP32 sum/mean of the last axis, keeping dimensions. This is the complete
// Reduction subset emitted by the RealViformer exporter; other forms fail.
class LastAxisReductionLayer final : public ncnn::Layer
{
  public:
    LastAxisReductionLayer();
    ~LastAxisReductionLayer() override;
    int load_param(const ncnn::ParamDict &pd) override;
    int create_pipeline(const ncnn::Option &opt) override;
    int destroy_pipeline(const ncnn::Option &opt) override;
    int forward(const ncnn::Mat &input, ncnn::Mat &output, const ncnn::Option &opt) const override;
    int forward(const ncnn::VkMat &input, ncnn::VkMat &output, ncnn::VkCompute &cmd,
                const ncnn::Option &opt) const override;

  private:
    int axis_ = -1;
    bool mean_ = false;
    float coefficient_ = 1.f;
    int long_lanes_ = 64;
    ncnn::Pipeline *short_pipeline_ = nullptr;
    ncnn::Pipeline *long_pipeline_ = nullptr;
};
ncnn::Layer *create_last_axis_reduction_layer(void *userdata);
} // namespace rvf
