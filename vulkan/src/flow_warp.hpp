#pragma once

#include <layer.h>

namespace rvf {

enum class FlowWarpPadding
{
    Zeros,
    Border,
};

class FlowWarpLayer final : public ncnn::Layer
{
public:
    explicit FlowWarpLayer(FlowWarpPadding padding);

    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;

    int forward(
        const std::vector<ncnn::Mat>& bottom_blobs,
        std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt) const override;

    int forward(
        const std::vector<ncnn::VkMat>& bottom_blobs,
        std::vector<ncnn::VkMat>& top_blobs,
        ncnn::VkCompute& command,
        const ncnn::Option& opt) const override;

private:
    FlowWarpPadding padding_;
    ncnn::Pipeline* pipeline_ = nullptr;
};

ncnn::Layer* create_flow_warp_zeros_layer(void* userdata);
ncnn::Layer* create_flow_warp_border_layer(void* userdata);

} // namespace rvf
