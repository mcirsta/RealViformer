#pragma once

#include <layer.h>

namespace rvf {

class AttentionMaskDescriptorLayer final : public ncnn::Layer
{
public:
    AttentionMaskDescriptorLayer();
    ~AttentionMaskDescriptorLayer() override;

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
    ncnn::Pipeline* dot_pipeline_ = nullptr;
    ncnn::Pipeline* reduce_pipeline_ = nullptr;
};

ncnn::Layer* create_attention_mask_descriptor_layer(void* userdata);

} // namespace rvf
