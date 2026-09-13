#pragma once

#include <layer.h>

namespace rvf
{

// ncnn's bundled Vulkan GELU always uses the tanh approximation. RealViformer
// needs the erf form; overriding this layer also preserves fast_gelu=1 graphs.
class GeluLayer final : public ncnn::Layer
{
  public:
    GeluLayer();
    ~GeluLayer() override;
    int load_param(const ncnn::ParamDict &parameters) override;
    int create_pipeline(const ncnn::Option &option) override;
    int destroy_pipeline(const ncnn::Option &option) override;
    int forward_inplace(ncnn::Mat &tensor, const ncnn::Option &option) const override;
    int forward_inplace(ncnn::VkMat &tensor, ncnn::VkCompute &command,
                        const ncnn::Option &option) const override;

  private:
    int fast_ = 0;
    ncnn::Pipeline *pipeline_ = nullptr;
};

ncnn::Layer *create_gelu_layer(void *userdata);

} // namespace rvf
