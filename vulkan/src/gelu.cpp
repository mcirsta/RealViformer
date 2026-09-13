#include "gelu.hpp"

#include <gpu.h>
#include <pipeline.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rvf
{
namespace
{
constexpr char shader[] = R"glsl(
#version 450
layout(binding=0) buffer Data { float data[]; };
layout(push_constant) uniform parameter { uint count; int fast; } p;

// Abramowitz-Stegun 7.1.26: max erf approximation error about 1.5e-7.
// Evaluate erfc directly for negative inputs to avoid subtracting near-one
// values in the small negative tail of GELU.
float gelu(float x) {
    if(p.fast!=0) return 0.5*x*(1.0+tanh(0.7978845608*(x+0.044715*x*x*x)));
    float z=abs(x)*0.7071067811865475;
    float t=1.0/(1.0+0.3275911*z);
    float erfc=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t
                 -0.284496736)*t+0.254829592)*t*exp(-z*z);
    return 0.5*x*(x<0.0?erfc:2.0-erfc);
}
void main() {
    for(uint i=gl_GlobalInvocationID.x;i<p.count;i+=gl_NumWorkGroups.x*64)
        data[i]=gelu(data[i]);
}
)glsl";
} // namespace

GeluLayer::GeluLayer()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;
    support_packing = false;
    support_vulkan_packing = false;
    support_fp16_storage = false;
    support_bf16_storage = false;
}

GeluLayer::~GeluLayer() { delete pipeline_; }

int GeluLayer::load_param(const ncnn::ParamDict &parameters)
{
    fast_ = parameters.get(0, 0);
    return 0;
}

int GeluLayer::create_pipeline(const ncnn::Option &option)
{
    if (!option.use_vulkan_compute)
        return 0;
    std::vector<std::uint32_t> spirv;
    int result = ncnn::compile_spirv_module(shader, sizeof(shader) - 1, option, spirv);
    if (result != 0)
        return result;
    pipeline_ = new ncnn::Pipeline(vkdev);
    pipeline_->set_local_size_xyz(64, 1, 1);
    return pipeline_->create(spirv.data(), spirv.size() * sizeof(std::uint32_t), {});
}

int GeluLayer::destroy_pipeline(const ncnn::Option &)
{
    delete pipeline_;
    pipeline_ = nullptr;
    return 0;
}

int GeluLayer::forward_inplace(ncnn::Mat &tensor, const ncnn::Option &option) const
{
    if (tensor.elempack != 1 || tensor.elemsize != sizeof(float))
        return -1;
#pragma omp parallel for num_threads(option.num_threads)
    for (int ch = 0; ch < tensor.c; ++ch)
    {
        float *data = tensor.channel(ch);
        for (int i = 0; i < tensor.w * tensor.h * tensor.d; ++i)
        {
            const float x = data[i];
            data[i] = fast_
                          ? 0.5f * x * (1.f + std::tanh(0.79788452f * (x + 0.044715f * x * x * x)))
                          : 0.5f * x * std::erfc(-0.70710678f * x);
        }
    }
    return 0;
}

int GeluLayer::forward_inplace(ncnn::VkMat &tensor, ncnn::VkCompute &command,
                               const ncnn::Option &) const
{
    if (tensor.elempack != 1 || tensor.elemsize != sizeof(float))
        return -1;
    std::vector<ncnn::vk_constant_type> constants(2);
    constants[0].u32 = static_cast<std::uint32_t>(tensor.total());
    constants[1].i = fast_;
    ncnn::VkMat dispatcher;
    dispatcher.w = static_cast<int>(std::min<std::size_t>(tensor.total(), 64U * 65535U));
    dispatcher.h = dispatcher.c = 1;
    command.record_pipeline(pipeline_, {tensor}, constants, dispatcher);
    return 0;
}

ncnn::Layer *create_gelu_layer(void *) { return new GeluLayer; }

} // namespace rvf
