#include "attention_mask_descriptor.hpp"

#include <gpu.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace rvf {
namespace {

constexpr char correlation_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer q_blob { sfp q_data[]; };
layout(binding = 1) readonly buffer k_blob { sfp k_data[]; };
layout(binding = 2) writeonly buffer logits_blob { sfp logits_data[]; };

layout(push_constant) uniform parameter
{
    int width;
    int channels;
    int heads;
    int q_cstep;
    int k_cstep;
    int logits_cstep;
} p;

shared afp partial_sum[64];

void main()
{
    const int lane = int(gl_LocalInvocationID.x);
    const int pair = int(gl_WorkGroupID.y);
    const int head = int(gl_WorkGroupID.z);
    const int query_channel = pair / p.channels;
    const int key_channel = pair - query_channel * p.channels;

    afp sum = afp(0.f);
    const int q_offset = head * p.q_cstep + query_channel * p.width;
    const int k_offset = head * p.k_cstep + key_channel * p.width;
    for (int index = lane; index < p.width; index += 64)
        sum += buffer_ld1(q_data, q_offset + index) * buffer_ld1(k_data, k_offset + index);

    partial_sum[lane] = sum;
    barrier();
    for (int stride = 32; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            partial_sum[lane] += partial_sum[lane + stride];
        barrier();
    }

    if (lane == 0)
    {
        const int output_offset = head * p.logits_cstep
                                + query_channel * p.channels + key_channel;
        buffer_st1(logits_data, output_offset, partial_sum[0]);
    }
}
)glsl";

constexpr char descriptor_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer logits_blob { sfp logits_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_data[]; };

layout(push_constant) uniform parameter
{
    int channels;
    int heads;
    int logits_cstep;
    int output_cstep;
} p;

void main()
{
    const int query_channel = int(gl_GlobalInvocationID.x);
    const int head = int(gl_GlobalInvocationID.y);
    if (query_channel >= p.channels || head >= p.heads)
        return;

    afp maximum = afp(-3.402823466e+38f);
    afp sum = afp(0.f);
    const int input_offset = head * p.logits_cstep + query_channel * p.channels;
    for (int key_channel = 0; key_channel < p.channels; ++key_channel)
    {
        const afp value = buffer_ld1(logits_data, input_offset + key_channel);
        maximum = max(maximum, value);
        sum += value;
    }

    const int output_offset = head * p.output_cstep + query_channel * 2;
    buffer_st1(output_data, output_offset, maximum);
    buffer_st1(output_data, output_offset + 1, sum / afp(p.channels));
}
)glsl";

int compile_pipeline(
    const ncnn::VulkanDevice* device,
    const ncnn::Option& opt,
    const char* source,
    std::size_t source_size,
    int local_x,
    ncnn::Pipeline*& pipeline)
{
    std::vector<std::uint32_t> spirv;
    const int compile_result = ncnn::compile_spirv_module(
        source, static_cast<int>(source_size), opt, spirv);
    if (compile_result != 0)
        return compile_result;

    pipeline = new ncnn::Pipeline(device);
    pipeline->set_local_size_xyz(local_x, 1, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    return pipeline->create(
        spirv.data(), spirv.size() * sizeof(std::uint32_t), specializations);
}

} // namespace

AttentionMaskDescriptorLayer::AttentionMaskDescriptorLayer()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = false;
    support_vulkan_packing = false;
    support_fp16_storage = false;
    support_bf16_storage = false;
}

int AttentionMaskDescriptorLayer::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute)
        return 0;
    int result = compile_pipeline(
        vkdev,
        opt,
        correlation_shader,
        sizeof(correlation_shader) - 1,
        64,
        dot_pipeline_);
    if (result != 0)
        return result;
    result = compile_pipeline(
        vkdev,
        opt,
        descriptor_shader,
        sizeof(descriptor_shader) - 1,
        1,
        reduce_pipeline_);
    return result;
}

int AttentionMaskDescriptorLayer::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete dot_pipeline_;
    dot_pipeline_ = nullptr;
    delete reduce_pipeline_;
    reduce_pipeline_ = nullptr;
    return 0;
}

int AttentionMaskDescriptorLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || top_blobs.size() != 1)
        return -1;
    // TorchScript records this module's two dependency edges in key/query
    // order even though its Python signature is query/key.
    const ncnn::Mat& k = bottom_blobs[0];
    const ncnn::Mat& q = bottom_blobs[1];
    if (q.dims != 3 || k.dims != 3 || q.elempack != 1 || k.elempack != 1
        || q.w != k.w || q.h != k.h || q.c != k.c)
        return -1;

    ncnn::Mat& output = top_blobs[0];
    output.create(2, q.h, q.c, sizeof(float), 1, opt.blob_allocator);
    if (output.empty())
        return -100;

    const int work_items = q.c * q.h;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int item = 0; item < work_items; ++item)
    {
        const int head = item / q.h;
        const int query_channel = item % q.h;
        const float* query = q.channel(head).row(query_channel);
        float maximum = -FLT_MAX;
        float sum = 0.f;
        for (int key_channel = 0; key_channel < q.h; ++key_channel)
        {
            const float* key = k.channel(head).row(key_channel);
            float correlation = 0.f;
            for (int index = 0; index < q.w; ++index)
                correlation += query[index] * key[index];
            maximum = std::max(maximum, correlation);
            sum += correlation;
        }
        float* descriptor = output.channel(head).row(query_channel);
        descriptor[0] = maximum;
        descriptor[1] = sum / q.h;
    }
    return 0;
}

int AttentionMaskDescriptorLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& command,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || top_blobs.size() != 1)
        return -1;
    const ncnn::VkMat& k = bottom_blobs[0];
    const ncnn::VkMat& q = bottom_blobs[1];
    if (q.dims != 3 || k.dims != 3 || q.elempack != 1 || k.elempack != 1
        || q.w != k.w || q.h != k.h || q.c != k.c)
        return -1;

    ncnn::VkMat logits(q.h, q.h, q.c, sizeof(float), 1, opt.workspace_vkallocator);
    if (logits.empty())
        return -100;
    ncnn::VkMat& output = top_blobs[0];
    output.create(2, q.h, q.c, sizeof(float), 1, opt.blob_vkallocator);
    if (output.empty())
        return -100;

    {
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = q;
        bindings[1] = k;
        bindings[2] = logits;
        std::vector<ncnn::vk_constant_type> constants(6);
        constants[0].i = q.w;
        constants[1].i = q.h;
        constants[2].i = q.c;
        constants[3].i = static_cast<int>(q.cstep);
        constants[4].i = static_cast<int>(k.cstep);
        constants[5].i = static_cast<int>(logits.cstep);

        ncnn::VkMat dispatcher;
        dispatcher.w = 64;
        dispatcher.h = q.h * q.h;
        dispatcher.c = q.c;
        command.record_pipeline(dot_pipeline_, bindings, constants, dispatcher);
    }
    {
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = logits;
        bindings[1] = output;
        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].i = q.h;
        constants[1].i = q.c;
        constants[2].i = static_cast<int>(logits.cstep);
        constants[3].i = static_cast<int>(output.cstep);

        ncnn::VkMat dispatcher;
        dispatcher.w = q.h;
        dispatcher.h = q.c;
        dispatcher.c = 1;
        command.record_pipeline(reduce_pipeline_, bindings, constants, dispatcher);
    }
    return 0;
}

ncnn::Layer* create_attention_mask_descriptor_layer(void* /*userdata*/)
{
    return new AttentionMaskDescriptorLayer;
}

} // namespace rvf
