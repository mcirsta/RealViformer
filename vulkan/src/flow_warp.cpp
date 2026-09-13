#include "flow_warp.hpp"

#include <gpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rvf {
namespace {

constexpr char flow_warp_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer feature_blob { sfp feature_data[]; };
layout(binding = 1) readonly buffer flow_blob { sfp flow_data[]; };
layout(binding = 2) writeonly buffer output_blob { sfp output_data[]; };

layout(push_constant) uniform parameter
{
    int width;
    int height;
    int channels;
    int feature_cstep;
    int flow_cstep;
    int output_cstep;
    int border_padding;
} p;

afp load_feature(int channel, int x, int y)
{
    if (p.border_padding != 0)
    {
        x = clamp(x, 0, p.width - 1);
        y = clamp(y, 0, p.height - 1);
    }
    else if (x < 0 || x >= p.width || y < 0 || y >= p.height)
    {
        return afp(0.f);
    }
    return buffer_ld1(feature_data, channel * p.feature_cstep + y * p.width + x);
}

float coordinate_divide(float numerator, int denominator)
{
    // Correctly rounded FP32 division by a small positive integer using only
    // portable 32-bit integer operations. Vulkan float division (and fma) may
    // be approximate; a one-ULP quotient error moves the sampling coordinate.
    uint bits = floatBitsToUint(numerator);
    int exponent = int((bits >> 23) & 255u);
    if (exponent == 0 || exponent == 255 || denominator == 1)
        return numerator / float(denominator);
    uint divisor = uint(denominator);
    uint significand = (bits & 0x7fffffu) | 0x800000u;
    int shift = findMSB(divisor);
    if (significand < (divisor << (23 - shift))) ++shift;
    if (exponent <= shift)
        return numerator / float(denominator);
    uint remainder = (significand % divisor) << shift;
    uint quotient = ((significand / divisor) << shift) + remainder / divisor;
    remainder %= divisor;
    if (remainder * 2u > divisor || (remainder * 2u == divisor && (quotient & 1u) != 0u))
        ++quotient;
    if (quotient == 0x1000000u) { quotient >>= 1; --shift; }
    return uintBitsToFloat((bits & 0x80000000u) | (uint(exponent - shift) << 23)
                          | (quotient & 0x7fffffu));
}

void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int channel = int(gl_GlobalInvocationID.z);
    if (x >= p.width || y >= p.height || channel >= p.channels)
        return;

    const int pixel = y * p.width + x;
    // Preserve the FP32 normalize/unnormalize round trip used by the trained
    // PyTorch path. Algebraically cancelling it changes subpixel positions.
    // precise does not propagate backwards across a function-call boundary.
    precise float numerator_x = 2.0 * (float(x) + buffer_ld1(flow_data, pixel));
    precise float numerator_y = 2.0 * (float(y) + buffer_ld1(flow_data, p.flow_cstep + pixel));
    precise float gx = coordinate_divide(numerator_x, max(p.width - 1, 1)) - 1.0;
    precise float gy = coordinate_divide(numerator_y, max(p.height - 1, 1)) - 1.0;
    precise float source_x = ((gx + 1.0) / 2.0) * float(p.width - 1);
    precise float source_y = ((gy + 1.0) / 2.0) * float(p.height - 1);
    if (isnan(source_x) || isnan(source_y))
    {
        buffer_st1(output_data, channel * p.output_cstep + pixel, source_x + source_y);
        return;
    }
    if (p.border_padding != 0)
    {
        source_x = clamp(source_x, 0.0, float(p.width - 1));
        source_y = clamp(source_y, 0.0, float(p.height - 1));
    }
    else if (source_x <= -1.0 || source_x >= float(p.width)
             || source_y <= -1.0 || source_y >= float(p.height))
    {
        // Reject outside coordinates before float-to-int conversion, including
        // finite displacements too large to be represented by an integer.
        buffer_st1(output_data, channel * p.output_cstep + pixel, afp(0.f));
        return;
    }
    const int x0 = int(floor(source_x));
    const int y0 = int(floor(source_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const afp dx = source_x - afp(x0);
    const afp dy = source_y - afp(y0);

    const afp top = mix(load_feature(channel, x0, y0),
                        load_feature(channel, x1, y0), dx);
    const afp bottom = mix(load_feature(channel, x0, y1),
                           load_feature(channel, x1, y1), dx);
    const afp value = mix(top, bottom, dy);
    buffer_st1(output_data, channel * p.output_cstep + pixel, value);
}
)glsl";

float sample(
    const ncnn::Mat& feature,
    int channel,
    int x,
    int y,
    FlowWarpPadding padding)
{
    if (padding == FlowWarpPadding::Border)
    {
        x = std::clamp(x, 0, feature.w - 1);
        y = std::clamp(y, 0, feature.h - 1);
    }
    else if (x < 0 || x >= feature.w || y < 0 || y >= feature.h)
    {
        return 0.f;
    }
    return feature.channel(channel).row(y)[x];
}

} // namespace

FlowWarpLayer::FlowWarpLayer(FlowWarpPadding padding) : padding_(padding)
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = false;
    support_vulkan_packing = false;
    support_fp16_storage = false;
    support_bf16_storage = false;
}

FlowWarpLayer::~FlowWarpLayer() { delete pipeline_; }

int FlowWarpLayer::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute)
        return 0;

    std::vector<std::uint32_t> spirv;
    const int compile_result = ncnn::compile_spirv_module(
        flow_warp_shader,
        static_cast<int>(sizeof(flow_warp_shader) - 1),
        opt,
        spirv);
    if (compile_result != 0)
        return compile_result;

    pipeline_ = new ncnn::Pipeline(vkdev);
    pipeline_->set_local_size_xyz(8, 8, 1);
    const std::vector<ncnn::vk_specialization_type> specializations;
    return pipeline_->create(
        spirv.data(), spirv.size() * sizeof(std::uint32_t), specializations);
}

int FlowWarpLayer::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    delete pipeline_;
    pipeline_ = nullptr;
    return 0;
}

int FlowWarpLayer::forward(
    const std::vector<ncnn::Mat>& bottom_blobs,
    std::vector<ncnn::Mat>& top_blobs,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || top_blobs.size() != 1)
        return -1;
    const ncnn::Mat& feature = bottom_blobs[0];
    const ncnn::Mat& flow = bottom_blobs[1];
    if (feature.empty() || flow.empty() || feature.n != 1 || flow.n != 1
        || feature.w > 16384 || feature.h > 16384
        || feature.c > 65535 || feature.total() > size_t(std::numeric_limits<int>::max())
        || feature.dims != 3 || flow.dims != 3 || feature.elempack != 1
        || flow.elempack != 1 || feature.elemsize != sizeof(float)
        || flow.elemsize != sizeof(float) || flow.c != 2
        || feature.w != flow.w || feature.h != flow.h)
        return -1;

    ncnn::Mat& output = top_blobs[0];
    output.create(
        feature.w,
        feature.h,
        feature.c,
        sizeof(float),
        1,
        opt.blob_allocator);
    if (output.empty())
        return -100;

    const float* flow_x = flow.channel(0);
    const float* flow_y = flow.channel(1);
    const int work_items = feature.c * feature.h;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int item = 0; item < work_items; ++item)
    {
        const int channel = item / feature.h;
        const int y = item % feature.h;
        float* destination = output.channel(channel).row(y);
        for (int x = 0; x < feature.w; ++x)
        {
            const int pixel = y * feature.w + x;
            const float gx = (2.f * (x + flow_x[pixel])) / std::max(feature.w - 1, 1) - 1.f;
            const float gy = (2.f * (y + flow_y[pixel])) / std::max(feature.h - 1, 1) - 1.f;
            float source_x = ((gx + 1.f) / 2.f) * (feature.w - 1);
            float source_y = ((gy + 1.f) / 2.f) * (feature.h - 1);
            if (std::isnan(source_x) || std::isnan(source_y))
            {
                destination[x] = source_x + source_y;
                continue;
            }
            if (padding_ == FlowWarpPadding::Border)
            {
                source_x = std::clamp(source_x, 0.f, float(feature.w - 1));
                source_y = std::clamp(source_y, 0.f, float(feature.h - 1));
            }
            else if (source_x <= -1.f || source_x >= feature.w || source_y <= -1.f ||
                     source_y >= feature.h)
            {
                destination[x] = 0.f;
                continue;
            }
            const int x0 = static_cast<int>(std::floor(source_x));
            const int y0 = static_cast<int>(std::floor(source_y));
            const float dx = source_x - x0;
            const float dy = source_y - y0;
            const float top = sample(feature, channel, x0, y0, padding_)
                            + dx * (sample(feature, channel, x0 + 1, y0, padding_)
                                    - sample(feature, channel, x0, y0, padding_));
            const float bottom = sample(feature, channel, x0, y0 + 1, padding_)
                               + dx * (sample(feature, channel, x0 + 1, y0 + 1, padding_)
                                       - sample(feature, channel, x0, y0 + 1, padding_));
            destination[x] = top + dy * (bottom - top);
        }
    }
    return 0;
}

int FlowWarpLayer::forward(
    const std::vector<ncnn::VkMat>& bottom_blobs,
    std::vector<ncnn::VkMat>& top_blobs,
    ncnn::VkCompute& command,
    const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2 || top_blobs.size() != 1)
        return -1;
    const ncnn::VkMat& feature = bottom_blobs[0];
    const ncnn::VkMat& flow = bottom_blobs[1];
    if (feature.empty() || flow.empty() || feature.n != 1 || flow.n != 1
        || feature.w > 16384 || feature.h > 16384
        || feature.c > 65535 || feature.total() > size_t(std::numeric_limits<int>::max())
        || feature.dims != 3 || flow.dims != 3 || feature.elempack != 1
        || flow.elempack != 1 || feature.elemsize != sizeof(float)
        || flow.elemsize != sizeof(float) || flow.c != 2 || feature.w != flow.w
        || feature.h != flow.h)
        return -1;

    ncnn::VkMat& output = top_blobs[0];
    output.create(
        feature.w,
        feature.h,
        feature.c,
        sizeof(float),
        1,
        opt.blob_vkallocator);
    if (output.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(3);
    bindings[0] = feature;
    bindings[1] = flow;
    bindings[2] = output;
    std::vector<ncnn::vk_constant_type> constants(7);
    constants[0].i = feature.w;
    constants[1].i = feature.h;
    constants[2].i = feature.c;
    constants[3].i = static_cast<int>(feature.cstep);
    constants[4].i = static_cast<int>(flow.cstep);
    constants[5].i = static_cast<int>(output.cstep);
    constants[6].i = padding_ == FlowWarpPadding::Border ? 1 : 0;

    ncnn::VkMat dispatcher;
    dispatcher.w = feature.w;
    dispatcher.h = feature.h;
    dispatcher.c = feature.c;
    command.record_pipeline(pipeline_, bindings, constants, dispatcher);
    return 0;
}

ncnn::Layer* create_flow_warp_zeros_layer(void* /*userdata*/)
{
    return new FlowWarpLayer(FlowWarpPadding::Zeros);
}

ncnn::Layer* create_flow_warp_border_layer(void* /*userdata*/)
{
    return new FlowWarpLayer(FlowWarpPadding::Border);
}

} // namespace rvf
