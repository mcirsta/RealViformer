#include "last_axis_reduction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gpu.h>
#include <limits>
#include <vector>

namespace rvf
{
namespace
{
constexpr char shader[] = R"glsl(
#version 450
layout(constant_id=0) const int lanes = 64;
layout(binding=0) readonly buffer input_blob { float a[]; };
layout(binding=1) writeonly buffer output_blob { float b[]; };
layout(push_constant) uniform parameter {
    int width; int rows_per_channel; int input_cstep; int output_cstep;
    int outputs; float coefficient;
} p;
shared float partial[256];
void main() {
    uint row_index = gl_WorkGroupID.y + gl_WorkGroupID.z * gl_NumWorkGroups.y;
    if (row_index >= uint(p.outputs)) return; // uniform for the entire workgroup
    int row = int(row_index);
    int lane = int(gl_LocalInvocationID.x);
    int channel = row / p.rows_per_channel;
    int local_row = row % p.rows_per_channel;
    int offset = channel * p.input_cstep + local_row * p.width;
    float sum = 0.0;
    for (int i = lane; i < p.width; i += lanes) sum += a[offset + i];
    partial[lane] = sum;
    barrier();
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier();
    }
    if (lane == 0) b[channel * p.output_cstep + local_row] = partial[0] * p.coefficient;
}
)glsl";

template <class T> bool valid(const T &input, int axis)
{
    return !input.empty() && input.dims >= 1 && input.dims <= 4 && input.n == 1 && input.w > 0 &&
           input.elemsize == sizeof(float) && input.elempack == 1 &&
           (axis == -1 || axis == input.dims - 1) &&
           input.total() <= static_cast<size_t>(std::numeric_limits<int>::max());
}

template <class T, class Allocator>
void create_output(const T &input, T &output, Allocator *allocator)
{
    if (input.dims == 1)
        output.create(1, sizeof(float), 1, allocator);
    if (input.dims == 2)
        output.create(1, input.h, sizeof(float), 1, allocator);
    if (input.dims == 3)
        output.create(1, input.h, input.c, sizeof(float), 1, allocator);
    if (input.dims == 4)
        output.create(1, input.h, input.d, input.c, sizeof(float), 1, allocator);
}
} // namespace

LastAxisReductionLayer::LastAxisReductionLayer()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;
    support_packing = support_vulkan_packing = false;
    support_fp16_storage = support_bf16_storage = false;
}

LastAxisReductionLayer::~LastAxisReductionLayer()
{
    delete short_pipeline_;
    delete long_pipeline_;
}

int LastAxisReductionLayer::load_param(const ncnn::ParamDict &pd)
{
    const int op = pd.get(0, 0);
    const ncnn::Mat axes = pd.get(3, ncnn::Mat());
    if ((op != 0 && op != 3) || pd.get(1, 1) != 0 || pd.get(4, 0) != 1 || pd.get(5, 0) != 1 ||
        axes.dims != 1 || axes.w != 1)
        return -1;
    axis_ = static_cast<const int *>(axes)[0];
    mean_ = op == 3;
    coefficient_ = pd.get(2, 1.f);
    return std::isfinite(coefficient_) ? 0 : -1;
}

int LastAxisReductionLayer::create_pipeline(const ncnn::Option &opt)
{
    if (!opt.use_vulkan_compute)
        return 0;
    const auto limit = std::min(vkdev->info.max_workgroup_size_x(),
                                vkdev->info.max_workgroup_invocations());
    if (limit < 64) return -1;
    long_lanes_ = limit >= 256 ? 256 : limit >= 128 ? 128 : 64;
    std::vector<std::uint32_t> spirv;
    int result = ncnn::compile_spirv_module(shader, sizeof(shader) - 1, opt, spirv);
    if (result)
        return result;
    short_pipeline_ = new ncnn::Pipeline(vkdev);
    short_pipeline_->set_local_size_xyz(64, 1, 1);
    std::vector<ncnn::vk_specialization_type> specialization(1);
    specialization[0].i = 64;
    result =
        short_pipeline_->create(spirv.data(), spirv.size() * sizeof(std::uint32_t), specialization);
    if (result)
        return result;
    long_pipeline_ = new ncnn::Pipeline(vkdev);
    long_pipeline_->set_local_size_xyz(long_lanes_, 1, 1);
    specialization[0].i = long_lanes_;
    return long_pipeline_->create(spirv.data(), spirv.size() * sizeof(std::uint32_t),
                                  specialization);
}

int LastAxisReductionLayer::destroy_pipeline(const ncnn::Option &)
{
    delete short_pipeline_;
    short_pipeline_ = nullptr;
    delete long_pipeline_;
    long_pipeline_ = nullptr;
    return 0;
}

int LastAxisReductionLayer::forward(const ncnn::Mat &input, ncnn::Mat &output,
                                    const ncnn::Option &opt) const
{
    if (!valid(input, axis_))
        return -1;
    create_output(input, output, opt.blob_allocator);
    if (output.empty())
        return -100;
    const int rows = input.h * input.d;
    const double coefficient = mean_ ? double(coefficient_) / input.w : coefficient_;
    // Long spatial sums need a wider accumulator on CPU; scalar FP32 sums
    // accumulate avoidable error as the source resolution increases.
#pragma omp parallel for num_threads(opt.num_threads)
    for (int row = 0; row < rows * input.c; ++row)
    {
        const int channel = row / rows, local = row % rows;
        const float *source =
            static_cast<const float *>(input.data) + channel * input.cstep + local * input.w;
        double sum = 0.0;
        for (int i = 0; i < input.w; ++i)
            sum += source[i];
        static_cast<float *>(output.data)[channel * output.cstep + local] =
            float(sum * coefficient);
    }
    return 0;
}

int LastAxisReductionLayer::forward(const ncnn::VkMat &input, ncnn::VkMat &output,
                                    ncnn::VkCompute &cmd, const ncnn::Option &opt) const
{
    if (!valid(input, axis_) || !short_pipeline_ || !long_pipeline_)
        return -1;
    create_output(input, output, opt.blob_vkallocator);
    if (output.empty())
        return -100;
    const int rows = input.h * input.d, count = rows * input.c;
    std::vector<ncnn::vk_constant_type> constants(6);
    constants[0].i = input.w;
    constants[1].i = rows;
    constants[2].i = int(input.cstep);
    constants[3].i = int(output.cstep);
    constants[4].i = count;
    constants[5].f = mean_ ? coefficient_ / input.w : coefficient_;
    ncnn::VkMat dispatcher;
    dispatcher.w = input.w <= 256 ? 64 : long_lanes_;
    dispatcher.h = std::min(count, 65535);
    dispatcher.c = (count + 65534LL) / 65535;
    cmd.record_pipeline(input.w <= 256 ? short_pipeline_ : long_pipeline_, {input, output},
                        constants, dispatcher);
    return 0;
}

ncnn::Layer *create_last_axis_reduction_layer(void *) { return new LastAxisReductionLayer; }
} // namespace rvf
