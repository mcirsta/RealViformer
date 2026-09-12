#include "weights.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace rvf {
namespace {

constexpr std::array<char, 8> kMagic{'R', 'V', 'F', 'W', 'T', '0', '0', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianTag = 0x01020304;
constexpr std::uint32_t kFloat32 = 1;
constexpr std::size_t kHeaderSize = 32;
constexpr std::size_t kRecordSize = 32;

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::runtime_error("weight file integer overflow");
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error("weight file integer overflow");
    }
    return left * right;
}

std::uint64_t align(std::uint64_t value, std::uint64_t alignment) {
    return checked_multiply((checked_add(value, alignment - 1) / alignment),
                            alignment);
}

template <typename T>
T read_scalar(const std::vector<std::byte>& bytes, std::uint64_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        throw std::runtime_error("truncated weight file");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

}  // namespace

WeightFile::WeightFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open native weights: " + path);
    }
    const std::streamsize file_size = file.tellg();
    if (file_size < static_cast<std::streamsize>(kHeaderSize)) {
        throw std::runtime_error("native weight file is too small: " + path);
    }
    file.seekg(0);
    bytes_.resize(static_cast<std::size_t>(file_size));
    if (!file.read(reinterpret_cast<char*>(bytes_.data()), file_size)) {
        throw std::runtime_error("cannot read native weights: " + path);
    }

    if (std::memcmp(bytes_.data(), kMagic.data(), kMagic.size()) != 0) {
        throw std::runtime_error("native weight file has an invalid magic value");
    }
    if (read_scalar<std::uint32_t>(bytes_, 8) != kVersion) {
        throw std::runtime_error("unsupported native weight format version");
    }
    if (read_scalar<std::uint32_t>(bytes_, 12) != kEndianTag) {
        throw std::runtime_error("native weight byte order is unsupported");
    }
    const std::uint32_t tensor_count = read_scalar<std::uint32_t>(bytes_, 16);
    data_offset_ = read_scalar<std::uint64_t>(bytes_, 24);
    if (data_offset_ < kHeaderSize || data_offset_ > bytes_.size()) {
        throw std::runtime_error("native weight data offset is invalid");
    }

    std::uint64_t cursor = kHeaderSize;
    tensors_.reserve(tensor_count);
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
        const std::uint64_t record_start = cursor;
        if (cursor > data_offset_ || kRecordSize > data_offset_ - cursor) {
            throw std::runtime_error("native weight table is truncated");
        }
        const std::uint32_t name_size = read_scalar<std::uint32_t>(bytes_, cursor);
        const std::uint32_t dtype = read_scalar<std::uint32_t>(bytes_, cursor + 4);
        const std::uint32_t dimensions =
            read_scalar<std::uint32_t>(bytes_, cursor + 8);
        const std::uint64_t data_offset =
            read_scalar<std::uint64_t>(bytes_, cursor + 16);
        const std::uint64_t data_size =
            read_scalar<std::uint64_t>(bytes_, cursor + 24);
        cursor += kRecordSize;
        if (dtype != kFloat32) {
            throw std::runtime_error("native weight tensor is not float32");
        }
        if (dimensions > 8) {
            throw std::runtime_error("native weight tensor rank is unreasonable");
        }

        WeightTensor tensor;
        tensor.shape.reserve(dimensions);
        std::uint64_t elements = 1;
        for (std::uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            const std::uint64_t extent =
                read_scalar<std::uint64_t>(bytes_, cursor);
            cursor += sizeof(std::uint64_t);
            elements = checked_multiply(elements, extent);
            tensor.shape.push_back(extent);
        }
        if (cursor > data_offset_ || name_size > data_offset_ - cursor) {
            throw std::runtime_error("native weight tensor name is truncated");
        }
        tensor.name.assign(reinterpret_cast<const char*>(bytes_.data() + cursor),
                           name_size);
        cursor += name_size;
        cursor = checked_add(record_start, align(cursor - record_start, 8));
        tensor.offset = data_offset;
        tensor.size = data_size;

        if (checked_multiply(elements, sizeof(float)) != tensor.size) {
            throw std::runtime_error("native weight tensor size does not match shape");
        }
        const std::uint64_t absolute_end =
            checked_add(data_offset_, checked_add(tensor.offset, tensor.size));
        if (absolute_end > bytes_.size()) {
            throw std::runtime_error("native weight tensor data is truncated");
        }
        tensors_.push_back(std::move(tensor));
    }

    if (!std::is_sorted(tensors_.begin(), tensors_.end(),
                        [](const WeightTensor& left, const WeightTensor& right) {
                            return left.name < right.name;
                        })) {
        throw std::runtime_error("native weight table is not sorted");
    }
}

std::span<const std::byte> WeightFile::data() const {
    return {bytes_.data() + data_offset_, bytes_.size() - data_offset_};
}

const WeightTensor* WeightFile::find(const std::string& name) const {
    const auto found = std::lower_bound(
        tensors_.begin(), tensors_.end(), name,
        [](const WeightTensor& tensor, const std::string& target) {
            return tensor.name < target;
        });
    return found != tensors_.end() && found->name == name ? &*found : nullptr;
}

}  // namespace rvf

