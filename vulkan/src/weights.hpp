#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rvf {

struct WeightTensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

class WeightFile {
public:
    explicit WeightFile(const std::string& path);

    [[nodiscard]] const std::vector<WeightTensor>& tensors() const {
        return tensors_;
    }
    [[nodiscard]] std::span<const std::byte> data() const;
    [[nodiscard]] const WeightTensor* find(const std::string& name) const;

private:
    std::vector<std::byte> bytes_;
    std::vector<WeightTensor> tensors_;
    std::uint64_t data_offset_ = 0;
};

}  // namespace rvf

