#pragma once

#include "tensor/Shape.h"

#include <cstddef> 
#include <initializer_list>
#include <vector>

namespace transformer {

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(const Shape& shape); // this constructor initializes a tensor with the given shape, allocating memory for the data based on the number of elements in the shape.

    explicit Tensor(std::initializer_list<size_t> shape); //

    Tensor(const Shape& shape, float value); // this constructor initializes a tensor with the given shape and fills it with the specified value.

    float* data();
    const float* data() const;

    size_t numel() const; // 

    size_t ndim() const;

    size_t dim(size_t axis) const;

    const Shape& shape() const;

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    void fill(float value);

    std::vector<float>& values();
    const std::vector<float>& values() const;

private:
    Shape shape_;
    std::vector<float> data_;
};

} // namespace transformer