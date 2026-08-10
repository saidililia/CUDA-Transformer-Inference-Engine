#include "tensor/Tensor.h"

#include <algorithm>
#include <stdexcept>

namespace transformer {

Tensor::Tensor(const Shape& shape)
    : shape_(shape),
      data_(shape.numel(), 0.0f) {}

Tensor::Tensor(std::initializer_list<size_t> shape)
    : Tensor(Shape(shape)) {}

Tensor::Tensor(const Shape& shape, float value)
    : shape_(shape),
      data_(shape.numel(), value) {}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

size_t Tensor::numel() const {
    return data_.size();
}

size_t Tensor::ndim() const {
    return shape_.size();
}

size_t Tensor::dim(size_t axis) const {
    return shape_[axis];
}

const Shape& Tensor::shape() const {
    return shape_;
}

float& Tensor::operator[](size_t index) {
    return data_.at(index);
}

const float& Tensor::operator[](size_t index) const {
    return data_.at(index);
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

std::vector<float>& Tensor::values() {
    return data_;
}

const std::vector<float>& Tensor::values() const {
    return data_;
}

} // namespace transformer