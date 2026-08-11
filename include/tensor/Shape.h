#pragma once // this is header protection to avoid multiple inclusions of the same header file

#include <cstddef> // this is a standard library header that provides definitions for size_t and other types
#include <initializer_list> // this is a standard library header that provides support for initializer lists
#include <numeric> // this is a standard library header that provides algorithms for numeric operations
#include <stdexcept> // this is a standard library header that provides exception classes
#include <vector> // this is a standard library header that provides the vector container


namespace transformer { // namespace transformer is a namespace that encapsulates the Shape class and any other related classes or functions to avoid name collisions with other libraries or code.

class Shape { // the Shape class represents the shape of a tensor, which is defined by its dimensions.
public:
    Shape() = default; // default constructor that initializes an empty shape

    Shape(std::initializer_list<size_t> dimensions)
        : dims_(dimensions) {} // constructor that initializes the shape with a list of dimensions

    explicit Shape(const std::vector<size_t>& dimensions)
        : dims_(dimensions) {} // constructor that initializes the shape with a vector of dimensions

    size_t size() const {
        return dims_.size();
    } // returns the number of dimensions in the shape

    size_t operator[](size_t index) const {
        return dims_.at(index);
    } // 

    const std::vector<size_t>& dimensions() const {
        return dims_;
    } // returns a const reference to the vector of dimensions

    size_t numel() const {
        if (dims_.empty()) {
            return 0;
        }

        return std::accumulate(
            dims_.begin(),
            dims_.end(),
            size_t{1},
            std::multiplies<size_t>()
        );
    } // calculates and returns the total number of elements in the tensor represented by this shape by multiplying all dimensions together. If the shape is empty, it returns 0.

private:
    std::vector<size_t> dims_; // 
};

} // namespace transformer