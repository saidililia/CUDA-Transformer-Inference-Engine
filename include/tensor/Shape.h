// shape header file for tensor class
#pragma once 

#include <cstddef> 
#include <initializer_list>  
#include <numeric> 
#include <stdexcept> 
#include <vector> 


namespace transformer {  

class Shape { 
public:
    Shape() = default; 

    Shape(std::initializer_list<size_t> dimensions) 
        : dims_(dimensions) {} 

    explicit Shape(const std::vector<size_t>& dimensions)
        : dims_(dimensions) {} 

    size_t size() const {
        return dims_.size(); // returns the number of dimensions in the shape
    } 

    size_t operator[](size_t index) const {
        return dims_.at(index); // returns the size of the dimension at the specified index of the shape's dimension vector
    } 

    const std::vector<size_t>& dimensions() const {
        return dims_; // returns the vector of dimensions representing the shape
    } 

    
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
    } 

private:
    std::vector<size_t> dims_; // example: {2, 3, 4} for a 3D tensor with shape (2, 3, 4)
};

} 