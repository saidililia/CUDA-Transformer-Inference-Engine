#include "tensor/Tensor.h"

#include <cassert>
#include <iostream>

using namespace transformer;

int main() {

    Tensor tensor({2, 3});

    assert(tensor.numel() == 6);
    assert(tensor.ndim() == 2);
    assert(tensor.dim(0) == 2);
    assert(tensor.dim(1) == 3);

    tensor[0] = 1.0f;
    tensor[5] = 42.0f;

    assert(tensor[0] == 1.0f);
    assert(tensor[5] == 42.0f);

    tensor.fill(3.0f);

    for (size_t i = 0;
         i < tensor.numel();
         ++i) {

        assert(tensor[i] == 3.0f);
    }

    std::cout
        << "Tensor tests passed.\n";

    return 0;
}