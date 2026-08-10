#pragma once

#include "tensor/Tensor.h"
#include "utils/Random.h"

namespace transformer {

class Embedding {
public:
    Embedding(
        size_t vocabulary_size,
        size_t embedding_dimension
    );

    Tensor forward(const Tensor& token_ids) const;

private:
    size_t vocabulary_size_;
    size_t embedding_dimension_;

    Tensor weights_;
};

} // namespace transformer