// embedding.h header file for the Embedding layer in the transformer namespace, which maps discrete token IDs to continuous vector representations (Embedding vectors).

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

    Tensor weights_; // embedding matrix of shape [vocabulary_size, embedding_dimension], where each row corresponds to the embedding vector for a specific token ID
};

} // namespace transformer