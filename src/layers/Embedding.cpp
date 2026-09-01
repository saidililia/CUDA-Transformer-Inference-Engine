#include "layers/Embedding.h"

#include <stdexcept>

namespace transformer {

Embedding::Embedding(
    size_t vocabulary_size,
    size_t embedding_dimension
)
    : vocabulary_size_(vocabulary_size),
      embedding_dimension_(embedding_dimension),
      weights_({
          vocabulary_size,
          embedding_dimension
      }) {

    Random random(42);

    for (size_t i = 0; i < weights_.numel(); ++i) {
        weights_[i] = random.normal(0.0f, 0.02f);
    } // randomly initialize the embedding matrix with values drawn from a normal distribution with mean 0 and standard deviation 0.02
}

Tensor Embedding::forward(const Tensor& token_ids) const {

    if (token_ids.ndim() != 1) {
        throw std::runtime_error(
            "Embedding expects a 1D token tensor"
        );
    }

    const size_t sequence_length = token_ids.dim(0);

    Tensor output({
        sequence_length,
        embedding_dimension_
    }); // embedding output tensor must be 2D: [sequence_length, embedding_hidden_size], example: [8, 256], 8 is the number of tokens in the input sequence, 256 is the embedding dimension

    for (size_t position = 0; position < sequence_length; ++position) {

        const size_t token_id =
            static_cast<size_t>(token_ids[position]);

        if (token_id >= vocabulary_size_) {
            throw std::runtime_error(
                "Token ID exceeds vocabulary size"
            );
        }

        for (size_t d = 0; d < embedding_dimension_; ++d) {

            output[
                position * embedding_dimension_ + d
            ] =
                weights_[
                    token_id * embedding_dimension_ + d
                ];
        }
    }

    return output;
}

} // namespace transformer