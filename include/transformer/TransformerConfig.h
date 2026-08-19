// TransformerConfig.h header file for the TransformerConfig struct in the transformer namespace, which holds configuration parameters for a transformer model, including vocabulary size, hidden size, number of attention heads, number of layers, intermediate size, maximum sequence length, and layer normalization epsilon value.

#pragma once

#include <cstddef>

namespace transformer {

struct TransformerConfig {

    size_t vocabulary_size = 10000;

    size_t hidden_size = 256;

    size_t num_heads = 4;

    size_t num_layers = 2;

    size_t intermediate_size = 1024;

    size_t max_sequence_length = 128;

    float layer_norm_epsilon = 1e-5f;
};

} // namespace transformer