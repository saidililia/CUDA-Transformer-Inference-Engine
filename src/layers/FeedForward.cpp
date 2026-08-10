#include "layers/FeedForward.h"

#include <cmath>

namespace transformer {

namespace {

float gelu(float x) {

    constexpr float kSqrtTwoOverPi =
        0.7978845608f;

    constexpr float kCoefficient =
        0.044715f;

    const float inner =
        kSqrtTwoOverPi *
        (x + kCoefficient * x * x * x);

    return 0.5f *
           x *
           (1.0f + std::tanh(inner));
}

} // anonymous namespace

FeedForward::FeedForward(
    size_t hidden_size,
    size_t intermediate_size
)
    : up_projection_(
          hidden_size,
          intermediate_size
      ),
      down_projection_(
          intermediate_size,
          hidden_size
      ) {}

Tensor FeedForward::forward(
    const Tensor& input
) const {

    Tensor hidden =
        up_projection_.forward(input);

    for (size_t i = 0;
         i < hidden.numel();
         ++i) {

        hidden[i] = gelu(hidden[i]);
    }

    return down_projection_.forward(hidden);
}

} // namespace transformer