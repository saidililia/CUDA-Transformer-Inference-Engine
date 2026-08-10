#include "layers/RoPE.h"

#include <cmath>
#include <stdexcept>

namespace transformer {

RoPE::RoPE(
    size_t head_dimension,
    size_t max_sequence_length,
    float theta
)
    : head_dimension_(head_dimension),
      max_sequence_length_(max_sequence_length),
      theta_(theta) {

    if (head_dimension_ % 2 != 0) {
        throw std::runtime_error(
            "RoPE requires an even head dimension"
        );
    }
}

void RoPE::apply(
    Tensor& query,
    Tensor& key,
    size_t sequence_length
) const {

    const size_t num_heads =
        query.dim(0);

    if (sequence_length > max_sequence_length_) {
        throw std::runtime_error(
            "Sequence length exceeds RoPE capacity"
        );
    }

    for (size_t head = 0; head < num_heads; ++head) {

        for (size_t position = 0;
             position < sequence_length;
             ++position) {

            for (size_t i = 0;
                 i < head_dimension_;
                 i += 2) {

                const float exponent =
                    static_cast<float>(i) /
                    static_cast<float>(head_dimension_);

                const float frequency =
                    1.0f /
                    std::pow(theta_, exponent);

                const float angle =
                    static_cast<float>(position) *
                    frequency;

                const float cosine =
                    std::cos(angle);

                const float sine =
                    std::sin(angle);

                const size_t base =
                    head *
                    sequence_length *
                    head_dimension_
                    +
                    position *
                    head_dimension_
                    +
                    i;

                const float q0 = query[base];
                const float q1 = query[base + 1];

                query[base] =
                    q0 * cosine - q1 * sine;

                query[base + 1] =
                    q0 * sine + q1 * cosine;

                const float k0 = key[base];
                const float k1 = key[base + 1];

                key[base] =
                    k0 * cosine - k1 * sine;

                key[base + 1] =
                    k0 * sine + k1 * cosine;
            }
        }
    }
}

} // namespace transformer