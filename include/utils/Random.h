// Random.h header file for the Random class in the transformer namespace, which provides methods for generating random numbers from normal and uniform distributions, useful for initializing weights and biases in neural networks.

#pragma once

#include <random>

namespace transformer {

class Random {
public:
    explicit Random(unsigned int seed = 42)
        : generator_(seed) {}

    float normal(float mean = 0.0f, float stddev = 0.02f) {
        std::normal_distribution<float> distribution(mean, stddev);
        return distribution(generator_);
    }

    float uniform(float min = -1.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> distribution(min, max);
        return distribution(generator_);
    }

private:
    std::mt19937 generator_;
};

} // namespace transformer