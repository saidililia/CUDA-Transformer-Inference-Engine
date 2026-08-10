#pragma once

#include <chrono>

namespace transformer {

class Timer {
public:
    void start();

    double stopMilliseconds();

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace transformer