// Timer.h header file for the Timer class in the transformer namespace, which provides functionality to measure elapsed time in milliseconds, useful for profiling and benchmarking code execution.

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