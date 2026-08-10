#include "utils/Timer.h"

namespace transformer {

void Timer::start() {

    start_time_ =
        std::chrono::high_resolution_clock::now();
}

double Timer::stopMilliseconds() {

    const auto end_time =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli>
        elapsed =
            end_time - start_time_;

    return elapsed.count();
}

} // namespace transformer