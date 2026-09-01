#include "utils/Profiler.h"

#include <iomanip>
#include <iostream>

namespace transformer {

void Profiler::start(const std::string& name) {
    auto& measurement = measurements_[name];

    measurement.startTime =
        std::chrono::high_resolution_clock::now();

    measurement.running = true;
}

void Profiler::stop(const std::string& name) {
    auto iterator = measurements_.find(name);

    if (iterator == measurements_.end()) {
        return;
    }

    auto& measurement = iterator->second;

    if (!measurement.running) {
        return;
    }

    const auto endTime =
        std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed =
        endTime - measurement.startTime;

    measurement.totalMilliseconds += elapsed.count();
    measurement.calls++;

    measurement.running = false;
}

void Profiler::reset() {
    measurements_.clear();
}

void Profiler::printReport() const {
    std::cout << "\n";
    std::cout << "CPU PROFILE\n";
    std::cout << "============================================================\n";

    std::cout
        << std::left << std::setw(30) << "Operation"
        << std::right << std::setw(14) << "Total (ms)"
        << std::setw(10) << "Calls"
        << std::setw(14) << "Avg (ms)"
        << "\n";

    std::cout << "------------------------------------------------------------\n";

    for (const auto& [name, measurement] : measurements_) {
        const double average =
            measurement.calls > 0
                ? measurement.totalMilliseconds /
                      static_cast<double>(measurement.calls)
                : 0.0;

        std::cout
            << std::left << std::setw(30) << name
            << std::right << std::setw(14)
            << std::fixed << std::setprecision(3)
            << measurement.totalMilliseconds
            << std::setw(10) << measurement.calls
            << std::setw(14) << average
            << "\n";
    }

    std::cout << "============================================================\n";
}

} 