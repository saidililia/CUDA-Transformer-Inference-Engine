#pragma once

#include <chrono>
#include <cstddef>
#include <map>
#include <string>

namespace transformer {

class Profiler {
public:
    void start(const std::string& name);

    void stop(const std::string& name);

    void reset();

    void printReport() const;

private:
    struct Measurement {
        double totalMilliseconds = 0.0;
        std::size_t calls = 0;

        std::chrono::high_resolution_clock::time_point startTime;
        bool running = false;
    };

    std::map<std::string, Measurement> measurements_;
};

} 