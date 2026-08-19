// Logger.h header file for the Logger class in the transformer namespace, which provides static methods for logging informational and error messages to facilitate debugging and monitoring of the transformer model's execution.

#pragma once

#include <string>

namespace transformer {

class Logger {
public:
    static void info(
        const std::string& message
    );

    static void error(
        const std::string& message
    );
};

} // namespace transformer