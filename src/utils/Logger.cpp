#include "utils/Logger.h"

#include <iostream>

namespace transformer {

void Logger::info(
    const std::string& message
) {
    std::cout
        << "[INFO] "
        << message
        << '\n';
}

void Logger::error(
    const std::string& message
) {
    std::cerr
        << "[ERROR] "
        << message
        << '\n';
}

} // namespace transformer