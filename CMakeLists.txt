cmake_minimum_required(VERSION 3.20)

project(
    cuda_transformer_inference_engine
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------
# Library
# --------------------------------------------------

add_library(transformer_core

    src/tensor/Tensor.cpp

    src/layers/Linear.cpp
    src/layers/Embedding.cpp
    src/layers/LayerNorm.cpp
    src/layers/Softmax.cpp
    src/layers/RoPE.cpp
    src/layers/Attention.cpp
    src/layers/FeedForward.cpp

    src/transformer/DecoderBlock.cpp
    src/transformer/Transformer.cpp

    src/utils/Timer.cpp
    src/utils/Logger.cpp
)

target_include_directories(
    transformer_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# --------------------------------------------------
# CPU inference executable
# --------------------------------------------------

add_executable(
    transformer_cpu
    src/cpu/CPUInference.cpp
)

target_link_libraries(
    transformer_cpu
    PRIVATE
        transformer_core
)

# --------------------------------------------------
# Tests
# --------------------------------------------------

add_executable(
    tensor_tests
    tests/tensor_tests.cpp
)

target_link_libraries(
    tensor_tests
    PRIVATE
        transformer_core
)

add_executable(
    layer_tests
    tests/layer_tests.cpp
)

target_link_libraries(
    layer_tests
    PRIVATE
        transformer_core
)

add_executable(
    attention_tests
    tests/attention_tests.cpp
)

target_link_libraries(
    attention_tests
    PRIVATE
        transformer_core
)

add_executable(
    transformer_tests
    tests/transformer_tests.cpp
)

target_link_libraries(
    transformer_tests
    PRIVATE
        transformer_core
)

# --------------------------------------------------
# Benchmark
# --------------------------------------------------

add_executable(
    cpu_benchmark
    benchmarks/cpu_benchmark.cpp
)

target_link_libraries(
    cpu_benchmark
    PRIVATE
        transformer_core
)

# --------------------------------------------------
# Compiler warnings
# --------------------------------------------------

if (MSVC)

    target_compile_options(
        transformer_core
        PRIVATE
            /W4
    )

else()

    target_compile_options(
        transformer_core
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

endif()