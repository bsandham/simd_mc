cmake_minimum_required(VERSION 3.25)
project(simd_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(simd_mc INTERFACE)
target_include_directories(simd_mc INTERFACE src/)
target_compile_options(simd_mc INTERFACE
    -O3
    -march=native
    -ffast-math
    -Wall -Wextra -Wpedantic
    -std=c++23
)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE simd_mc)


add_executable(basic_barrier benchmarks/basic_barrier.cpp )
target_link_libraries(basic_barrier PRIVATE simd_mc)
add_executable(benchmark_crossover benchmarks/benchmark_crossover.cpp )
target_link_libraries(benchmark_crossover PRIVATE simd_mc)
enable_testing()
add_executable(test_vanilla src/tests/test_vanilla.cpp )
target_link_libraries(test_vanilla PRIVATE simd_mc)
add_test(NAME vanilla_convergence COMMAND test_vanilla)
add_executable(test_barrier src/tests/test_barrier.cpp )
target_link_libraries(test_barrier PRIVATE simd_mc)
add_test(NAME barrier_convergence COMMAND test_barrier)
