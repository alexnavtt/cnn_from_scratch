cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cnn_from_scratch)
set(version 1.0)

set(CXX_STANDARD c++17)

include_directories(
    include
)

add_executable(main src/main.cpp)
target_compile_options(main PRIVATE -g)
