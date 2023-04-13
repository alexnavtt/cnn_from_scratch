cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cnn_from_scratch)
set(version 1.0)

add_compile_options(-std=c++17)

include_directories(
    include
    src
)

enable_testing()
find_package(GTest REQUIRED)

add_executable(main src/main.cpp src/SimpleMatrix.cpp)
target_compile_options(main PRIVATE -g)

include(GoogleTest)
add_executable(MatrixTest test/MatrixTest.cpp src/SimpleMatrix.cpp)
target_link_libraries(MatrixTest GTest::gtest GTest::gtest_main)
gtest_discover_tests(MatrixTest)