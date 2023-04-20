cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cnn_from_scratch)
set(version 1.0)

add_compile_options(
    -std=c++17 
    -fsanitize=address 
    -fno-omit-frame-pointer
)

include_directories(
    include
    src
)

enable_testing()
find_package(GTest REQUIRED)

add_executable(main src/main.cpp)
target_compile_options(main PRIVATE -g)
target_link_libraries(main -fsanitize=address)

include(GoogleTest)
add_executable(MatrixTest test/MatrixTest.cpp)
target_link_libraries(MatrixTest GTest::gtest GTest::gtest_main -fsanitize=address)
gtest_discover_tests(MatrixTest)

add_executable(rvalue_detection src/rvalue_detection.cpp)
target_compile_options(rvalue_detection PRIVATE -g)
target_link_libraries(rvalue_detection -fsanitize=address)