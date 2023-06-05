cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cnn_from_scratch)
set(version 1.0)

add_compile_options(
    -std=c++17 
    # -fsanitize=address 
    -fno-omit-frame-pointer
)

find_package(cpp_timer)

include_directories(
    include
    src
    ${cpp_timer_INCLUDE_DIRS}
)

enable_testing()
find_package(GTest REQUIRED)

add_executable(main src/main.cpp src/MNISTReader.cpp)
target_compile_options(main PRIVATE -g -O3)
target_link_libraries(main -fsanitize=address ${cpp_timer_LIBRARIES})

include(GoogleTest)
add_executable(MatrixTest test/MatrixTest.cpp)
target_link_libraries(MatrixTest GTest::gtest GTest::gtest_main -fsanitize=address)
gtest_discover_tests(MatrixTest)

add_executable(ModelLayerTest test/ModelLayerTest.cpp)
target_link_libraries(ModelLayerTest GTest::gtest GTest::gtest_main -fsanitize=address ${cpp_timer_LIBRARIES})
gtest_discover_tests(ModelLayerTest)

add_executable(ForwardPassTest test/ForwardPassTest.cpp)
target_link_libraries(ForwardPassTest GTest::gtest GTest::gtest_main -fsanitize=address ${cpp_timer_LIBRARIES})
gtest_discover_tests(ForwardPassTest)

add_executable(rvalue_detection src/rvalue_detection.cpp)
target_compile_options(rvalue_detection PRIVATE -g)
target_link_libraries(rvalue_detection -fsanitize=address)