cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cnn_from_scratch)
set(version 1.0)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(
    # -fsanitize=address 
    -fno-omit-frame-pointer
)

option(ENABLE_CUDA_ACCELERATION "Enable CUDA GPU acceleration" ON)

# Check for CUDA
enable_language(CUDA OPTIONAL)
include(CheckLanguage)
check_language(CUDA)
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

# If enabled and present, add CUDA to the project
if (CMAKE_CUDA_COMPILER AND ("CUDA" IN_LIST languages) AND ENABLE_CUDA_ACCELERATION)
    add_compile_definitions(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    message("Compiling with CUDA")
else()
    message("Compiling without CUDA")
endif()

find_package(cpp_timer)

include_directories(
    include
    src
    ${cpp_timer_INCLUDE_DIRS}
)

add_compile_definitions(DATA_DIR="${CMAKE_SOURCE_DIR}/data")

add_library(ModelLayers SHARED
    src/Layers/Pooling.cpp
    src/Layers/Activation.cpp
    src/Layers/Softmax.cpp
    src/Layers/ConnectedLayer.cpp
    src/Layers/Kernel.cpp
    src/timerConfig.cpp
)
target_link_libraries(ModelLayers ${cpp_timer_LIBRARIES})
target_compile_options(ModelLayers PRIVATE -g -O3)

set(CORE_LIBS
    ${cpp_timer_LIBRARIES}
    ModelLayers
    -fsanitize=address
)

enable_testing()
find_package(GTest REQUIRED)

add_executable(main src/main.cpp src/MNISTReader.cpp)
target_compile_options(main PRIVATE -g -O3)
target_link_libraries(main ${CORE_LIBS})

add_executable(cudaTest src/cudaTest.cu)
target_compile_options(cudaTest PRIVATE -O3)

include(GoogleTest)
add_executable(MatrixTest test/MatrixTest.cpp)
target_link_libraries(MatrixTest GTest::gtest GTest::gtest_main -fsanitize=address ModelLayers)
gtest_discover_tests(MatrixTest)

add_executable(ModelLayerTest test/ModelLayerTest.cpp)
target_link_libraries(ModelLayerTest GTest::gtest GTest::gtest_main -fsanitize=address ${cpp_timer_LIBRARIES} ModelLayers)
gtest_discover_tests(ModelLayerTest)

add_executable(ForwardPassTest test/ForwardPassTest.cpp)
target_link_libraries(ForwardPassTest GTest::gtest GTest::gtest_main -fsanitize=address ${cpp_timer_LIBRARIES} ModelLayers)
gtest_discover_tests(ForwardPassTest)

add_executable(BackwardPassTest test/BackwardPassTest.cpp)
target_link_libraries(BackwardPassTest GTest::gtest GTest::gtest_main -fsanitize=address ${cpp_timer_LIBRARIES} ModelLayers)
gtest_discover_tests(BackwardPassTest)

add_executable(rvalue_detection src/rvalue_detection.cpp)
target_compile_options(rvalue_detection PRIVATE -g)
target_link_libraries(rvalue_detection -fsanitize=address)