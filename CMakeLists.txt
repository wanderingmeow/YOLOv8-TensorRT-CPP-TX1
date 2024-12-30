cmake_minimum_required(VERSION 3.22)
project(YOLOv8_TensorRT_CPP)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CMake Options
option(ENABLE_BENCHMARKS "Benchmark the preprocessing, inference, and post processing" OFF)
if (ENABLE_BENCHMARKS)
    add_compile_definitions(ENABLE_BENCHMARKS)
endif()

# Eign3 is required for OpenCV built with Eigen3
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 was not found. Run 'sudo apt-get install libeigen3-dev'.")
endif()

# Only use required components from OpenCV
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs cudaarithm cudaimgproc cudawarping
    videoio dnn highgui
)

# Build the TensorRT inference engine library
# TensorRT is found and linked by the tensorrt-cpp-api
add_subdirectory(libs/tensorrt-cpp-api)

# Build the YoloV8 library
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api ${OpenCV_LIBS})
target_include_directories(YoloV8_TRT PUBLIC libs/tensorrt-cpp-api/src)

# Build and link the executables
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark YoloV8_TRT)

add_executable(detect_object_image src/object_detection_image.cpp)
target_link_libraries(detect_object_image YoloV8_TRT)

add_executable(detect_object_video src/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video YoloV8_TRT)

add_compile_definitions(CLI_UseBoostAsio)
add_compile_definitions(CLI_USE_BOOSTASIO_SCHEDULER)
set(Boost_NO_BOOST_CMAKE ON)
add_definitions( -DBOOST_ALL_NO_LIB ) # for windows
find_package(Boost CONFIG REQUIRED)

add_subdirectory(libs/cli/)

add_executable(cli_app src/cli_app.cpp)
target_include_directories(cli_app PUBLIC ${cli_INCLUDE_DIRS})
target_link_libraries(cli_app YoloV8_TRT cli)
