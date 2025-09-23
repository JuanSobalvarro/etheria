cmake_minimum_required(VERSION 3.20)
project(etheria LANGUAGES CXX CUDA)

# C++ and CUDA
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# find cuda
find_package(CUDAToolkit REQUIRED)

# add subdirectories
add_subdirectory(src)
add_subdirectory(extern/pybind11)
add_subdirectory(bindings)