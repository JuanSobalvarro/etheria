cmake_minimum_required(VERSION 3.20)
project(etheria LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Pybind11
add_subdirectory(extern/pybind11)

file(GLOB_RECURSE ETH_SRC CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
)

add_library(eth_lib STATIC ${ETH_SRC})
target_include_directories(eth_lib PUBLIC include)
target_link_libraries(eth_lib PRIVATE CUDA::cudart)

file(GLOB_RECURSE BINDINGS_SRC CONFIGURE_DEPENDS bindings/*.cpp)
pybind11_add_module(core ${BINDINGS_SRC})
target_link_libraries(core PRIVATE eth_lib CUDA::cudart)
target_include_directories(core PRIVATE include)

# 4. C++ tests
# add_subdirectory(cpptests)

