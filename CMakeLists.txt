cmake_minimum_required(VERSION 3.20)
project(etheria LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(src)

# CUDA
find_package(CUDAToolkit REQUIRED)


file(GLOB_RECURSE ETH_SRC CONFIGURE_DEPENDS
src/*.cpp
src/*.cu
)
# exclude the bindings folder
list(FILTER ETH_SRC EXCLUDE REGEX "bindings/.*")

# etheria library

add_library(eth_lib STATIC ${ETH_SRC})
target_include_directories(eth_lib PUBLIC include)
target_link_libraries(eth_lib PRIVATE CUDA::cudart)

# Pybind11
add_subdirectory(extern/pybind11)

file(GLOB_RECURSE BINDINGS_SRC CONFIGURE_DEPENDS src/bindings/*.cpp)
pybind11_add_module(_etheria ${BINDINGS_SRC})
target_link_libraries(_etheria PRIVATE eth_lib CUDA::cudart pybind11::module)
target_include_directories(_etheria PRIVATE include)

