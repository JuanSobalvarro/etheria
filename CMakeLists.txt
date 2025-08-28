cmake_minimum_required(VERSION 3.14)
project(NeuralFromScratch LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# === Sources ===
file(GLOB SRC_FILES "src/*.cpp")

# Add Python module target
add_subdirectory(extern/pybind11)
pybind11_add_module(neuralscratch bindings/bindings.cpp ${SRC_FILES})

target_compile_definitions(neuralscratch PRIVATE PYBIND11_CPP_LOCAL_ERROR_REPORTING=0)

# === Optional compile flags ===
if(MSVC)
    # Windows compiler flags
    target_compile_options(neuralscratch PRIVATE /W3 /O2)
else()
    target_compile_options(neuralscratch PRIVATE -Wall -O2 -g)
endif()
