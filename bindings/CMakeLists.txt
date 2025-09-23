pybind11_add_module(etheria bindings.cpp)

target_link_libraries(etheria PRIVATE eth_nn eth_algebra CUDA::cudart)
target_include_directories(etheria PRIVATE ../include)
