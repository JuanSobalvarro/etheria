file(GLOB SRC_CPP "*.cpp")
file(GLOB SRC_CUDA "*.cu")

add_library(eth_algebra STATIC ${SRC_CPP} ${SRC_CUDA})
target_include_directories(eth_algebra PUBLIC ../../include)
target_link_libraries(eth_algebra PRIVATE CUDA::cudart)
