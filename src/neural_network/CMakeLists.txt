file(GLOB SRC_CPP "*.cpp")

add_library(eth_nn STATIC ${SRC_CPP})
target_include_directories(eth_nn PUBLIC ../../include)
target_link_libraries(eth_nn PRIVATE eth_algebra)
