add_executable(test_fit fit.cpp)

# Link your core library
target_link_libraries(test_fit PRIVATE eth_lib CUDA::cudart)

# Include headers
target_include_directories(test_fit PRIVATE ../include)
