// This file is part of the Etheria module cuda.
// It contains helpers related to CUDA
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>


// define macro for checking CUDA errors
inline void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[CUDA] ") + cudaGetErrorString(err) +
            " at " + file + ":" + std::to_string(line));
    }
}

#define CUDA_CALL(expr) \
    check_cuda_error((expr), __FILE__, __LINE__)

namespace eth::cuda 
{

// device management
bool isCUDAAvailable();
int numberCUDADevices();
bool isCUDACompatible(int device_id);
std::vector<std::string> listCUDADevices();
void setDevice(int device_id);
int currentCUDADevice();
std::string deviceDetails(int device_id);

// memory management
void allocate_device_memory(void** dev_ptr, size_t size);
void free_device_memory(void* dev_ptr);
void copy_to_device(void* dest, const void* src, size_t size);
void copy_to_host(void* dest, const void* src, size_t size);

} // namespace eth::cuda
