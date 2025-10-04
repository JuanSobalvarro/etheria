#include "cuda/cuda_helper.cuh"
#include <cstdio>
#include <stdexcept>

namespace eth::cuda 
{

bool isCUDAAvailable() 
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return (error_id == cudaSuccess && deviceCount > 0);
}

int numberCUDADevices() 
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    return (error_id == cudaSuccess) ? deviceCount : 0;
}

bool isCUDACompatible(int device_id) 
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || device_id < 0 || device_id >= deviceCount)
        return false;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    return deviceProp.major >= 3;
}

std::vector<std::string> listCUDADevices() 
{
    std::vector<std::string> deviceNames;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) return deviceNames;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        deviceNames.push_back(deviceProp.name);
    }
    return deviceNames;
}

void checkCuda(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess) {
        if (msg) fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        else fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

void setDevice(int device_id) 
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error("Invalid CUDA device ID");
    cudaSetDevice(device_id);
}

} // namespace eth::cuda
