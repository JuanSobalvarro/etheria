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

int currentCUDADevice() 
{
    int device_id = -1;
    cudaError_t err = cudaGetDevice(&device_id);
    checkCuda(err, "Failed to get current CUDA device");
    return device_id;
}

std::string deviceDetails(int device_id) 
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error("Invalid CUDA device ID");

    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_id);
    checkCuda(err, "Failed to get device properties");

    char details[1024];
    snprintf(details, sizeof(details),
             "Device %d: %s\n" // 1st line
             "  Compute Capability: %d.%d\n" // 2nd line
             "  Total Global Memory: %.2f GB\n" // 3rd line
             "  Multiprocessors: %d\n" // 4th line
             "  Memory Bus Width: %d bits\n" // 5th line
             "  L2 Cache Size: %d KB\n" // 6th line
             "  Max Threads per Block: %d\n" // 7th line
             "  Max Threads Dimension: [%d, %d, %d]\n" // 8th line
             "  Max Grid Size: [%d, %d, %d]\n", // 9th line
             device_id, deviceProp.name, // 1
             deviceProp.major, deviceProp.minor, // 2
             static_cast<float>(deviceProp.totalGlobalMem) / (1 << 30), // 3
             deviceProp.multiProcessorCount, // 4
             deviceProp.memoryBusWidth, // 5
             deviceProp.l2CacheSize / 1024, // 6
             deviceProp.maxThreadsPerBlock, // 7
             deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2], // 8
             deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]); // 9

    return std::string(details);
}

} // namespace eth::cuda
