#include "cuda/cuda_helper.cuh"
#include <cstdio>
#include <stdexcept>

namespace eth::cuda 
{

// device management

bool isCUDAAvailable() 
{
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    return (deviceCount > 0);
}

int numberCUDADevices() 
{
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

bool isCUDACompatible(int device_id) 
{
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    if (device_id < 0 || device_id >= deviceCount)
        return false;

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_id));
    return deviceProp.major >= 3;
}

std::vector<std::string> listCUDADevices() 
{
    std::vector<std::string> deviceNames;
    int deviceCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        CUDA_CALL(cudaGetDeviceProperties(&deviceProp, i));
        deviceNames.push_back(deviceProp.name);
    }
    return deviceNames;
}

void setDevice(int device_id) 
{
    int device_count = 0;
    CUDA_CALL(cudaGetDeviceCount(&device_count));
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error("Invalid CUDA device ID");
    CUDA_CALL(cudaSetDevice(device_id));
}

int currentCUDADevice() 
{
    int device_id = -1;
    CUDA_CALL(cudaGetDevice(&device_id));
    return device_id;
}

std::string deviceDetails(int device_id) 
{
    int device_count = 0;
    CUDA_CALL(cudaGetDeviceCount(&device_count));
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error("Invalid CUDA device ID");

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_id));

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

// memory management

// allocate memory on current cuda device
void allocate_device_memory(void** dev_ptr, size_t size) 
{
    CUDA_CALL(cudaMalloc(dev_ptr, size));
}

// free memory on current cuda device
void free_device_memory(void* dev_ptr) 
{
    CUDA_CALL(cudaFree(dev_ptr));
}

// copy memory between host and CURRENT device
void copy_to_device(void* dest, const void* src, size_t size) 
{
    CUDA_CALL(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

// copy memory between CURRENT device and host
void copy_to_host(void* dest, const void* src, size_t size) 
{
    CUDA_CALL(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
}

} // namespace eth::cuda
