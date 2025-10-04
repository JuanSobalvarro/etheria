#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace eth::cuda {

bool isCUDAAvailable();
int numberCUDADevices();
bool isCUDACompatible(int device_id);
std::vector<std::string> listCUDADevices();
void checkCuda(cudaError_t err, const char* msg = nullptr);
void setDevice(int device_id);

} // namespace eth::cuda
