// This file is part of the Etheria module cuda.
// It contains helpers related to CUDA
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
int currentCUDADevice();
std::string deviceDetails(int device_id);

} // namespace eth::cuda
