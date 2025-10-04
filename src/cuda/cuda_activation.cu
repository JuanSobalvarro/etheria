#include <cuda_runtime.h>
#include "activation/activation.hpp"

namespace eth::act::cuda
{

// ======================================================
// In-place kernels
// ======================================================
__global__ void linear_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // identity, nothing changes
        data[idx] = data[idx];
    }
}

__global__ void der_linear_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = 1.0f;
    }
}

__global__ void relu_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void der_relu_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = (data[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

__global__ void sigmoid_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void der_sigmoid_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float sig = 1.0f / (1.0f + expf(-data[idx]));
        data[idx] = sig * (1.0f - sig);
    }
}

__global__ void tanh_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = tanhf(data[idx]);
    }
}

__global__ void der_tanh_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float t = tanhf(data[idx]);
        data[idx] = 1.0f - t * t;
    }
}

__global__ void softplus_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = logf(1.0f + expf(data[idx]));
    }
}

__global__ void der_softplus_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// ======================================================
// Batch dispatcher (in-place)
// ======================================================
void batch_forward_activation(
    act::ActivationFunctionType act_type,
    float* data,
    int size
)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    switch (act_type)
    {
    case ActivationFunctionType::LINEAR:
        linear_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::RELU:
        relu_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::SIGMOID:
        sigmoid_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::TANH:
        tanh_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::SOFTPLUS:
        softplus_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    default:
        // unknown activation
        break;
    }
}

void batch_derivative_activation(
    act::ActivationFunctionType act_type,
    float* data,
    int size
)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    switch (act_type)
    {
    case ActivationFunctionType::LINEAR:
        der_linear_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::RELU:
        der_relu_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::SIGMOID:
        der_sigmoid_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::TANH:
        der_tanh_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    case ActivationFunctionType::SOFTPLUS:
        der_softplus_kernel<<<numBlocks, blockSize>>>(data, size);
        break;
    default:
        // unknown activation
        break;
    }
}

} // namespace eth::act::cuda
