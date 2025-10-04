#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "mlp/layer.hpp"
#include "cuda/cuda_helper.cuh"
#include "activation/activation.hpp"
#include "types.hpp"
#include <memory>
#include <stdexcept>
#include "initialization/initialization.hpp"

namespace eth::mlp::cuda
{

    // DEVICE DATA STRUCTURES

struct DeviceMatrix {
    float* data;  // flattened [rows * cols]
    int rows;
    int cols;

    #ifdef __CUDACC__
        __host__ __device__ __forceinline__ float& operator()(int r, int c) {
            return data[r * cols + c];
        }
        __host__ __device__ __forceinline__ const float& operator()(int r, int c) const {
            return data[r * cols + c];
        }
    #else
        inline float& operator()(int r, int c) {
            return data[r * cols + c];
        }
        inline const float& operator()(int r, int c) const {
            return data[r * cols + c];
        }
    #endif
};

struct DeviceVector {
    float* data; 
    int size;

    #ifdef __CUDACC__
        __host__ __device__ __forceinline__ float& operator()(int i) {
            return data[i];
        }

        __host__ __device__ __forceinline__ const float& operator()(int i) const {
            return data[i];
        }
    #else
        inline float& operator()(int i) {
            return data[i];
        }

        inline const float& operator()(int i) const {
            return data[i];
        }
    #endif
};

// MODEL

struct GPUModel
{
    DeviceMatrix* weights;  // array of weight matrices (one per layer)
    DeviceVector* biases;   // array of bias vectors (one per layer)
    int num_layers;
    int input_size;         // size of input layer (implicit, not stored in Layer)
    std::vector<Layer> layers; // host-side layer metadata (activations, sizes)
};

// life cycle management

GPUModel createModelOnGPU(
    const int input_size,
    const std::vector<Layer>& layers
);

GPUModel copyModelToGPU(
    const std::vector<Matrix>& weights,
    const std::vector<Vector>& biases,
    const int input_size,
    const std::vector<Layer>& layers
);

void freeModelOnGPU(GPUModel& model);

// operations

Matrix forward_batch(
    const GPUModel& model,
    const Matrix& input_batch
);

float fit_batch(
    GPUModel& model,
    const Matrix& input_batch,
    const Matrix& target_batch,
    float learning_rate
);

void forward_layer(
    const float* weights,   // flattened [output_size * input_size]
    const float* biases,    // [output_size]
    const float* inputs,    // [input_size]
    float* outputs,         // [output_size]
    int input_size,
    int output_size,
    act::ActivationFunctionType act_type
);

void output_delta(
    const float* outputs,
    const float* targets,
    float* deltas,
    int output_size,
    int batch_size,
    act::ActivationFunctionType act_type
);

void hidden_deltas(
    const float* weights,
    const float* next_deltas,
    const float* outputs,
    float* deltas,
    int input_size,
    int output_size,
    int batch_size,
    act::ActivationFunctionType act_type
);

void update_weights(
    float* weights,
    const float* biases,
    const float* inputs,
    const float* deltas,
    int input_size,
    int output_size,
    int batch_size,
    float learning_rate
);

// utils

std::vector<Matrix> getWeightsFromGPU(const GPUModel& model);
std::vector<Vector> getBiasesFromGPU(const GPUModel& model);

} // namespace eth::mlp::cuda
