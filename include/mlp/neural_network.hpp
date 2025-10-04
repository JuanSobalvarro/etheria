#pragma once

#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include "activation/activation.hpp"
#include "layer.hpp"
#include "types.hpp"
#include "alg/ops.hpp"
#include "mlp/cpu_model.hpp"  // CPU batch helpers
#include "cuda/gpu_model.cuh"
#include "cuda/cuda_helper.cuh"
#include "utils.hpp"
#include "initialization/initialization.hpp"
#include <memory>

namespace eth::mlp { // multi-layer perceptron (MLP) namespace

enum class Backend { CPU, CUDA };

struct NeuralNetworkConfig 
{
    int input_size;
    std::vector<Layer> layers; // hidden + output layers
    Backend backend = Backend::CPU; // default CPU
    int device_id = -1; // Default 0 but when CPU is used this will be ignored
    bool verbose = false;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const NeuralNetworkConfig& cfg);

    // config
    NeuralNetworkConfig& getConfig() { return config; }
    void setVerbose(bool v) { config.verbose = v; }

    // Forward pass – returns output activations
    Vector predict(const Vector& input) const;

    void useCUDADevice(int device_id = 0); // switch to CUDA backend if available
    void useCPU();                         // switch back to CPU backend

    // Train with stochastic gradient descent (one batch at a time)
    void fit(const Matrix& inputs,
             const Matrix& targets,
             int epochs,
             float learning_rate,
             float momentum = 0.9f,
             bool verbose = true);

    void evaluate(const Matrix& inputs,
                  const Matrix& targets,
                  double& loss,
                  double& accuracy) const;

    // Accessors
    const std::vector<Matrix>& getWeights() const;
    const std::vector<Vector>& getBiases() const;

private:
    NeuralNetworkConfig config;

    // GPU model for CUDA backend
    std::unique_ptr<eth::mlp::cuda::GPUModel> gpu_model;
    std::unique_ptr<eth::mlp::cpu::CPUModel> cpu_model;

    // initialization helper
    void initializeNN();
};

} // namespace eth::mlp
