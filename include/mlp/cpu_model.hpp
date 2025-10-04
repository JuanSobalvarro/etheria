#pragma once

#include "mlp/layer.hpp"
#include "types.hpp"
#include "alg/ops.hpp"
#include <vector>

namespace eth::mlp::cpu
{

struct CPUModel
{
    std::vector<Matrix> weights;  // W[l] is Matrix: neurons_out x neurons_in
    std::vector<Vector> biases;   // b[l] length: neurons_out
    std::vector<Layer> layers;
};

CPUModel create_cpu_model(
    const std::vector<Layer>& layers, 
    int input_size
);

Vector forward(
    const CPUModel& model,
    const Vector& input
);

Matrix forward_batch(
    const CPUModel& model,
    const Matrix& input_batch
);

float fit_batch(
    CPUModel& model,
    const Matrix& input_batch,
    const Matrix& target_batch,
    float learning_rate,
    float momentum = 0.9f
);

void forward_layer(
    const float** weights,
    const float* biases,
    const float* input,
    float* output,
    int input_size,
    int output_size,
    act::ActivationFunctionType act_type
);


std::vector<Matrix> getWeights(const CPUModel& model);

std::vector<Vector> getBiases(const CPUModel& model);

} // namespace eth::mlp::cpu
