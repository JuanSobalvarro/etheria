#include "initialization/initialization.hpp"
#include "utils.hpp"
#include <cmath>
#include <stdexcept>
#include <random>

namespace eth::init
{

// Xavier initialization for sigmoid/tanh/linear
Vector xavier_initialization(int fan_in, int fan_out) 
{
    Vector vals(fan_in); // one vector per neuron
    Scalar limit = std::sqrt(6.0 / (fan_in + fan_out)); // correct Xavier bound

    std::uniform_real_distribution<Scalar> dist(-limit, limit);

    for (auto &v : vals)
        v = dist(utils::global_rng());

    return vals;
}

// He initialization for ReLU
Vector he_initialization(int fan_in, int /*fan_out*/) 
{
    Vector vals(fan_in); // one vector per neuron
    Scalar stddev = std::sqrt(2.0 / fan_in);

    std::normal_distribution<Scalar> dist(0.0, stddev);

    for (auto &v : vals)
        v = dist(utils::global_rng());

    return vals;
}

// Normal initialization for Softplus
Vector normal_initialization(int fan_in, int /*fan_out*/) 
{
    Vector vals(fan_in);
    Scalar stddev = 1.0 / std::sqrt(fan_in);

    std::normal_distribution<Scalar> dist(0.0, stddev);

    for (auto &v : vals)
        v = dist(utils::global_rng());

    return vals;
}

// Choose initialization based on activation type
Vector initialization_distribution(act::ActivationFunctionType act_type, int fan_in, int fan_out) 
{
    switch (act_type) 
    {
        case act::ActivationFunctionType::LINEAR:
        case act::ActivationFunctionType::SIGMOID:
        case act::ActivationFunctionType::TANH:
        case act::ActivationFunctionType::SOFTMAX:
            return xavier_initialization(fan_in, fan_out);

        case act::ActivationFunctionType::RELU:
            return he_initialization(fan_in, fan_out);

        case act::ActivationFunctionType::SOFTPLUS:
            return normal_initialization(fan_in, fan_out);

        default:
            throw std::invalid_argument("Unknown activation function type for initialization");
    }
}

} // namespace eth::init