/*
* Header for the activation functions
*/
#pragma once

#include "tensor/tensor.hpp"
#include <stdexcept>
#include <cmath>

namespace eth
{

enum class ActivationFunction
{
    LINEAR = 0,
    RELU = 1,
    SIGMOID = 2,
    TANH = 3,
};

class Activation
{
public:
    static Tensor apply(const Tensor& input, ActivationFunction func)
    {
        switch (func)
        {
        case ActivationFunction::LINEAR:
            return linear(input);
        case ActivationFunction::RELU:
            return relu(input);
        case ActivationFunction::SIGMOID:
            return sigmoid(input);
        case ActivationFunction::TANH:
            return tanh(input);
        default:
            throw std::invalid_argument("Unsupported activation function");
        }
    }

    static Tensor linear(const Tensor& input)
    {
        return input; // linear activation is identity
    }

    static Tensor relu(const Tensor& input)
    {
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            float val = result.get(i);
            result.set(i, val > 0.0f ? val : 0.0f);
        }
        return result;
    }

    static Tensor sigmoid(const Tensor& input)
    {
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            float val = result.get(i);
            result.set(i, 1.0f / (1.0f + std::exp(-val)));
        }
        return result;
    }

    static Tensor tanh(const Tensor& input)
    {
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            float val = result.get(i);
            result.set(i, std::tanh(val));
        }
        return result;
    }
};

} // namespace eth
