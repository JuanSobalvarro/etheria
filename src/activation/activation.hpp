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

    static Tensor derivative(const Tensor& input, ActivationFunction func)
    {
        switch (func)
        {
        case ActivationFunction::LINEAR:
            return linear_derivative(input);
        case ActivationFunction::RELU:
            return relu_derivative(input);
        case ActivationFunction::SIGMOID:
            return sigmoid_derivative(input);
        case ActivationFunction::TANH:
            return tanh_derivative(input);
        default:
            throw std::invalid_argument("Unsupported activation function for derivative");
        }
    }

    static Tensor linear(const Tensor& input)
    {
        return input; // linear activation is identity
    }

    static Tensor linear_derivative(const Tensor& input)
    {
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            result.set(i, 1.0f); // derivative of linear is 1
        }
        return result;
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

    static Tensor relu_derivative(const Tensor& input)
    {
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            float val = result.get(i);
            result.set(i, val > 0.0f ? 1.0f : 0.0f);
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

    static Tensor sigmoid_derivative(const Tensor& input)
    {
        Tensor sig = sigmoid(input);
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            float s = sig.get(i);
            result.set(i, s * (1.0f - s));
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

    static Tensor tanh_derivative(const Tensor& input)
    {
        Tensor result = input; // copy input tensor
        for (int i = 0; i < result.size(); ++i)
        {
            float val = result.get(i);
            float t = std::tanh(val);
            result.set(i, 1.0f - t * t);
        }
        return result;
    }
};

} // namespace eth
