/*
* This file is part of the Neural Network library. This contains the activation functions
* used in the neural network.
*/
#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

// NOTE: Original OO interface (ActivationFunction subclasses + factory) is kept
// for backward compatibility with the object-based network. For the new
// matrix-based implementation (and any future performance-oriented code), a
// lightweight functional API is provided below to avoid heap allocations and
// virtual dispatch per activation.

enum ActivationFunctionType {
    LINEAR = 0,
    SIGMOID = 1,
    RELU = 2,
    TANH = 3,
    SOFTPLUS = 4
};


// Activation function interface
class ActivationFunction {
public:
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual ~ActivationFunction() {}
};

class Linear : public ActivationFunction {
public:
    double activate(double x) const override {
        return x;
    }

    double derivative(double x) const override {
        return 1.0;
    }
};

// Sigmoid activation function
class Sigmoid : public ActivationFunction {
public:
    double activate(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double derivative(double x) const override {
        double sig = activate(x);
        return sig * (1 - sig);
    }
};

// ReLU activation function
class ReLU : public ActivationFunction {
public:
    double activate(double x) const override {
        return x > 0 ? x : 0;
    }

    double derivative(double x) const override {
        return x > 0 ? 1 : 0;
    }
};

// Tanh activation function
class Tanh : public ActivationFunction {
public:
    double activate(double x) const override {
        return std::tanh(x);
    }

    double derivative(double x) const override {
        return 1 - std::pow(std::tanh(x), 2);
    }
};

class Softplus : public ActivationFunction {
public:
    double activate(double x) const override {
        return std::log(1 + std::exp(x));
    }

    double derivative(double x) const override {
        return 1 / (1 + std::exp(-x));
    }
};

class ActivationFactory {
public:
    static ActivationFunction* createActivationFunction(const ActivationFunctionType& type) {
        switch (type) {
            case LINEAR:
                return new Linear();
            case SIGMOID:
                return new Sigmoid();
            case RELU:
                return new ReLU();
            case TANH:
                return new Tanh();
            case SOFTPLUS:
                return new Softplus();
            default:
                throw std::invalid_argument("Invalid activation function type");
        }
        
    }
};

// ---------------------------------------------------------------------------
// Functional (stateless) activation utilities
// ---------------------------------------------------------------------------
namespace act {

// Scalar forward
inline double forward(ActivationFunctionType type, double x) {
    switch (type) {
        case LINEAR:   return x;
        case SIGMOID:  return 1.0 / (1.0 + std::exp(-x));
        case RELU:     return x > 0 ? x : 0.0;
        case TANH:     return std::tanh(x);
        case SOFTPLUS: return std::log1p(std::exp(x));
        default:       return x;
    }
}

// Scalar derivative (with respect to pre-activation z)
inline double derivative(ActivationFunctionType type, double x) {
    switch (type) {
        case LINEAR:   return 1.0;
        case SIGMOID:  { double s = 1.0 / (1.0 + std::exp(-x)); return s * (1.0 - s); }
        case RELU:     return x > 0 ? 1.0 : 0.0;
        case TANH:     { double t = std::tanh(x); return 1.0 - t * t; }
        case SOFTPLUS: { double e = std::exp(-x); return 1.0 / (1.0 + e); }
        default:       return 1.0;
    }
}

// Range helpers (in-place). Accept any contiguous span (C++23 std::span)
template <class Container>
inline void forward_inplace(ActivationFunctionType type, Container& values) {
    for (auto &v : values) v = forward(type, v);
}

template <class Container>
inline void derivative_inplace(ActivationFunctionType type, Container& values) {
    for (auto &v : values) v = derivative(type, v);
}

template <class Container>
inline std::vector<double> forward_copy(ActivationFunctionType type, const Container& src) {
    std::vector<double> out(src.size());
    size_t idx = 0; for (auto &val : src) out[idx++] = forward(type, val);
    return out;
}

template <class Container>
inline std::vector<double> derivative_copy(ActivationFunctionType type, const Container& zvalues) {
    std::vector<double> out(zvalues.size());
    size_t idx = 0; for (auto &val : zvalues) out[idx++] = derivative(type, val);
    return out;
}

} // namespace act

#endif // ACTIVATION_FUNCTIONS_HPP