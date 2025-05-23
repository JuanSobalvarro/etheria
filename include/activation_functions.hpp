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

#endif // ACTIVATION_FUNCTIONS_HPP