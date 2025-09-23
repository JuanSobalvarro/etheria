// Matrix-based neural network (lightweight) – avoids per-neuron/connection objects
// This co-exists with the object graph implementation for easier comparison.
#pragma once

#include <vector>
#include <random>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include "activation_functions.hpp"
#include "layer.hpp"

namespace eth::mlp { // neural from scratch

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>; // row-major: rows x cols, rows = out features

struct NeuralNetworkConfig {
    int input_size;
    std::vector<Layer> layers; // hidden + output layers
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const NeuralNetworkConfig& cfg);

    // Forward pass – returns output activations
    Vector predict(const Vector& input) const;

    // Train with stochastic gradient descent (one sample at a time)
    void train(const std::vector<Vector>& inputs,
               const std::vector<Vector>& targets,
               int epochs,
               double learning_rate,
               bool verbose = true);
    void evaluate(const std::vector<Vector>& inputs,
                  const std::vector<Vector>& targets,
                  double& loss,
                  double& accuracy) const;

    // Accessors
    const std::vector<Matrix>& getWeights() const { return weights; }
    const std::vector<Vector>& getBiases() const { return biases; }

private:
    NeuralNetworkConfig config;
    std::vector<Matrix> weights; // W[l] shape: layer_sizes[l+1] x layer_sizes[l]
    std::vector<Vector> biases;  // b[l] length: layer_sizes[l+1]

    // Helper utilities
    static double activation(double x, act::ActivationFunctionType type);
    static double activation_derivative(double x, act::ActivationFunctionType type); // derivative wrt pre-activation z
    static Vector applyActivation(const Vector& z, act::ActivationFunctionType type);
    static Vector applyActivationDerivative(const Vector& z, act::ActivationFunctionType type);

    static Vector matvec(const Matrix& W, const Vector& v);              // W * v
    static Matrix outer(const Vector& a, const Vector& b);                // a (rows) * b^T
    static Vector add(const Vector& a, const Vector& b);
    static void inplace_axpy(Vector& y, const Vector& x, double alpha);   // y += alpha * x
};

} // namespace nn
