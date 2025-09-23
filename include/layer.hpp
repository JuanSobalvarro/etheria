/*
* This file is used when we are creating a neural network, we do not operate over this layer object
* directly, only used for initialization (constructor of NeuralNetwork), stores metadata about the layer
* such as number of neurons and activation function type.
*/
#pragma once

#include "activation_functions.hpp"
#include <vector>

namespace eth::mlp
{
class Layer
{
public:
    Layer(int units, act::ActivationFunctionType activation)
        : units(units), activation(activation) {}
    
    int getUnits() const { return units; }
    act::ActivationFunctionType getActivation() const { return activation; }
private:
    int units; // number of neurons in this layer
    act::ActivationFunctionType activation;

};

} // namespace eth
