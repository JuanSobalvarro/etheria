/*
* This file contains the activation functions used in neural networks for CPU.
*/
#pragma once

#include "types.hpp"

namespace eth::act 
{

enum ActivationFunctionType {
    LINEAR   = 0,
    SIGMOID  = 1,
    RELU     = 2,
    TANH     = 3,
    SOFTPLUS = 4,
    SOFTMAX   = 5 
};

float linear(float x);
float der_linear(float x);

float sigmoid(float x);
float der_sigmoid(float x);

float relu(float x);
float der_relu(float x);

float tanh_act(float x);
float der_tanh(float x);

float softplus(float x);
float der_softplus(float x);

Vector softmax(const Vector& x);
Vector der_softmax(const Vector& x, int index);

float forward_activation(ActivationFunctionType act_type, float x);
float forward_activation_derivative(ActivationFunctionType act_type, float x);

Vector forward_activation(ActivationFunctionType act_type, const Vector& x);
Vector forward_activation_derivative(ActivationFunctionType act_type, const Vector& x);

void inplace_forward_activation(ActivationFunctionType act_type, Vector& x);
void inplace_forward_activation_derivative(ActivationFunctionType act_type, Vector& x);

} // namespace eth::activation