/*
* This file contains functions of initialization of neural networks based on the activation function.
*/
#pragma once

#include "types.hpp"
#include "activation/activation.hpp"

namespace eth::init
{

Vector xavier_initialization(int fan_in, int fan_out);
Vector he_initialization(int fan_in, int /*fan_out*/);
Vector normal_initialization(int fan_in, int /*fan_out*/);

Vector initialization_distribution(act::ActivationFunctionType act_type, int fan_in, int fan_out);

} // namespace eth::initialization