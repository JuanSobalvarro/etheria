#pragma once

#include "tensor/tensor.hpp"
#include <string>

namespace eth
{

Tensor add(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

Tensor activation(const Tensor& input, const std::string& activation);

}