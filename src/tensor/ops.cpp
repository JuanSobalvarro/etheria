#include "tensor/ops.hpp"

#include <stdexcept>
#include <cmath>

namespace eth
{

Tensor add(const Tensor& a, const Tensor& b) 
{
    Tensor result = a; // Copy a to result
    result.add(b);    // In-place add b to result
    return result;
}
Tensor matmul(const Tensor& a, const Tensor& b) 
{
    Tensor result = a;
    result.matmul(b);
    return result;
}

Tensor activation(const Tensor& input, const std::string& activation) 
{
    Tensor result = input; // Copy input to result
    if (activation == "linear")
    {
        // Do nothing, linear activation is just the identity function
    }
    else if (activation == "relu") 
    {
        for (size_t i = 0; i < result.data.size(); ++i) 
        {
            result.data[i] = std::max(0.0f, result.data[i]);
        }
    } 
    else if (activation == "sigmoid") 
    {
        for (size_t i = 0; i < result.data.size(); ++i) 
        {
            result.data[i] = 1.0f / (1.0f + std::exp(-result.data[i]));
        }
    } 
    else if (activation == "tanh") 
    {
        for (size_t i = 0; i < result.data.size(); ++i) 
        {
            result.data[i] = std::tanh(result.data[i]);
        }
    } 
    else 
    {
        throw std::invalid_argument("Unsupported activation function: " + activation);
    }
    return result;
}

} // namespace eth