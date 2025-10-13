// tensor.cpp
#include "tensor.hpp"
#include <stdexcept>


using namespace eth;

Tensor::Tensor(std::vector<size_t> shape): shape(shape) 
{
    size_t total = 1;
    for (auto s : shape) total *= s;
    data.resize(total);
}

void Tensor::add(const Tensor& other) 
{
    if (data.size() != other.data.size()) 
        throw std::runtime_error("Shape mismatch between " + std::to_string(shape.size()) + " and " + std::to_string(other.shape.size()));
    for (size_t i = 0; i < data.size(); ++i)
        data[i] += other.data[i];
    
}

void Tensor::matmul(const Tensor& other) 
{
    if (shape.size() != 2 || other.shape.size() != 2 || shape[1] != other.shape[0]) 
        throw std::runtime_error("Shape mismatch for matmul between " + std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + " and " + std::to_string(other.shape[0]) + "x" + std::to_string(other.shape[1]));

    size_t m = shape[0];
    size_t n = shape[1];
    size_t p = other.shape[1];

    std::vector<float> result(m * p, 0.0f);

    for (size_t i = 0; i < m; ++i) 
    {
        for (size_t j = 0; j < p; ++j) 
        {
            for (size_t k = 0; k < n; ++k) 
            {
                result[i * p + j] += data[i * n + k] * other.data[k * p + j];
            }
        }
    }

    shape = {m, p};
    data = std::move(result);
}

void Tensor::fill(float value) 
{
    std::fill(data.begin(), data.end(), value);
}

float Tensor::get_item(size_t index) const 
{
    if (index >= data.size()) 
        throw std::out_of_range("Index out of range");

    return data[index];
}

void Tensor::set_item(size_t index, float value) 
{
    if (index >= data.size()) 
        throw std::out_of_range("Index out of range");
        
    data[index] = value;
}
