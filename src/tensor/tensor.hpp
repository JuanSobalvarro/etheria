// tensor.hpp
#pragma once
#include <vector>
#include <memory>
#include <string>

namespace eth 
{
class Tensor 
{
public:
    std::vector<size_t> shape;
    std::vector<float> data;

    Tensor(std::vector<size_t> shape);
    void fill(float value);
    void add(const Tensor& other);
    void matmul(const Tensor& other);

    float get_item(size_t index) const;
    void set_item(size_t index, float value);
};

} // namespace eth
