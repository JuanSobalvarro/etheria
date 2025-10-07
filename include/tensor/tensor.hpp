#pragma once

#include "types.hpp"
#include <vector>

namespace eth::tensor
{

class Tensor
{
public:
    Tensor(const std::vector<size_t>& shape, eth::DType dtype);
    ~Tensor();

private:
    eth::DType dtype;
    std::vector<size_t> shape;
    void* data;
};


} // namespace eth::tensor