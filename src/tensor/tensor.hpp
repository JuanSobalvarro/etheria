/*
* Higher level Tensor class built on top of ITensor
* which provides tensor math operations.
*/
#pragma once

#include "itensor.hpp"


namespace eth
{

// Tensor class for multidimensional arrays of float data
class Tensor : public ITensor
{
public:
    // TODO: implement automatic differentiation support we only store a flag for now
    bool requires_grad = false; // flag for automatic differentiation

    Tensor(std::vector<int> shape, bool requires_grad);
    Tensor(float value, bool requires_grad); // rank 0 tensor (scalar)
    Tensor(std::vector<float> data, bool requires_grad); // infers shape as 1D tensor, tensor of rank 1 (vector)
    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad); // tensor with rank > 1

    // copy operations
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // move operations
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // math operations CPU side only for now
    Tensor add(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const; // element-wise multiplication
    Tensor outer_product(const Tensor& other) const; // aka tensor product 
    Tensor dot_product(const Tensor& other) const; // tensor dot product
    Tensor transpose(const std::vector<int>& axes) const; // permute dimensions
    Tensor contraction(const std::vector<std::pair<int, int>>& axes) const; // tensor contraction along specified axes

private:
    // Increment multi-dimensional index "odometer-style"
    static bool next_index(std::vector<int>& idx, const std::vector<int>& shape)
    {
        for (int i = (int)idx.size() - 1; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < shape[i]) return true;
            idx[i] = 0;
        }
        return false; // finished all combinations
    }
};

//
} // namespace eth