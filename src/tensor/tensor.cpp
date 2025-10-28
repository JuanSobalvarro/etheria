#include "tensor.hpp"

namespace eth
{

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : ITensor(shape), requires_grad(requires_grad)
{
}
Tensor::Tensor(float value, bool requires_grad)
    : ITensor(value), requires_grad(requires_grad)
{
}
Tensor::Tensor(std::vector<float> data, bool requires_grad)
    : ITensor(data), requires_grad(requires_grad)
{
}
Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad)
    : ITensor(data, shape), requires_grad(requires_grad)
{
}

// copy operations
Tensor::Tensor(const Tensor& other) 
    : ITensor(other), requires_grad(other.requires_grad) \
{}

Tensor& Tensor::operator=(const Tensor& other) 
{
    ITensor::operator=(other);
    requires_grad = other.requires_grad;
    return *this;
}

// move operations
Tensor::Tensor(Tensor&& other) noexcept
    : ITensor(std::move(other)), requires_grad(other.requires_grad)
{
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        ITensor::operator=(std::move(other));
        requires_grad = other.requires_grad;
    }
    return *this;
}

Tensor Tensor::add(const Tensor& other) const
{
    // For an add operation the both tensors must have the same shape
    if (get_shape() != other.get_shape())
    {
        throw std::invalid_argument("Incompatible tensor shapes");
    }

    Tensor result(get_shape(), requires_grad || other.requires_grad);
    for (size_t i = 0; i < size(); ++i)
    {
        float sum = get(i) + other.get(i);
        result.set(i, sum);
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const
{
    // For an element-wise multiplication operation the both tensors must have the same shape
    if (get_shape() != other.get_shape())
    {
        throw std::invalid_argument("Incompatible tensor shapes");
    }

    Tensor result(get_shape(), requires_grad || other.requires_grad);
    for (size_t i = 0; i < size(); ++i)
    {
        float prod = get(i) * other.get(i);
        result.set(i, prod);
    }
    return result;
}

Tensor Tensor::outer_product(const Tensor& other) const
{
    const std::vector<int>& shape_a = get_shape();
    const std::vector<int>& shape_b = other.get_shape();

    // New tensor shape is concatenation of both shapes
    std::vector<int> new_shape = shape_a;
    new_shape.insert(new_shape.end(), shape_b.begin(), shape_b.end());

    Tensor result(new_shape, requires_grad || other.requires_grad);

    std::vector<int> idx_a(shape_a.size(), 0);
    do {
        std::vector<int> idx_b(shape_b.size(), 0);
        do {
            // Combined index for result tensor
            std::vector<int> idx_result = idx_a;
            idx_result.insert(idx_result.end(), idx_b.begin(), idx_b.end());

            result.set(idx_result, get(idx_a) * other.get(idx_b));
        } while (next_index(idx_b, shape_b));
    } while (next_index(idx_a, shape_a));

    return result;
}

// so dot product aka matrix multiplication AKA tensor contraction over last axis of first tensor and first axis of second tensor
Tensor Tensor::dot_product(const Tensor& other) const
{
    // sanity shape for matrix
    const std::vector<int>& shape_a = get_shape();
    const std::vector<int>& shape_b = other.get_shape();
    if (shape_a.size() != 2 || shape_b.size() != 2 || shape_a[1] != shape_b[0])
    {
        throw std::invalid_argument("Incompatible tensor shapes for dot product");
    }

    std::vector<int> result_shape = {shape_a[0], shape_b[1]};

    Tensor result(result_shape, requires_grad || other.requires_grad);

    for (int i = 0; i < shape_a[0]; ++i)
    {
        for (int j = 0; j < shape_b[1]; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < shape_a[1]; ++k)
            {
                sum += get({i, k}) * other.get({k, j});
            }
            result.set({i, j}, sum);
        }
    }
    return result;
}

Tensor Tensor::transpose(const std::vector<int>& axes) const
{
    throw std::runtime_error("Tensor::transpose not implemented yet.");
}

Tensor Tensor::contraction(const std::vector<std::pair<int, int>>& axes) const
{
    throw std::runtime_error("Tensor::contract not implemented yet.");
}

//
} // namespace eth
