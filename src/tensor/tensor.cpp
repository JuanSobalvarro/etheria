#include "tensor.hpp"

namespace eth
{

Tensor::Tensor(std::vector<int> shape, bool requires_grad, int device_id)
    : ITensor(shape, device_id), requires_grad(requires_grad)
{
}
Tensor::Tensor(float value, bool requires_grad, int device_id)
    : ITensor(value, device_id), requires_grad(requires_grad)
{
}
Tensor::Tensor(std::vector<float> data, bool requires_grad, int device_id)
    : ITensor(data, device_id), requires_grad(requires_grad)
{
}
Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad, int device_id)
    : ITensor(data, shape, device_id), requires_grad(requires_grad)
{
}

// copy operations
Tensor::Tensor(const Tensor& other) 
    : ITensor(other), requires_grad(other.requires_grad)
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

Tensor::~Tensor()
{
    this->ITensor::~ITensor();
}

Tensor Tensor::add(const Tensor& other) const
{
    if (other.get_rank() == 1)
    {
        return add_scalar(other.get(0));
    }

    if (other.get_rank() != this->get_rank())
    {
        throw std::invalid_argument("Incompatible tensor ranks for addition. Got ranks: " +
            std::to_string(this->get_rank()) + " and " + std::to_string(other.get_rank()) + ")");
    }

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

Tensor Tensor::add_scalar(const float scalar) const
{
    Tensor result(get_shape(), requires_grad);
    for (size_t i = 0; i < size(); ++i)
    {
        float sum = get(i) + scalar;
        result.set(i, sum);
    }
    return result;
}

Tensor Tensor::subtract(const Tensor& other) const
{
    // For a subtract operation the both tensors must have the same shape
    if (get_shape() != other.get_shape())
    {
        throw std::invalid_argument("Incompatible tensor shapes");
    }

    Tensor result(get_shape(), requires_grad || other.requires_grad);
    for (size_t i = 0; i < size(); ++i)
    {
        float diff = get(i) - other.get(i);
        result.set(i, diff);
    }
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const
{
    // For an element-wise multiplication operation the both tensors must have the same shape?
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

Tensor Tensor::scalar_multiply(float scalar) const
{
    Tensor result(get_shape(), requires_grad);
    for (size_t i = 0; i < size(); ++i)
    {
        float prod = get(i) * scalar;
        result.set(i, prod);
    }
    return result;
}

// aka tensor product define as tensor v dot product tensor w transposed
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
    const int rank_a = get_rank();
    const int rank_b = other.get_rank();
    const std::vector<int>& shape_a = get_shape();
    const std::vector<int>& shape_b = other.get_shape();
    if (rank_a != 2 || rank_b != 2)
    {
        throw std::invalid_argument("Dot product currently only supports rank 2 tensors (matrices). Got ranks: " +
            std::to_string(rank_a) + " and " + std::to_string(rank_b) + ")");
    }
    
    if (shape_a[1] != shape_b[0])
    {
        throw std::invalid_argument("Incompatible tensor shapes for dot product. Got shapes: (" +
            std::to_string(shape_a[0]) + ", " + std::to_string(shape_a[1]) + ") and (" +
            std::to_string(shape_b[0]) + ", " + std::to_string(shape_b[1]) + ")");
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
                float a_val = get({i,k});
                float b_val = other.get({k,j});
                sum += a_val * b_val;
            }
            result.set({i,j}, sum);
        }
    }

    return result;
}

Tensor Tensor::transpose(const std::vector<int>& axes) const
{
    // so to make a transpose of a tensor we need to permute its dimensions according to axes given
    const std::vector<int>& original_shape = get_shape();
    if (axes.size() != original_shape.size())
    {
        throw std::invalid_argument("Axes size must match tensor rank for transpose");
    }
    // new shape after transpose
    std::vector<int> new_shape(axes.size());
    for (size_t i = 0; i < axes.size(); ++i)
    {
        // the axe at i says which original dimension "swaps" to this position
        new_shape[i] = original_shape[axes[i]];
    }
    std::vector<float> prev_data = to_vector();
    Tensor result(prev_data, new_shape, requires_grad);
    // now we need to set the values in the transposed positions
    std::vector<int> idx_new(axes.size(), 0);
    do {
        // calculate the corresponding original index
        std::vector<int> idx_original(axes.size());
        for (size_t i = 0; i < axes.size(); ++i)
        {
            // essentially we are swapping back? the shape we did before
            idx_original[axes[i]] = idx_new[i];
        }
        result.set(idx_new, get(idx_original));
    } while (next_index(idx_new, new_shape));
    return result;
}

Tensor Tensor::contraction(const std::vector<std::pair<int, int>>& axes) const
{
    throw std::runtime_error("Tensor::contract not implemented yet.");
}

// get subtensor single index
Tensor Tensor::get_subtensor(size_t index) const
{
    const std::vector<int>& shape = get_shape();
    if (shape.size() < 1)
    {
        throw std::invalid_argument("Tensor must have at least rank 1 to get subtensor");
    }
    if (index >= static_cast<size_t>(shape[0]))
    {
        throw std::out_of_range("Index out of range for subtensor");
    }

    // New shape is original shape without the first dimension
    std::vector<int> new_shape(shape.begin() + 1, shape.end());
    Tensor result(new_shape, requires_grad);

    if (shape.size() == 1)
    {
        // Special case: rank 1 tensor, return a scalar
        result = Tensor(get(static_cast<int>(index)), requires_grad);
        return result;
    }

    // Calculate the size of the subtensor, starting from second dimension
    int subtensor_size = 1;
    for (size_t i = 1; i < shape.size(); ++i)
    {
        subtensor_size *= shape[i];
    }

    // Copy data for the subtensor
    int offset = index * subtensor_size;
    for (int i = 0; i < subtensor_size; ++i)
    {
        result.set(i, get(offset + i));
    }

    return result;
}

// get subtensor with multi index it means it will reduce rank by each index provided, practically the same think as above but more general
Tensor Tensor::get_subtensor(const std::vector<size_t> indices) const
{
    const std::vector<int>& shape = get_shape();
    if (indices.size() > shape.size())
    {
        throw std::invalid_argument("Indices size must be less than tensor rank to get subtensor. Indices: " + std::to_string(indices.size()) + " Shape rank: " + std::to_string(shape.size()) + ")");
    }
    // Validate indices
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] < 0 || indices[i] >= shape[i])
        {
            throw std::out_of_range("Index out of range for subtensor");
        }
    }

    // New shape is original shape without the first 'indices.size()' dimensions
    std::vector<int> new_shape(shape.begin() + indices.size(), shape.end());
    Tensor result(new_shape, requires_grad);

    // Calculate the size of the subtensor
    int subtensor_size = 1;
    for (size_t i = indices.size(); i < shape.size(); ++i)
    {
        subtensor_size *= shape[i];
    }

    // Calculate the offset in the original tensor
    int offset = 0;
    int stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= static_cast<int>(indices.size()); --i)
    {
        offset += stride * 0; // we will add the fixed indices later
        stride *= shape[i];
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        offset += indices[i] * stride;
        stride *= shape[i];
    }

    // Copy data for the subtensor
    for (int i = 0; i < subtensor_size; ++i)
    {
        result.set(i, get(offset + i));
    }

    return result;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const
{
    // Calculate total number of elements in current tensor
    int current_size = size();

    // Calculate total number of elements in new shape
    int new_size = 1;
    for (int dim : new_shape)
    {
        new_size *= dim;
    }

    if (current_size != new_size)
    {
        throw std::invalid_argument("Total number of elements must remain the same for reshape. Current size: " +
            std::to_string(current_size) + ", New size: " + std::to_string(new_size));
    }

    // Create new tensor with the new shape and copy data
    std::vector<float> current_data = to_vector();
    Tensor result(current_data, new_shape, requires_grad);

    return result;
}

//
} // namespace eth
