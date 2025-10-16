#include "tensor/tensor.hpp"

namespace eth
{
Tensor::Tensor() 
: shape({}), data(nullptr), device_id(-1), num_elements(0), owns_data(true) 
{}

Tensor::Tensor(std::vector<int> shape) : shape(shape), data(nullptr), device_id(-1), owns_data(true)
{
    num_elements = 1;
    // since the size of the data can be define by its dimensions (eg. 3(first dim) x 4 (second dim) = 12 elements)
    for (int dim : shape) 
    {
        num_elements *= dim;
    }

    allocate_memory();
}

Tensor::Tensor(std::vector<int> shape, float* data) 
: shape(shape), data(data), device_id(-1), owns_data(false)
{
    num_elements = 1;
    for (int dim : shape) 
    {
        num_elements *= dim;
    }
    // we assume that the user provides valid data with the correct number of elements
    // and that the data is already allocated
    // we do not allocate memory in this constructor
}

Tensor::Tensor(const Tensor& other) 
: shape(other.shape), device_id(other.device_id), num_elements(other.num_elements), owns_data
(other.owns_data)
{
    if (other.data && owns_data) 
    {
        allocate_memory();
        std::copy(other.data, other.data + num_elements, data);
    } 
    else 
    {
        data = other.data; // shallow copy if we don't own the data
    }
}

Tensor::Tensor(Tensor&& other) noexcept
: shape(std::move(other.shape)), data(other.data), device_id(other.device_id), num_elements(other.num_elements), owns_data(other.owns_data)
{
    other.data = nullptr; // leave other in a valid state
    other.num_elements = 0;
    other.device_id = -1;
    other.owns_data = false;
}

Tensor::~Tensor() 
{
    free_memory();
}

float Tensor::get(const std::vector<int>& indices) const 
{
    int flat_index = calculate_index(indices);
    return get_value_at_flat_index(flat_index);
}

void Tensor::set(const std::vector<int>& indices, float value) 
{
    int flat_index = calculate_index(indices);
    set_value_at_flat_index(flat_index, value);
}

const std::vector<int>& Tensor::get_shape() const 
{
    return shape;
}

int Tensor::get_num_elements() const 
{
    return num_elements;
}

int Tensor::get_current_device_id() const 
{
    return device_id;
}

void Tensor::to_cpu() 
{
    if (device_id == -1) 
        return; // already on CPU

    if (!owns_data)
        throw std::runtime_error("Cannot move to CPU: Tensor does not own its data.");

    float* new_data = new float[num_elements];
    cuda::copy_to_host(new_data, data, num_elements * sizeof(float));
    // free old data
    free_memory();
    data = new_data;
    device_id = -1;
}

void Tensor::to_gpu(int device_id_) 
{
    if (device_id == device_id_) 
        return; // already on the desired GPU

    // some overhead checks
    if (!cuda::isCUDAAvailable())
        throw std::runtime_error("CUDA is not available.");

    if (!cuda::isCUDACompatible(device_id_))
        throw std::runtime_error("The specified GPU device is not compatible.");

    if (!owns_data)
        throw std::runtime_error("Cannot move to GPU: Tensor does not own its data.");

    if (device_id_ < 0)
        throw std::runtime_error("Invalid device ID. Must be >= 0 for GPU.");

    float* new_data = nullptr;
    cuda::allocate_device_memory((void**)&new_data, num_elements * sizeof(float));
    
    // first lets not worry about going from one gpu to another gpu
    // we will just copy from cpu to gpu
    if (device_id != -1)
        throw std::runtime_error("Direct GPU to GPU transfer not implemented. Move to CPU first.");

    cuda::copy_to_device(new_data, data, num_elements * sizeof(float));
    // free old data
    free_memory();

    data = new_data;
    device_id = device_id_;
}

// Inplace addition
void Tensor::add(const Tensor& other) 
{
    if (shape != other.shape)
        throw std::runtime_error("Shape mismatch for addition.");

    if (device_id != other.device_id)
        throw std::runtime_error("Device mismatch for addition.");

    if (device_id == -1) 
    {
        // CPU addition
        for (int i = 0; i < num_elements; ++i) 
        {
            data[i] += other.data[i];
        }
    } 
    else 
    {
        // GPU addition
        // For simplicity, we will not implement GPU kernel here
        throw std::runtime_error("GPU addition not implemented.");
    }
}

// Inplace multiplication
void Tensor::multiply(const Tensor& other) 
{
    if (shape != other.shape)
        throw std::runtime_error("Shape mismatch for multiplication.");

    if (device_id != other.device_id)
        throw std::runtime_error("Device mismatch for multiplication.");

    if (device_id == -1) 
    {
        // CPU multiplication
        for (int i = 0; i < num_elements; ++i) 
        {
            data[i] *= other.data[i];
        }
    } 
    else 
    {
        // GPU multiplication
        // For simplicity, we will not implement GPU kernel here
        throw std::runtime_error("GPU multiplication not implemented.");
    }
}

// Private methods

void Tensor::allocate_memory() 
{
    if (owns_data && !data) 
    {
        if (device_id == -1) 
        {
            // Allocate on CPU
            data = new float[num_elements];
        } 
        else 
        {
            // Allocate on GPU
            cuda::allocate_device_memory((void**)&data, num_elements * sizeof(float));
        }
    }
}

void Tensor::free_memory() 
{
    if (owns_data && data) 
    {
        if (device_id == -1) 
        {
            // Free CPU memory
            delete[] data;
        } 
        else 
        {
            // Free GPU memory
            cuda::free_device_memory(data);
        }
        data = nullptr;
    }
}

int Tensor::calculate_index(const std::vector<int>& indices) const
{
    if (indices.size() != shape.size())
        throw std::runtime_error("Index dimensionality does not match the tensor shape.");

    int flat_index = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) 
    {
        if (indices[i] < 0 || indices[i] >= shape[i])
            throw std::runtime_error("Index out of bounds.");
        
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }
    return flat_index;
}

void Tensor::set_value_at_flat_index(int index, float value) 
{
    if (index < 0 || index >= num_elements)
        throw std::runtime_error("Flat index out of bounds.");

    if (device_id == -1) 
    {
        data[index] = value;
    } 
    else 
    {
        // For simplicity, we will copy the value to host, set it, and copy back
        float temp;
        cuda::copy_to_host(&temp, data + index, sizeof(float));
        temp = value;
        cuda::copy_to_device(data + index, &temp, sizeof(float));
    }
}

float Tensor::get_value_at_flat_index(int index) const 
{
    if (index < 0 || index >= num_elements)
        throw std::runtime_error("Flat index out of bounds.");

    if (device_id == -1) 
    {
        return data[index];
    } 
    else 
    {
        float temp;
        cuda::copy_to_host(&temp, data + index, sizeof(float));
        return temp;
    }
}

//
} // namespace eth
