#include "itensor.hpp"

namespace eth
{

// TODO: implement creation directly on GPU i am lazy right now
ITensor::ITensor(std::vector<int> shape, int device_id)
    : shape(std::move(shape)), device_id(device_id)
{
    // this should be handle automatically below but just in case we manually create a rank 0 tensor
    if (this->shape.empty())
    {
        // rank 0 tensor (scalar)
        data = new float[1];
        num_elements = 1;
        rank = 0;
        return;
    }


    // allocate data based on shape
    int num_elements = 1;
    for (int dim : this->shape)
    {
        num_elements *= dim;
    }
    data = new float[num_elements];
    this->num_elements = num_elements;
    this->rank = shape.size();
}

ITensor::ITensor(float value, int device_id)
    : shape({}), device_id(device_id)
{
    data = new float[1];
    data[0] = value;
    num_elements = 1;
    rank = 0;
}

ITensor::ITensor(std::vector<float> data, int device_id)
    : shape({static_cast<int>(data.size())}), device_id(device_id)
{
    this->data = new float[data.size()];
    std::copy(data.begin(), data.end(), this->data);
    num_elements = static_cast<int>(data.size());
    rank = 1;
}

ITensor::ITensor(std::vector<float> data, std::vector<int> shape, int device_id)
    : data(nullptr), shape(std::move(shape)), device_id(device_id)
{
    // sanity check if data is compatible with shape
    int expected_size = 1;
    for (int dim : this->shape)
    {
        expected_size *= dim;
    }
    if (expected_size != static_cast<int>(data.size()))
    {
        throw std::invalid_argument("Data size does not match shape dimensions.");
    }
    this->data = new float[expected_size];
    std::copy(data.begin(), data.end(), this->data);
    num_elements = expected_size;
    rank = this->shape.size();
}

ITensor::~ITensor()
{
    free_memory();
    delete[] data;
}

// copy operations
ITensor::ITensor(const ITensor& other)
    : shape(other.shape), num_elements(other.num_elements), device_id(other.device_id)
{
    data = new float[num_elements];
    std::copy(other.data, other.data + num_elements, data);
}

ITensor& ITensor::operator=(const ITensor& other)
{
    if (this != &other)
    {
        delete[] data;

        shape = other.shape;
        num_elements = other.num_elements;
        device_id = other.device_id;
        rank = other.rank;

        data = new float[num_elements];
        std::copy(other.data, other.data + num_elements, data);
    }
    return *this;
}

ITensor::ITensor(ITensor&& other) noexcept
    : data(other.data), shape(std::move(other.shape)), num_elements(other.num_elements), device_id(other.device_id)
{
    other.data = nullptr;
    other.num_elements = 0;
    other.rank = 0;
}

ITensor& ITensor::operator=(ITensor&& other) noexcept
{
    if (this != &other)
    {
        delete[] data;

        data = other.data;
        shape = std::move(other.shape);
        num_elements = other.num_elements;
        device_id = other.device_id;

        other.data = nullptr;
        other.num_elements = 0;
        other.rank = 0;
    }
    return *this;
}

float ITensor::get(const std::vector<int>& indices) const
{
    // calculate flat index
    int flat_index = calculate_index(indices);
    return data[flat_index];
}

void ITensor::set(const std::vector<int>& indices, float value)
{
    // calculate flat index
    int flat_index = calculate_index(indices);
    data[flat_index] = value;
}

float ITensor::get(int index) const
{
    return data[index];
}

void ITensor::set(int index, float value)
{
    data[index] = value;
}

void ITensor::move_to_gpu(int device_id_)
{
    if (device_id == device_id_) 
        return; // already on the desired GPU

    // some overhead checks
    if (!cuda::isCUDAAvailable())
        throw std::runtime_error("CUDA is not available.");

    if (!cuda::isCUDACompatible(device_id_))
        throw std::runtime_error("The specified GPU device is not compatible.");

    if (device_id_ < 0)
        throw std::runtime_error("Invalid device ID. Must be >= 0 for GPU.");

    cuda::setDevice(device_id_);

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

void ITensor::move_to_cpu()
{
    if (device_id == -1) 
        return; // already on CPU

    float* new_data = new float[num_elements];
    cuda::copy_to_host(new_data, data, num_elements * sizeof(float));
    // free old data
    free_memory();
    data = new_data;
    device_id = -1;
}

int ITensor::current_device() const
{
    return device_id;
}

void ITensor::allocate_memory()
{
    // ensure any existing memory is freed first
    free_memory();

    if (!data)
    {
        int num_elements = 1;
        for (int dim : shape)
        {
            num_elements *= dim;
        }

        rank = shape.size();

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

void ITensor::free_memory()
{
    if (data)
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

int ITensor::calculate_index(const std::vector<int>& indices) const
{
    int flat_index = 0;
    int stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }
    return flat_index;
}

// getters

const std::vector<int>& ITensor::get_shape() const
{
    return shape;
}

const int ITensor::get_rank() const
{
    return rank;
}

int ITensor::size() const
{
    return num_elements;
}

float* ITensor::get_data() const
{
    return data;
}

std::vector<float> ITensor::to_vector() const
{
    std::vector<float> vec(num_elements);
    if (device_id == -1)
    {
        // Data is on CPU
        std::copy(data, data + num_elements, vec.begin());
    }
    else
    {
        // Data is on GPU, need to copy to CPU first
        float* temp = new float[num_elements];
        cuda::copy_to_host(temp, data, num_elements * sizeof(float));
        std::copy(temp, temp + num_elements, vec.begin());
        delete[] temp;
    }
    return vec;
}

//
} // namespace eth
