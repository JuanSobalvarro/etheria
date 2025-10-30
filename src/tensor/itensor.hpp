/*
* IN this module we define the internal tensor class ITensor
* which handles low level tensor data management and access.
* It is just the memory management and data access layer.
* Higher level tensor operations are defined in Tensor class.
*/
#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

#include "cuda/cuda_helper.cuh"

namespace eth
{

class ITensor
{
public:
    ITensor(std::vector<int> shape, int device_id = -1);
    ITensor(float value, int device_id = -1); // rank 0 tensor (scalar)
    ITensor(std::vector<float> data, int device_id = -1); // infers shape as 1D tensor, tensor of rank 1 (vector)
    ITensor(std::vector<float> data, std::vector<int> shape, int device_id = -1); // tensor with rank > 1

    ~ITensor();

    // copy operations
    ITensor(const ITensor& other);
    ITensor& operator=(const ITensor& other);
    ITensor(ITensor&& other) noexcept;
    ITensor& operator=(ITensor&& other) noexcept;

    // multi index access
    float get(const std::vector<int>& indices) const;
    void set(const std::vector<int>& indices, float value);

    // flattened index access
    float get(int index) const;
    void set(int index, float value);

    void move_to_gpu(int device_id);
    void move_to_cpu();
    int current_device() const;

    // getters
    const std::vector<int>& get_shape() const;
    const int get_rank() const;
    int size() const;
    float* get_data() const;

    std::vector<float> to_vector() const;

private:
    float* data;

    // ok so about shape... we store the shape as a vector of ints which determine the dimensions(axes)
    // of the tensor. For example a shape of {3, 4, 5} means a tensor of rank 3 (3 dimensions)
    // where the first dimension has size 3, the second 4 and the third 5.
    // another example a shape of {2, 2} means a rank 2 tensor (matrix) of size 2x2
    // a shape of {10} means a rank 1 tensor (vector) of size 10
    // and a shape of {} means a rank 0 tensor (scalar)
    std::vector<int> shape;
    int num_elements; // total number of elements as 1D flattened array

    int device_id; // cpu = -1, gpu >= 0

    void allocate_memory();
    void free_memory();
    
    int calculate_index(const std::vector<int>& indices) const;
};

//
} // namespace eth
