#pragma once

#include <vector>
#include "cuda/cuda_helper.cuh"


namespace eth
{

// Tensor class for multidimensional arrays of float data
class Tensor
{
public:
    Tensor();
    Tensor(std::vector<int> shape);
    Tensor(std::vector<int> shape, float* data);
    
    // copy constructor
    Tensor(const Tensor& other);
    // move constructor
    Tensor(Tensor&& other) noexcept;

    ~Tensor();

    float get(const std::vector<int>& indices) const;
    void set(const std::vector<int>& indices, float value);

    const std::vector<int>& get_shape() const;
    int get_num_elements() const;

    int get_current_device_id() const;

    // device management
    void to_cpu();
    void to_gpu(int device_id = 0);

    // inplace operations
    void add(const Tensor& other);
    void multiply(const Tensor& other);

    // copy operations

private:
    std::vector<int> shape;
    float* data;
    int device_id; // cpu = -1, gpu >= 0
    int num_elements;
    // why owns data? because if we pass by pointer we do not "own" that data so the freeing does not correspond
    // to tensor object
    bool owns_data;

    void allocate_memory();
    void free_memory();

    // since the data is flattened we need to calculate the index given the vector access
    int calculate_index(const std::vector<int>& indices) const;

    void set_value_at_flat_index(int index, float value);
    float get_value_at_flat_index(int index) const;
};

} // namespace eth