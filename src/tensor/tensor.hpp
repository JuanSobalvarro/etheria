#pragma once

#include <dtype/dtype.hpp>
#include <memory>
#include <vector>

namespace eth
{

typedef std::vector<size_t> TensorShape;

// This class represents a multi-dimensional array (tensor) of elements of type T.
template<typename T>
class Tensor
{
public:
    Tensor(const TensorShape& shape, DType dtype = DType::fromType<T>())
    {
        m_shape = shape;
        m_dtype = dtype;

        size_t totalSize = 1;
        for (size_t dim : shape)
        {
            totalSize *= dim;
        }

        m_data = std::make_unique<T[]>(totalSize);
    }
    ~Tensor() = default;

    // Accessors
    const TensorShape& shape() const
    {
        return m_shape;
    }

    DType dtype() const
    {
        return m_dtype;
    }

    T* data()
    {
        return m_data.get();
    }

    const T* data() const
    {
        return m_data.get();
    }

    // Index access (flattened)
    T& operator[](size_t index)
    {
        return m_data[index];
    }

    // multi-dimensional index access
    T& operator[](const TensorShape& indices)
    {
        size_t index = getIndex(indices);
        return m_data[index];
    }

private:
    std::unique_ptr<T[]> m_data;
    TensorShape m_shape;
    DType m_dtype;

    // we need a get index function to convert multi-dimensional indices to a single index (since tensor is flattened)
    size_t getIndex(const TensorShape& indices) const
    {
        if (indices.size() != m_shape.size())
        {
            throw std::out_of_range("Number of indices does not match tensor dimensions");
        }

        size_t index = 0;
        size_t stride = 1;
        for (int i = m_shape.size() - 1; i >= 0; --i)
        {
            if (indices[i] >= m_shape[i])
            {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * stride;
            stride *= m_shape[i];
        }
        return index;
    }
};

} // namespace eth