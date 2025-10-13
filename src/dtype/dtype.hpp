#pragma once

#include <type_traits>
#include <cstdint>

namespace eth
{

enum class DType
{
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
    Bool
};

inline size_t itemsize(DType dtype)
{
    switch (dtype)
    {
        case DType::Float32:
            return sizeof(float);
        case DType::Float64:
            return sizeof(double);
        case DType::Int32:
            return sizeof(int32_t);
        case DType::Int64:
            return sizeof(int64_t);
        case DType::UInt8:
            return sizeof(uint8_t);
        case DType::Bool:
            return sizeof(bool);
        default:
            throw std::invalid_argument("Unsupported DType");
    }
}

template<typename T>
DType fromType()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return DType::Float32;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return DType::Float64;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return DType::Int32;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return DType::Int64;
    }
    else if constexpr (std::is_same_v<T, uint8_t>)
    {
        return DType::UInt8;
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        return DType::Bool;
    }
    else
    {
        static_assert(!sizeof(T*), "Unsupported type for DType");
    }
}

} // namespace eth