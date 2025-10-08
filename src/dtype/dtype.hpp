#pragma once

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

template<typename T>
DType fromType();

template<>
DType fromType<float>()
{
    return DType::Float32;
}

} // namespace eth