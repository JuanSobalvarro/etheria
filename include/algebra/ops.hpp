#pragma once

#include "types.hpp"

namespace eth::algebra
{
    enum class Backend { CPU, CUDA };

    // Matrix-vector product: W * v
    Vector matrixvectorproduct(const Matrix& W, const Vector& v, Backend backend = Backend::CPU);
    
    // Outer product: a (rows) * b^T
    Matrix outerproduct(const Vector& a, const Vector& b, Backend backend = Backend::CPU);
    
    // Element-wise vector addition: a + b
    Vector vectoradd(const Vector& a, const Vector& b, Backend backend = Backend::CPU);
    
    // In-place axpy operation: y += alpha * x
    void inplace_axpy(Vector& y, const Vector& x, Scalar alpha, Backend backend = Backend::CPU);

} // namespace eth::algebra
