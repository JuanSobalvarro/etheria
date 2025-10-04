#pragma once

#include "activation/activation.hpp"
#include "types.hpp"
#include <string>

namespace eth::alg
{
    // Element-wise vector addition: a + b
    Vector vectoradd(const Vector& a, const Vector& b);

    // Matrix-vector product: W * v
    // multiplication of an m x n matrix W by a n x 1 vector v resulting in a m x 1 vector
    Vector matrixvectorproduct(const Matrix& W, const Vector& v);

    // Outer product: a ⊗ b is define as the multiplication of a column vector a by a row vector b
    // where the matrix is defined as the result of multiplying each element of a by each element of b
    Matrix outerproduct(const Vector& a, const Vector& b);

    // In-place axpy operation: y += alpha * x
    // (in few words, it scales vector x by alpha and adds it to y) alpha * x + y = y
    void inplace_axpy(Vector& y, const Vector& x, Scalar alpha);

    Vector applyFunction(const Vector& v, act::ActivationFunctionType act_type, bool derivative = false);

} // namespace eth::alg
