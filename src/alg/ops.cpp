#include "alg/ops.hpp"
#include <stdexcept> // for runtime_error

namespace eth::alg 
{

// --- Matrix-vector product: y = W * v ---
Vector matrixvectorproduct(const Matrix& W, const Vector& v) 
{
    if (W.empty() || W[0].size() != v.size())
        throw std::runtime_error("matrixvectorproduct: dimension mismatch");

    Vector result(W.size(), 0.0);
    for (size_t i = 0; i < W.size(); i++) {
        for (size_t j = 0; j < W[i].size(); j++) {
            result[i] += W[i][j] * v[j];
        }
    }
    return result;
}

// --- Outer product: M = a * b^T ---
Matrix outerproduct(const Vector& a, const Vector& b) 
{
    if (a.empty() || b.empty())
        throw std::runtime_error("outerproduct: dimension mismatch");

    size_t rows = a.size();
    size_t cols = b.size();

    Matrix result(rows, Vector(cols, 0.0));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

// --- Vector addition: c = a + b ---
Vector vectoradd(const Vector& a, const Vector& b) 
{
    if (a.size() != b.size())
        throw std::runtime_error("vectoradd: dimension mismatch");

    Vector result(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}


// --- AXPY: y += alpha * x ---
void inplace_axpy(Vector& y, const Vector& x, Scalar alpha) 
{
    if (y.size() != x.size())
        throw std::runtime_error("inplace_axpy: dimension mismatch. Dim1: " + std::to_string(y.size()) + " Dim2: " + std::to_string(x.size()));

    for (size_t i = 0; i < y.size(); i++) {
        y[i] += alpha * x[i];
    }
}

Vector applyFunction(const Vector& v, act::ActivationFunctionType act_type, bool derivative) 
{
    if (derivative) 
    {
        return act::forward_activation_derivative(act_type, v);
    }
    return act::forward_activation(act_type, v);
}

void inplace_applyFunction(Vector& v, act::ActivationFunctionType act_type, bool derivative) 
{
    if (derivative) 
    {
        for (auto& val : v) {
            val = act::forward_activation_derivative(act_type, val);
        }
    } 
    else 
    {
        for (auto& val : v) {
            val = act::forward_activation(act_type, val);
        }
    }
}
} // namespace eth::alg