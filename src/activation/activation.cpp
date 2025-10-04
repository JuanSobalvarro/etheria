#include "activation/activation.hpp"
#include "types.hpp"

#include <cmath>
#include <stdexcept>

namespace eth::act
{

float linear(float x)      
{ 
    return x; 
}
float der_linear(float)    
{ 
    return 1.0; 
}

float sigmoid(float x)     
{ 
    return 1.0f / (1.0f + std::exp(-x)); 
}

float der_sigmoid(float x) 
{ 
    float s = sigmoid(x); 
    return s * (1.0f - s); 
}

float relu(float x)        
{ 
    return x > 0.0f ? x : 0.0f; 
}

float der_relu(float x) 
{ 
    return x > 0.0f ? 1.0f : 0.0f; 
}

float tanh_act(float x)    
{ 
    return std::tanh(x); 
}

float der_tanh(float x)    
{ 
    float t = tanh_act(x); 
    return 1.0f - t * t; 

}

float softplus(float x)    
{ 
    return std::log(1.0f + std::exp(x)); 
}

float der_softplus(float x) 
{ 
    return sigmoid(x); 
}

Vector softmax(const Vector& x) 
{
    Vector s(x.size());
    float max_elem = *std::max_element(x.begin(), x.end());
    float sum_exp = 0.0f;

    for (size_t i = 0; i < x.size(); ++i) 
    {
        s[i] = std::exp(x[i] - max_elem); // for numerical stability
        sum_exp += s[i];
    }
    for (size_t i = 0; i < x.size(); ++i) 
    {
        s[i] /= sum_exp;
    }
    return s;
}

Vector der_softmax(const Vector& x, int index) 
{
    Vector s = softmax(x);
    Vector ds(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i)
    {
        if (i == index) 
        {
            ds[i] = s[i] * (1.0f - s[i]);
        } 
        else 
        {
            ds[i] = -s[i] * s[index];
        }
    }
    return ds;
}

float forward_activation(ActivationFunctionType act_type, float x) 
{
    switch (act_type) 
    {
        case LINEAR:   return linear(x);
        case SIGMOID:  return sigmoid(x);
        case RELU:     return relu(x);
        case TANH:     return tanh_act(x);
        case SOFTPLUS: return softplus(x);
        default: throw std::invalid_argument("for_act: Unknown activation function type");
    }
}

float forward_activation_derivative(ActivationFunctionType act_type, float x) 
{
    switch (act_type) 
    {
        case LINEAR:   return der_linear(x);
        case SIGMOID:  return der_sigmoid(x);
        case RELU:     return der_relu(x);
        case TANH:     return der_tanh(x);
        case SOFTPLUS: return der_softplus(x);
        default: throw std::invalid_argument("forward_act_der: Unknown activation function type");
    }
}

Vector forward_activation(ActivationFunctionType act_type, const Vector& x)
{
    Vector result(x.size());

    if (act_type == SOFTMAX) 
    {
        return softmax(x);
    }

    for (size_t i = 0; i < x.size(); ++i) 
    {
        result[i] = forward_activation(act_type, x[i]);
    }
    return result;
}

Vector forward_activation_derivative(ActivationFunctionType act_type, const Vector& x)
{
    Vector result(x.size());

    if (act_type == SOFTMAX) 
    {
        throw std::invalid_argument("forward_activation_derivative: Derivative of softmax for vector input requires index. Use der_softmax instead.");
    }

    for (size_t i = 0; i < x.size(); ++i) 
    {
        result[i] = forward_activation_derivative(act_type, x[i]);
    }
    return result;
}

void inplace_forward_activation(ActivationFunctionType act_type, Vector& x) 
{
    for (Scalar& v : x) 
    {
        v = forward_activation(act_type, v);
    }
}

void inplace_forward_activation_derivative(ActivationFunctionType act_type, Vector& x) 
{
    for (Scalar& v : x) 
    {
        v = forward_activation_derivative(act_type, v);
    }
}


} // namespace eth::act