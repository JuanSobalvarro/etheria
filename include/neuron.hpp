#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

#include "activation_functions.hpp"
#include "connection.hpp"


class Neuron {
public:
    Neuron(const ActivationFunctionType activation_function_type);
    ~Neuron();

    void setWeights(const std::vector<double>& weights);
    void setBias(double bias);
    void setInputs(const std::vector<Connection*>& inputs);

    double getBias() const;
    std::vector<double> getWeights() const;

    ActivationFunction* getActivationFunction() const;

    double activate();

    Connection* getOutput() const;

private:
    std::vector<double> weights;
    std::vector<Connection*> inputs;
    Connection* output;
    double bias;

    ActivationFunctionType activation_function_type;
    ActivationFunction* act_func;
};

#endif // NEURON_HPP