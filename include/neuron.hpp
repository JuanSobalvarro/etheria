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

    void setDelta(double delta);
    void setWeights(const std::vector<double>& weights);
    void setBias(double bias);
    void setInputs(const std::vector<Connection*>& inputs);

    double getDelta() const;
    double getBias() const;
    std::vector<double> getWeights() const;
    double getZValue() const { return zValue; }

    std::vector<double> updateWeightsTraining(double learningRate);
    double updateBiasTraining(double learningRate);

    ActivationFunction* getActivationFunction() const;

    double activate();

    Connection* getOutput() const;
    std::vector<Connection*> getInputs() const;

private:
    std::vector<double> weights;
    std::vector<Connection*> inputs;
    Connection* output;
    double bias;
    double delta = 0.0;
    double zValue = 0.0;

    ActivationFunctionType activation_function_type;
    ActivationFunction* act_func;
};

#endif // NEURON_HPP