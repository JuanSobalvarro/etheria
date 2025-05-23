#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <string>
#include <stdexcept>
#include "connection.hpp"
#include "neuron.hpp"
#include "activation_functions.hpp"

class Layer {
public:
    Layer(int numNeurons, std::vector<Connection*>& inputs, const ActivationFunctionType activation_function_type);
    ~Layer();

    void setWeights(const std::vector<std::vector<double>>& weights);
    void setBiases(const std::vector<double>& biases);

    std::vector<double> activate();

private:
    std::vector<Neuron*> neurons;
    std::vector<Connection*> inputs;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    ActivationFunctionType activation_function_type;

    void setInputs(std::vector<Connection*>& inputs);
};

#endif // LAYER_HPP