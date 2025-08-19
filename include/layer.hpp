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

    
    void setInputs(std::vector<Connection*>& inputs);
    
    std::vector<Connection*> getInputs() const;
    std::vector<double> getInputsValue() const;
    std::vector<Connection*> getOutputs() const;
    std::vector<double> getOutputsValue() const;
    std::vector<Neuron*> getNeurons() const;
    
    std::vector<double> activate();

private:
    std::vector<Neuron*> neurons;
    std::vector<Connection*> inputs;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    ActivationFunctionType activation_function_type;

};

#endif // LAYER_HPP