#include "layer.hpp"
#include <iostream>


Layer::Layer(int numNeurons, std::vector<Connection*>& inputs, const ActivationFunctionType activationFunctionType) 
{
    std::cout << "Layer created\n";
    if (numNeurons <= 0) 
    {
        throw std::invalid_argument("Number of neurons must be greater than zero");
    }
    if (inputs.empty()) 
    {
        throw std::invalid_argument("Inputs cannot be empty");
    }
    if (activationFunctionType < LINEAR || activationFunctionType > SOFTPLUS) 
    {
        throw std::invalid_argument("Invalid activation function type");
    }

    this->activation_function_type = activationFunctionType;
    
    for (int i = 0; i < numNeurons; i++) {
        Neuron* n = new Neuron(activationFunctionType);
        neurons.push_back(n);
    }
    this->setInputs(inputs);
    this->setWeights(std::vector<std::vector<double>>(numNeurons, std::vector<double>(inputs.size(), 0.0)));
    this->setBiases(std::vector<double>(numNeurons, 0.0));
}

Layer::~Layer() 
{
    std::cout << "Layer destroyed\n";
    for (Neuron* neuron : neurons) {
        delete neuron;
    }
    neurons.clear();
    inputs.clear();
    weights.clear();
    biases.clear();
}

void Layer::setInputs(std::vector<Connection*>& inputs) 
{
    if (inputs.empty()) 
    {
        throw std::invalid_argument("Inputs cannot be empty");
    }

    if (neurons.empty()) 
    {
        throw std::invalid_argument("Layer must contain at least one neuron");
    }

    // Set the inputs for each neuron in the layer
    for (Neuron* neuron : neurons) {
        neuron->setInputs(inputs);
    }

    // Store the inputs for later use
    this->inputs = inputs;
}

void Layer::setWeights(const std::vector<std::vector<double>>& weights) 
{
    // Set the weights for each neuron in the layer
    for (size_t i = 0; i < neurons.size(); ++i) {
        neurons[i]->setWeights(weights[i]);
    }
}

void Layer::setBiases(const std::vector<double>& biases) 
{
    // Set the biases for each neuron in the layer
    for (size_t i = 0; i < this->neurons.size(); ++i) {
        this->neurons[i]->setBias(biases[i]);
    }
}

std::vector<double> Layer::activate() 
{
    std::vector<double> outputs;
    for (Neuron* neuron : neurons) {
        outputs.push_back(neuron->activate());
    }
    return outputs;
}
