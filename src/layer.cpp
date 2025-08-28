#include "layer.hpp"
#include <iostream>
#include <random>


Layer::Layer(int numNeurons, std::vector<Connection*>& inputs, const ActivationFunctionType activationFunctionType) 
{
    if (numNeurons <= 0) 
        throw std::invalid_argument("Number of neurons must be greater than zero");
    if (inputs.empty()) 
        throw std::invalid_argument("Inputs cannot be empty");
    if (activationFunctionType < LINEAR || activationFunctionType > SOFTPLUS) 
        throw std::invalid_argument("Invalid activation function type");

    this->activation_function_type = activationFunctionType;
    
    for (int i = 0; i < numNeurons; i++) {
        Neuron* n = new Neuron(activationFunctionType);
        neurons.push_back(n);
    }

    this->setInputs(inputs);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t numInputs = inputs.size();
    double he_stddev = std::sqrt(2.0 / numInputs);  // He initialization for ReLU
    std::normal_distribution<> d(0.0, he_stddev);

    std::vector<std::vector<double>> initial_weights;
    std::vector<double> initial_biases;

    for (int i = 0; i < numNeurons; i++) {
        std::vector<double> neuron_weights;
        for (size_t j = 0; j < numInputs; j++) {
            neuron_weights.push_back(d(gen));  // Randomized weight
        }
        initial_weights.push_back(neuron_weights);

        initial_biases.push_back(0.0);  // Optional: small positive value for ReLU, e.g., 0.01
    }

    this->setWeights(initial_weights);
    this->setBiases(initial_biases);
}

Layer::~Layer() 
{
    // std::cout << "Layer destroyed\n";
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
    if (neurons.empty()) 
    {
        throw std::runtime_error("Layer has no neurons to activate");
    }
    std::vector<double> outputs;
    for (Neuron* neuron : neurons) {
        outputs.push_back(neuron->activate());
    }
    
    return outputs;
}

std::vector<Connection*> Layer::getInputs() const 
{
    return inputs;
}

std::vector<double> Layer::getInputsValue() const 
{
    std::vector<double> inputValues;
    for (Connection* conn : inputs) {
        inputValues.push_back(conn->getValue());
    }
    return inputValues;
}

std::vector<Connection*> Layer::getOutputs() const 
{
    std::vector<Connection*> outputs;
    for (const Neuron* neuron : neurons) {
        outputs.push_back(neuron->getOutput());
    }
    return outputs;
}

std::vector<double> Layer::getOutputsValue() const 
{
    std::vector<double> outputValues;
    for (const Neuron* neuron : neurons) {
        outputValues.push_back(neuron->getOutput()->getValue());
    }
    return outputValues;
}

std::vector<Neuron*> Layer::getNeurons() const
{
    return neurons;
}