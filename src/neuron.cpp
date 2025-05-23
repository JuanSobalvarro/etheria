#include "neuron.hpp"
#include <iostream>


Neuron::Neuron(const ActivationFunctionType activation_function_type)
{
    std::cout << "Neuron created\n";
    this->output = new Connection();
    this->bias = 0.0;

    this->activation_function_type = activation_function_type;
    this->act_func = ActivationFactory::createActivationFunction(activation_function_type);
}

Neuron::~Neuron()
{
    std::cout << "Neuron destroyed\n";
    delete output;
    delete act_func;
}

void Neuron::setWeights(const std::vector<double>& weights)
{
    this->weights = weights;
}

std::vector<double> Neuron::getWeights() const
{
    return weights;
}

double Neuron::getBias() const
{
    return bias;
}

void Neuron::setBias(double bias)
{
    this->bias = bias;
}

void Neuron::setInputs(const std::vector<Connection*>& inputs)
{
    this->inputs = inputs;
}

double Neuron::activate()
{
    if (inputs.empty())
    {
        throw std::invalid_argument("Inputs cannot be empty");
    }
    if (weights.empty())
    {
        throw std::invalid_argument("Weights cannot be empty");
    }

    if (inputs.size() != weights.size())
    {
        throw std::invalid_argument("Number of inputs and weights must be the same");
    }

    double sum = 0.0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        sum += inputs[i]->getValue() * weights[i];
    }
    sum += this->bias;

    double result = act_func->activate(sum);
    
    this->output->changeValue(result);
    
    return result;
}