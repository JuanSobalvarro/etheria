#include "neuron.hpp"
#include <iostream>
#include <random>


Neuron::Neuron(const ActivationFunctionType activation_function_type)
{
    this->output = new Connection();
    // this->bias = static_cast<double>(rand()) / RAND_MAX;
    this->bias = 0;
    
    this->activation_function_type = activation_function_type;
    this->act_func = ActivationFactory::createActivationFunction(activation_function_type);

}

Neuron::~Neuron()
{
    // std::cout << "Neuron destroyed\n";
    delete output;
    delete act_func;
}

void Neuron::setWeights(const std::vector<double>& weights)
{
    if (weights.empty()) {
        throw std::invalid_argument("Weights cannot be empty");
    }
    // if (weights.size() != this->weights.size()) {
    //     throw std::invalid_argument("Weights size must match the current weights size. Current: " + std::to_string(this->weights.size()) + ", New: " + std::to_string(weights.size()));
    // }
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

void Neuron::setDelta(double delta)
{
    this->delta = delta;
}

ActivationFunction* Neuron::getActivationFunction() const
{
    return act_func;
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

    zValue = sum; // Store the pre-activation value

    double result = act_func->activate(sum);
    
    this->output->changeValue(result);

    // std::cout << "Neuron activated. Output: " << result << "\n";

    return result;
}

Connection* Neuron::getOutput() const
{
    return output;
}

std::vector<Connection*> Neuron::getInputs() const
{
    return inputs;
}   

double Neuron::getDelta() const
{
    return delta;
}

std::vector<double> Neuron::updateWeightsTraining(double learningRate)
{
    for (size_t i = 0; i < weights.size(); i++)
    {
        double inputValue = inputs[i]->getValue();
        weights[i] -= learningRate * delta * inputValue;
    }
    return weights;
}


double Neuron::updateBiasTraining(double learningRate)
{
    bias -= learningRate * delta;
    return bias;
}
