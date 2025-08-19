#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include "activation_functions.hpp"
#include "neuron.hpp"
#include "layer.hpp"


class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, const std::vector<int> hiddenLayerSizes, int outputSize, const ActivationFunctionType activationFunctionType);
    ~NeuralNetwork();
    
    std::vector<double> predict(const std::vector<double>& inputs);
    void train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& labels, int epochs, double learningRate);

    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

    void printNeuralNetwork() const;
private:
    std::vector<Connection*> inputConnections;
    Layer* inputLayer;
    std::vector<Layer*> hiddenLayers;
    Layer* outputLayer;
};

#endif // NEURAL_NETWORK_H