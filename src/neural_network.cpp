#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int>& hiddenLayerSizes, int outputSize, const std::string& activationFunctionType) 
{
}

NeuralNetwork::~NeuralNetwork() 
{
    // Destruct each layer
    inputLayer.~Layer();
    for (size_t i = 0; i < hiddenLayers.size(); i++) 
    {
        hiddenLayers[i].~Layer();
    }
    outputLayer.~Layer();
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) 
{
    std::vector<std::vector<double>> firstInputs = {inputs};
    // Set input layer values
    inputLayer.setInputs(firstInputs);

    // Activate hidden layers
    
    for (size_t i = 0; i < hiddenLayers.size(); i++) 
    {
        hiddenLayers[i].activate();
    }

    // Activate output layer
    outputLayer.activate();

    // Return output layer values
    return outputLayer.getOutputs();
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& labels, int epochs, double learningRate) 
{
    // Training logic goes here
    // This is a placeholder for the training process
    for (int epoch = 0; epoch < epochs; epoch++) 
    {
        for (size_t i = 0; i < trainingData.size(); i++) 
        {
            std::vector<double> inputs = trainingData[i];
            std::vector<double> expectedOutput = labels[i];

            // Forward pass
            std::vector<double> predictedOutput = predict(inputs);

            // Learning logic here
        }
    }
}

void NeuralNetwork::saveModel(const std::string& filename) 
{
}

void NeuralNetwork::loadModel(const std::string& filename) 
{
}
