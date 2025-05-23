#include "lib/nn/neural_network.hpp"

int main() {
    // Example usage of the NeuralNetwork class with Celsius to Fahrenheit conversion
    // Define the neural network structure
    int inputSize = 1;
    std::vector<int> hiddenSizes = {1}; // One hidden layer with 1 neuron
    int outputSize = 1;

    // Create the neural network
    NeuralNetwork nn(inputSize, hiddenSizes, outputSize);

    // Example training data (Celsius to Fahrenheit conversion)
    std::vector<std::vector<double>> trainingData = {
        {0.0}, {10.0}, {20.0}, {30.0}, {40.0}, {50.0}
    };
    std::vector<std::vector<double>> labels = {
        {32.0}, {50.0}, {68.0}, {86.0}, {104.0}, {122.0}
    };

    nn.train(trainingData, labels, 1000, 0.01);

    // Test prediction
    std::vector<double> testInput = {25.0}; // Celsius
    std::cout << "Predicting Fahrenheit for " << testInput[0] << " Celsius..." << std::endl;
    std::vector<double> prediction = nn.predict(testInput);
    std::cout << "Predicted Fahrenheit: " << prediction[0] << std::endl;

    return 0;
}