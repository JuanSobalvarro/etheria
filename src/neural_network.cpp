#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int> hiddenLayerSizes, int outputSize, const ActivationFunctionType activationFunctionType) 
{
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Input and output sizes must be greater than zero");
    }

    // if (hiddenLayerSizes.empty()) {
    //     throw std::invalid_argument("Hidden layer sizes must be specified");
    // }

    if (activationFunctionType < LINEAR || activationFunctionType > SOFTPLUS) {
        throw std::invalid_argument("Invalid activation function type");
    }

    // Initialize input layer
    this->inputConnections = std::vector<Connection*>(inputSize, nullptr);

    for (int i = 0; i < inputSize; i++) {
        Connection* conn = new Connection();
        this->inputConnections[i] = conn;
    }

    this->inputLayer = new Layer(inputSize, inputConnections, activationFunctionType);

    // Initialize hidden layers
    for (size_t i = 0; i < hiddenLayerSizes.size(); i++) 
    {
        // layer size is the number of neurons of the layer
        int currentLayerSize = hiddenLayerSizes[i];
        std::vector<Connection*> prevLayerOutputs;

        if (i == 0) {
            // First hidden layer, connect to input layer
            prevLayerOutputs = this->inputLayer->getOutputs();
        } else {
            // Subsequent hidden layers, connect to previous hidden layer
            prevLayerOutputs = hiddenLayers[i - 1]->getOutputs();
        }

        Layer* new_layer = new Layer(currentLayerSize, prevLayerOutputs, activationFunctionType);
        hiddenLayers.push_back(new_layer);
    }

    // Initialize output layer
    std::vector<Connection*> outputLayerInputs;
    if (hiddenLayers.empty()) {
        outputLayerInputs = std::vector<Connection*>(inputLayer->getNeurons().size(), nullptr);
        for (size_t i = 0; i < inputLayer->getNeurons().size(); i++) {
            outputLayerInputs[i] = inputLayer->getOutputs()[i];
        }
    }
    else {
        outputLayerInputs = std::vector<Connection*>(hiddenLayers.back()->getNeurons().size(), nullptr);
        for (size_t i = 0; i < hiddenLayers.back()->getNeurons().size(); i++) {
            outputLayerInputs[i] = hiddenLayers.back()->getOutputs()[i];
        }
    }

    this->outputLayer = new Layer(outputSize, outputLayerInputs, activationFunctionType);
}

NeuralNetwork::~NeuralNetwork() 
{
    // Destruct each layer
    delete inputLayer;
    for (size_t i = 0; i < hiddenLayers.size(); i++) 
    {
        delete hiddenLayers[i];
    }
    delete outputLayer;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) 
{   
    // Check if the input Layer inputs are initialized
    if (inputLayer->getInputs().empty()) {
        throw std::runtime_error("Input Layer is not initialized");
    }

    // Check if the sizes are the same
    if (inputs.size() != inputLayer->getInputs().size()) {
        throw std::invalid_argument("Input size does not match the network input size");
    }

    std::vector<Connection*> firstInputs = inputLayer->getInputs();
    int inputSize = inputs.size();

    for (int i = 0; i < inputSize; i++) {
        Connection* conn = firstInputs[i];
        if (conn) {
            conn->changeValue(inputs[i]);
        }
    }

    // std::cout << "Activating first layer...\n";

    // activate first layer
    inputLayer->activate();

    // std::cout << "Input layer activated.\n";

    // Activate hidden layers
    for (size_t i = 0; i < hiddenLayers.size(); i++) 
    {
        hiddenLayers[i]->activate();
    }
    // std::cout << "Hidden layers activated.\n";

    // Activate output layer
    outputLayer->activate();

    // Return output layer values
    // std::cout << "Prediction complete.\n";
    return outputLayer->getOutputsValue();
}


void NeuralNetwork::train(
    const std::vector<std::vector<double>>& trainingData,
    const std::vector<std::vector<double>>& labels,
    int epochs,
    double learningRate
) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

        for (size_t sampleIdx = 0; sampleIdx < trainingData.size(); sampleIdx++) {
            std::vector<double> inputs = trainingData[sampleIdx];
            std::vector<double> expected = labels[sampleIdx];

            // ===== 1. Forward pass =====
            std::vector<double> predicted = predict(inputs);

            // Compute error for RMSE
            for (size_t j = 0; j < predicted.size(); j++) {
                double error = predicted[j] - expected[j];
                totalError += error * error;
            }

            // ===== 2. Compute output layer deltas =====
            for (size_t j = 0; j < outputLayer->getNeurons().size(); j++) {
                Neuron* neuron = outputLayer->getNeurons()[j];
                double z = neuron->getZValue(); // pre-activation value
                double error = predicted[j] - expected[j];
                // std::cout << "Output Neuron " << j << " - Predicted: " << predicted[j] << ", Expected: " << expected[j] << ", Error: " << error << "\n";
                neuron->setDelta(error * neuron->getActivationFunction()->derivative(z));
            }

            // ===== 3. Backpropagate hidden layers =====
            for (int l = (int)hiddenLayers.size() - 1; l >= 0; l--) {
                Layer* currentLayer = hiddenLayers[l];
                Layer* nextLayer = (l == (int)hiddenLayers.size() - 1) ? outputLayer : hiddenLayers[l + 1];

                for (size_t j = 0; j < currentLayer->getNeurons().size(); j++) {
                    Neuron* neuron = currentLayer->getNeurons()[j];
                    double sum = 0.0;

                    for (size_t k = 0; k < nextLayer->getNeurons().size(); k++) {
                        Neuron* nextNeuron = nextLayer->getNeurons()[k];
                        sum += nextNeuron->getWeights()[j] * nextNeuron->getDelta();
                    }

                    double z = neuron->getZValue();
                    neuron->setDelta(sum * neuron->getActivationFunction()->derivative(z));
                }
            }

            // ===== 4. Update weights and biases =====
            // Output layer
            for (Neuron* neuron : outputLayer->getNeurons()) {
                neuron->updateWeightsTraining(learningRate);
                neuron->updateBiasTraining(learningRate);
            }

            // Hidden layers
            for (Layer* layer : hiddenLayers) {
                for (Neuron* neuron : layer->getNeurons()) {
                    neuron->updateWeightsTraining(learningRate);
                    neuron->updateBiasTraining(learningRate);
                }
            }

            // Input layer
            for (Neuron* neuron : inputLayer->getNeurons()) {
                neuron->updateWeightsTraining(learningRate);
                neuron->updateBiasTraining(learningRate);
            }
        }

        // Print RMSE every 100 epochs
        if (epoch % 100 == 0) {
            double rmse = std::sqrt(totalError / trainingData.size());
            std::cout << "Epoch " << epoch << ", RMSE: " << rmse << "\n";
        }
    }


}


void NeuralNetwork::test(std::vector<std::vector<double>>& testData, std::vector<std::vector<double>>& testLabels)
{
    std::cout << "Testing Neural Network...\n";
    double error = 0.0;
    double percentageError = 0.0;

    for (size_t i = 0; i < testData.size(); i++) {
        std::vector<double> input = testData[i];
        std::vector<double> expected = testLabels[i];
        std::vector<double> output = predict(input);

        // Compute error (e.g., RMSE)
        for (size_t j = 0; j < expected.size(); j++) {
            double diff = output[j] - expected[j];
            error += diff * diff;
            percentageError += std::abs(diff);
        }
    }

    error = std::sqrt(error / testData.size());
    percentageError = (1 - percentageError / testData.size()) * 100;
    std::cout << "Testing completed. RMSE: " << error << "\n";
    std::cout << "Fiability: " << percentageError << "%\n";
}


void NeuralNetwork::saveModel(const std::string& filename) 
{
}

void NeuralNetwork::loadModel(const std::string& filename) 
{
}


void NeuralNetwork::printNeuralNetwork()
{
    std::cout << "\n=============================\n";
    std::cout << "   Neural Network Structure   \n";
    std::cout << "=============================\n";

    // Input Layer
    std::cout << "Input Layer (" << inputLayer->getNeurons().size() << " neurons)\n";
    std::cout << "    ";
    for (Neuron* neuron : inputLayer->getNeurons())
    {
        std::cout << "\tNEURON <+>\n";
        std::cout << "\t\tBias: " << neuron->getBias() << "\n";
        std::cout << "\t\tWeights: [ ";
        for (double weight: neuron->getWeights()) 
        {
            std::cout << weight << " ";
        }
        std::cout << "]\n";
    } 
    std::cout << "\n";
    
    // Hidden Layers
    for (size_t i = 0; i < hiddenLayers.size(); i++) 
    {
        std::cout << "Hidden Layer " << i + 1 
                  << " (" << hiddenLayers[i]->getNeurons().size() << " neurons)\n";
        std::cout << "    ";
        for (size_t j = 0; j < hiddenLayers[i]->getNeurons().size(); j++)
        {
            std::cout << "\tNEURON <+>\n";
            std::cout << "\t\tBias: " << hiddenLayers[i]->getNeurons()[j]->getBias() << "\n";
            std::cout << "\t\tWeights: [ ";
            for (double weight: hiddenLayers[i]->getNeurons()[j]->getWeights()) 
            {
                std::cout << weight << " ";
            }
            std::cout << "]\n";
        }
    }

    // Output Layer
    std::cout << "Output Layer (" << outputLayer->getNeurons().size() << " neurons)\n";
    for (size_t i = 0; i < outputLayer->getNeurons().size(); i++)
    {
        std::cout << "\tNEURON <+>\n";
        std::cout << "\t\tBias: " << outputLayer->getNeurons()[i]->getBias() << "\n";
        std::cout << "\t\tWeights: [ ";
        for (double weight: outputLayer->getNeurons()[i]->getWeights()) 
        {
            std::cout << weight << " ";
        }
        std::cout << "]\n";
    }
}
