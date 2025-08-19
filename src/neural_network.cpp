#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int> hiddenLayerSizes, int outputSize, const ActivationFunctionType activationFunctionType) 
{
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Input and output sizes must be greater than zero");
    }

    if (hiddenLayerSizes.empty()) {
        throw std::invalid_argument("Hidden layer sizes must be specified");
    }

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
    std::vector<Connection*> outputConnections(hiddenLayers.back()->getNeurons().size(), nullptr);

    // Initialize output connections of the output layer
    for (size_t i = 0; i < outputConnections.size(); i++) {
        Connection* conn = new Connection();
        outputConnections[i] = conn;
    }

    this->outputLayer = new Layer(outputSize, outputConnections, activationFunctionType);
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

/*
* Each element = how far the prediction is from the target.
* This is Δoutput for training.
*/
std::vector<double> computeOutputErrors(const std::vector<double>& expected, const std::vector<double>& predicted) {
    if (expected.size() != predicted.size())
        throw std::invalid_argument("Expected and predicted sizes do not match");

    std::vector<double> errors(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        errors[i] = expected[i] - predicted[i];  // simple difference
    }
    return errors;
}


void NeuralNetwork::train(
    const std::vector<std::vector<double>>& trainingData,
    const std::vector<std::vector<double>>& labels,
    int epochs,
    double learningRate
) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalError = 0.0;

        for (size_t i = 0; i < trainingData.size(); i++) {
            std::vector<double> inputs = trainingData[i];
            std::vector<double> expected = labels[i];

            // --------- Forward pass ---------
            std::vector<double> predicted = predict(inputs);

            // std::cout << "Predicting okay!!!\n";

            // --------- Compute output layer error ---------
            std::vector<double> outputErrors = computeOutputErrors(expected, predicted);

            // std::cout << "Output errors computed!!!\n";

            // Accumulate total error for monitoring
            for (double e : outputErrors) totalError += e * e;

            // --------- Backward pass ---------

            // Step 1: Compute output layer deltas
            std::vector<double> outputDeltas(outputLayer->getNeurons().size());
            for (size_t j = 0; j < outputLayer->getNeurons().size(); j++) {
                Neuron* neuron = outputLayer->getNeurons()[j];
                outputDeltas[j] = outputErrors[j] * neuron->getActivationFunction()->derivative(predicted[j]);
            }

            // Step 2: Compute hidden layer deltas (backprop)
            std::vector<std::vector<double>> hiddenDeltas(hiddenLayers.size());
            for (int l = hiddenLayers.size() - 1; l >= 0; l--) {
                Layer* layer = hiddenLayers[l];
                hiddenDeltas[l].resize(layer->getNeurons().size());

                for (size_t n = 0; n < layer->getNeurons().size(); n++) {
                    Neuron* neuron = layer->getNeurons()[n];
                    double output = neuron->getOutput()->getValue(); // neuron output
                    double sum = 0.0;

                    if (l == (int)hiddenLayers.size() - 1) {
                        // last hidden layer, connect to output layer
                        for (size_t j = 0; j < outputLayer->getNeurons().size(); j++)
                            sum += outputDeltas[j] * outputLayer->getNeurons()[j]->getWeights()[n];
                    } else {
                        // hidden layer connects to next hidden layer
                        for (size_t j = 0; j < hiddenLayers[l+1]->getNeurons().size(); j++)
                            sum += hiddenDeltas[l+1][j] * hiddenLayers[l+1]->getNeurons()[j]->getWeights()[n];
                    }

                    hiddenDeltas[l][n] = sum * neuron->getActivationFunction()->derivative(output);
                }
            }

            // Step 3: Update output layer weights and biases
            for (size_t j = 0; j < outputLayer->getNeurons().size(); j++) {
                Neuron* neuron = outputLayer->getNeurons()[j];
                std::vector<double> neuronInputs = hiddenLayers.empty() 
                    ? inputLayer->getOutputsValue() 
                    : hiddenLayers.back()->getOutputsValue();

                for (size_t w = 0; w < neuron->getWeights().size(); w++)
                    neuron->getWeights()[w] += learningRate * outputDeltas[j] * neuronInputs[w];

                neuron->setBias(neuron->getBias() + learningRate * outputDeltas[j]);
            }

            // Step 4: Update hidden layer weights and biases
            for (size_t l = 0; l < hiddenLayers.size(); l++) {
                Layer* layer = hiddenLayers[l];
                std::vector<double> neuronInputs = (l == 0) 
                    ? inputLayer->getOutputsValue() 
                    : hiddenLayers[l-1]->getOutputsValue();

                for (size_t n = 0; n < layer->getNeurons().size(); n++) {
                    Neuron* neuron = layer->getNeurons()[n];

                    for (size_t w = 0; w < neuron->getWeights().size(); w++)
                        neuron->getWeights()[w] += learningRate * hiddenDeltas[l][n] * neuronInputs[w];

                    neuron->setBias(neuron->getBias() + learningRate * hiddenDeltas[l][n]);
                }
            }
        }

        // Print RMSE every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch 
                      << ", RMSE: " << std::sqrt(totalError / trainingData.size()) << "\n";
        }
    }
}



void NeuralNetwork::saveModel(const std::string& filename) 
{
}

void NeuralNetwork::loadModel(const std::string& filename) 
{
}


void NeuralNetwork::printNeuralNetwork() const
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
