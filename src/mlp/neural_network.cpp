#include "mlp/neural_network.hpp"


namespace eth::mlp {

NeuralNetwork::NeuralNetwork(const NeuralNetworkConfig& cfg): config(cfg) 
{
    if (config.layers.empty()) 
    {
        throw std::invalid_argument("NeuralNetwork must have at least one layer");
    }
    if (config.input_size <= 0) 
    {
        throw std::invalid_argument("Input size must be positive");
    }

    initializeNN();
}

void NeuralNetwork::initializeNN()
{
    if (config.input_size <= 0)
        throw std::invalid_argument("Input size must be positive");

    if (config.layers.empty())
        throw std::invalid_argument("Network must have at least one layer");

    if (config.backend == Backend::CPU) 
    {
        cpu_model = std::make_unique<eth::mlp::cpu::CPUModel>(
            eth::mlp::cpu::create_cpu_model(config.layers, config.input_size)
        );
    } 
    else if (config.backend == Backend::CUDA) 
    {
        gpu_model = std::make_unique<eth::mlp::cuda::GPUModel>(
            eth::mlp::cuda::createModelOnGPU(config.input_size, config.layers)
        );
    } 
    else 
    {
        throw std::runtime_error("Unknown backend");
    }

    if (config.verbose) {
        std::cout << "Neural Network initialized with "
                  << config.layers.size() << " layers. At Backend: " << (config.backend == Backend::CPU ? "CPU" : "CUDA") << std::endl;
    }
}


void NeuralNetwork::useCUDADevice(int device_id) 
{
    config.backend = Backend::CUDA;
    config.device_id = device_id;
    eth::cuda::setDevice(device_id);

    if (!gpu_model) 
    {
        if (!cpu_model)
        {
            gpu_model = std::make_unique<eth::mlp::cuda::GPUModel>(
                eth::mlp::cuda::createModelOnGPU(config.input_size, config.layers)
            );
        }
        else
        {
            gpu_model = std::make_unique<eth::mlp::cuda::GPUModel>(
                eth::mlp::cuda::copyModelToGPU(cpu_model->weights, cpu_model->biases, config.input_size, config.layers)
            );
        }
    }

    // Optionally free CPU model to save memory
    cpu_model.reset();
}

void NeuralNetwork::useCPU() 
{
    config.backend = Backend::CPU;
    config.device_id = -1;

    if (!cpu_model)  // allocate CPU model only when needed
        cpu_model = std::make_unique<eth::mlp::cpu::CPUModel>(
            eth::mlp::cpu::create_cpu_model(config.layers, config.input_size)
        );

    // Optionally free GPU model to save memory
    gpu_model.reset();
}

const std::vector<Matrix>& NeuralNetwork::getWeights() const 
{
    if (config.backend == Backend::CPU && cpu_model) 
    {
        return cpu_model->weights;
    } 
    else if (config.backend == Backend::CUDA && gpu_model) 
    {
        // Note: This is a shallow copy; for deep copy, implement a method to download from GPU
        throw std::runtime_error("Direct access to GPU weights not implemented");
    } 
    else 
    {
        throw std::runtime_error("Model not initialized");
    }
}

const std::vector<Vector>& NeuralNetwork::getBiases() const 
{
    if (config.backend == Backend::CPU && cpu_model) 
    {
        return cpu_model->biases;
    } 
    else if (config.backend == Backend::CUDA && gpu_model) 
    {
        // Note: This is a shallow copy; for deep copy, implement a method to download from GPU
        throw std::runtime_error("Direct access to GPU biases not implemented");
    } 
    else 
    {
        throw std::runtime_error("Model not initialized");
    }
}

Vector NeuralNetwork::predict(const Vector& input) const
{
    Vector result;
    if (input.size() != static_cast<size_t>(config.input_size))
        throw std::invalid_argument("Input size mismatch");

    if (config.backend == Backend::CPU)
    {
        if (!cpu_model)
            throw std::runtime_error("CPU model not initialized");

        Matrix input_batch = {input};
        Matrix output_batch = eth::mlp::cpu::forward_batch(*cpu_model, input_batch);
        result = output_batch[0];
    }
    else if (config.backend == Backend::CUDA)
    {
        if (!gpu_model)
            throw std::runtime_error("GPU model not initialized");

        Matrix input_batch = {input};
        Matrix output_batch = eth::mlp::cuda::forward_batch(*gpu_model, input_batch);
        result = output_batch[0];
    }
    else
    {
        throw std::runtime_error("Unknown backend");
    }
    return result;
}

void NeuralNetwork::fit(
    const Matrix& inputs,
    const Matrix& targets,
    int epochs,
    float learning_rate,
    float momentum,
    bool verbose
) 
{
    if (inputs.size() != targets.size()) throw std::invalid_argument("Inputs/targets size mismatch");
    if (inputs.empty()) throw std::invalid_argument("Empty training data");

    if (config.backend == Backend::CPU) 
    {
        if (!cpu_model)
            throw std::runtime_error("CPU model not initialized");

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float loss = eth::mlp::cpu::fit_batch(
                *cpu_model, inputs, targets, learning_rate, momentum
            );
            if (verbose) std::cout << "Epoch " << epoch << " - MSE: " << loss << std::endl;
        }
    } 
    else if (config.backend == Backend::CUDA) 
    {
        if (!gpu_model)
            throw std::runtime_error("GPU model not initialized");

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float loss = eth::mlp::cuda::fit_batch(*gpu_model, inputs, targets, float(learning_rate));
            if (verbose) std::cout << "Epoch " << epoch << " - MSE: " << loss << std::endl;
        }
    }
    else 
    {
        throw std::runtime_error("Unknown backend");
    }
}
 

/*
* Evaluate the network on a dataset and return accuracy in the range of [0, 1]
*/
void NeuralNetwork::evaluate(
    const std::vector<Vector>& inputs,
    const std::vector<Vector>& targets,
    double& loss,
    double& accuracy
) const 
{
    if (inputs.size() != targets.size()) throw std::invalid_argument("Inputs/targets size mismatch");
    if (inputs.empty()) {
        loss = 0.0;
        accuracy = 0.0;
        return;
    }

    double total_loss = 0.0;
    int correct = 0;

    for (size_t sample = 0; sample < inputs.size(); ++sample) {
        const Vector& x = inputs[sample];
        const Vector& y = targets[sample];

        Vector y_pred = predict(x);
        if (y_pred.size() != y.size()) throw std::runtime_error("Target size mismatch");

        // MSE loss contribution
        for (size_t i = 0; i < y.size(); ++i) {
            double diff = y_pred[i] - y[i];
            total_loss += 0.5 * diff * diff / y.size(); // 0.5 for simpler derivative
        }

        // Accuracy (for classification, assuming one-hot targets)
        size_t pred_class = std::distance(y_pred.begin(), std::max_element(y_pred.begin(), y_pred.end()));
        size_t true_class = std::distance(y.begin(), std::max_element(y.begin(), y.end()));
        if (pred_class == true_class) {
            correct++;
        }
    }

    loss = total_loss / inputs.size();
    accuracy = static_cast<double>(correct) / inputs.size();
}

} // namespace eth::mlp
