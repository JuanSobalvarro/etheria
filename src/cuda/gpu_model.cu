
#include "cuda/gpu_model.cuh" 

namespace eth::mlp::cuda
{
    
// life cycle management

GPUModel createModelOnGPU(
    const int input_size,
    const std::vector<Layer>& layers
)
{
    GPUModel model;
    model.num_layers = static_cast<int>(layers.size());
    model.input_size = input_size;
    model.layers = layers; // copy layer metadata to device struct

    cudaMalloc(&model.weights, model.num_layers * sizeof(DeviceMatrix));
    cudaMalloc(&model.biases, model.num_layers * sizeof(DeviceVector));

    for (const auto& layer : layers) 
    {
        int neurons = layer.getUnits();
        if (neurons <= 0) 
        {
            throw std::runtime_error("Layer must have a positive number of neurons");
        }
        
        DeviceMatrix d_weights;
        d_weights.rows = neurons;
        d_weights.cols = (model.num_layers == 1) ? input_size : layers[&layer - &layers[0] - 1].getUnits();
        size_t weight_size = d_weights.rows * d_weights.cols * sizeof(float);
        cudaMalloc(&d_weights.data, weight_size);
        
        // initialize weights depending on its activation function
        // the initialization methods written can initialize the weights in each neuron (vector)
        // so we have to iterate over each neuron and call the method
        std::vector<float> w_init;
        for (int i = 0; i < d_weights.rows * d_weights.cols; i++)
        {
            w_init = init::initialization_distribution(layer.getActivation(), d_weights.cols, d_weights.rows);
            for (int j = 0; j < d_weights.cols; j++)
                d_weights.data[i * d_weights.cols + j] = w_init[j % w_init.size()];  // wrap if necessary
        }
        cudaMemcpy(&model.weights[&layer - &layers[0]], &d_weights, sizeof(DeviceMatrix), cudaMemcpyHostToDevice);
        
        // Allocate and initialize biases to small random values
        DeviceVector d_biases;
        d_biases.size = neurons;
        size_t bias_size = neurons * sizeof(float);
        cudaMalloc(&d_biases.data, bias_size);
        std::vector<float> b_init(neurons);
        for (int i = 0; i < neurons; i++)
            b_init[i] = 0.01f * ((float)rand() / RAND_MAX - 0.5f);
        cudaMemcpy(d_biases.data, b_init.data(), bias_size, cudaMemcpyHostToDevice);
        cudaMemcpy(&model.biases[&layer - &layers[0]], &d_biases, sizeof(DeviceVector), cudaMemcpyHostToDevice);
    }

    return model;
}

GPUModel copyModelToGPU(
    const std::vector<Matrix>& weights,
    const std::vector<Vector>& biases,
    const int input_size,
    const std::vector<Layer>& layers
)
{
    GPUModel model;
    model.num_layers = static_cast<int>(layers.size());
    model.input_size = input_size;
    model.layers = layers; // copy layer metadata to device struct
    model.weights = new DeviceMatrix[model.num_layers];
    model.biases = new DeviceVector[model.num_layers];

    for (int l = 0; l < model.num_layers; l++) 
    {
        int out_units = layers[l].getUnits();
        int in_units = (l == 0) ? input_size : layers[l - 1].getUnits();

        // Allocate and copy weights
        DeviceMatrix d_weights;
        d_weights.rows = out_units;
        d_weights.cols = in_units;
        size_t weight_size = out_units * in_units * sizeof(float);
        cudaMalloc(&d_weights.data, weight_size);
        cudaMemcpy(d_weights.data, weights[l].data(), weight_size, cudaMemcpyHostToDevice);
        model.weights[l] = d_weights;

        // Allocate and copy biases
        DeviceVector d_biases;
        d_biases.size = out_units;
        size_t bias_size = out_units * sizeof(float);
        cudaMalloc(&d_biases.data, bias_size);
        cudaMemcpy(d_biases.data, biases[l].data(), bias_size, cudaMemcpyHostToDevice);
        model.biases[l] = d_biases;
    }

    return model;
}

void freeModelOnGPU(GPUModel& model)
{
    if (model.num_layers <= 0) return; // model not initialized

    // fetch back device pointers to free them one by one
    std::vector<DeviceMatrix> h_weights(model.num_layers);
    std::vector<DeviceVector> h_biases(model.num_layers);

    cudaMemcpy(h_weights.data(), model.weights, model.num_layers * sizeof(DeviceMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_biases.data(),  model.biases,  model.num_layers * sizeof(DeviceVector), cudaMemcpyDeviceToHost);

    for (int l = 0; l < model.num_layers; ++l) {
        if (h_weights[l].data) cudaFree(h_weights[l].data);
        if (h_biases[l].data)  cudaFree(h_biases[l].data);
    }

    cudaFree(model.weights);
    cudaFree(model.biases);

    model.num_layers = 0;
    model.weights = nullptr;
    model.biases = nullptr;
}

// operations

Matrix forward_batch(
    const GPUModel& model,
    const Matrix& input_batch
)
{
    throw std::runtime_error("Not implemented yet");
}

float fit_batch(
    GPUModel& model,
    const Matrix& input_batch,
    const Matrix& target_batch,
    float learning_rate
)
{
    throw std::runtime_error("Not implemented yet");
}

void forward_layer(
    const float* weights,   // flattened [output_size * input_size]
    const float* biases,    // [output_size]
    const float* inputs,    // [input_size]
    float* outputs,         // [output_size]
    int input_size,
    int output_size,
    act::ActivationFunctionType act_type
)
{
    throw std::runtime_error("Not implemented yet");
}

void output_delta(
    const float* outputs,
    const float* targets,
    float* deltas,
    int output_size,
    int batch_size,
    act::ActivationFunctionType act_type
)
{
    throw std::runtime_error("Not implemented yet");
}

void hidden_deltas(
    const float* weights,
    const float* next_deltas,
    const float* outputs,
    float* deltas,
    int input_size,
    int output_size,
    int batch_size,
    act::ActivationFunctionType act_type
)
{
    throw std::runtime_error("Not implemented yet");
}

void update_weights(
    float* weights,
    const float* biases,
    const float* inputs,
    const float* deltas,
    int input_size,
    int output_size,
    int batch_size,
    float learning_rate
)
{
    throw std::runtime_error("Not implemented yet");
}

std::vector<Matrix> getWeightsFromGPU(const GPUModel& model)
{
    std::vector<Matrix> weights(model.num_layers);

    std::vector<DeviceMatrix> h_weights(model.num_layers);
    cudaMemcpy(h_weights.data(), model.weights, model.num_layers * sizeof(DeviceMatrix), cudaMemcpyDeviceToHost);

    for (int l = 0; l < model.num_layers; ++l) {
        int rows = h_weights[l].rows;
        int cols = h_weights[l].cols;
        weights[l].resize(rows, Vector(cols));

        std::vector<float> h_data(rows * cols);
        cudaMemcpy(h_data.data(), h_weights[l].data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                weights[l][i][j] = h_data[i * cols + j];
            }
        }
    }

    return weights;
}

std::vector<Vector> getBiasesFromGPU(const GPUModel& model)
{
    std::vector<Vector> biases(model.num_layers);

    std::vector<DeviceVector> h_biases(model.num_layers);
    cudaMemcpy(h_biases.data(), model.biases, model.num_layers * sizeof(DeviceVector), cudaMemcpyDeviceToHost);

    for (int l = 0; l < model.num_layers; ++l) {
        int size = h_biases[l].size;
        biases[l].resize(size);

        cudaMemcpy(biases[l].data(), h_biases[l].data, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    return biases;
}

} // namespace eth::mlp::cuda
