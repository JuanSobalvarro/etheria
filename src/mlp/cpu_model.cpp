#include "mlp/cpu_model.hpp"
#include "initialization/initialization.hpp"
#include "alg/ops.hpp"
#include "utils.hpp"
#include <cmath>
#include <stdexcept>
#include <random>

namespace eth::mlp::cpu
{

CPUModel create_cpu_model(const std::vector<Layer>& layers, int input_size)
{
    CPUModel model;
    model.layers = layers;

    int prev_size = input_size;

    for (const auto& layer : layers)
    {
        int neurons = layer.getUnits();

        if (layer.getActivation() == act::ActivationFunctionType::SOFTMAX && neurons <= 1)
            throw std::invalid_argument("Softmax layer must have more than one neuron");

        // Weight matrix: each neuron gets 'prev_size' weights
        Matrix W(neurons, Vector(prev_size));
        for (int i = 0; i < neurons; ++i)
        {
            Vector w_init = init::initialization_distribution(layer.getActivation(), prev_size, neurons);
            W[i] = std::move(w_init);  // assign directly
        }

        Vector b(neurons);

        if (layer.getActivation() == act::ActivationFunctionType::SOFTMAX && neurons > 1)
        {
            b = Vector(neurons, 0.0f); // Initialize biases to zero
        }
        else
        {
            std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
            for (auto &val : b)
                val = dist(utils::global_rng());
        }
        // b = Vector(neurons, 0.0f); // Initialize biases to zero

        model.weights.push_back(std::move(W));
        model.biases.push_back(std::move(b));

        prev_size = neurons;
    }

    return model;
}

// -------------------- Forward Layer --------------------
void forward_layer(const float** weights,
                   const float* biases,
                   const float* input,
                   float* output,
                   int input_size,
                   int output_size,
                   act::ActivationFunctionType act_type)
{
    Vector input_vec(input, input + input_size);
    Vector z(output_size, 0.0f);

    // Compute weighted sum + bias
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
            z[i] += weights[i][j] * input_vec[j];
        z[i] += biases[i];
    }

    Vector a = eth::alg::applyFunction(z, act_type);

    for (int i = 0; i < output_size; ++i)
        output[i] = a[i];
}

void forward_layer(
    const CPUModel& model,
    int layer_index,
    const Vector& input,
    Vector& output
)
{
     if (layer_index < 0 || layer_index >= model.layers.size())
          throw std::out_of_range("Invalid layer index");
    
     int input_size = input.size();
     int output_size = model.layers[layer_index].getUnits();
    
     if (output.size() != output_size)
          output.resize(output_size);
    
     std::vector<const float*> W_ptrs(output_size);
     for (int i = 0; i < output_size; ++i)
          W_ptrs[i] = model.weights[layer_index][i].data();
    
     forward_layer(W_ptrs.data(),
                    model.biases[layer_index].data(),
                    input.data(),
                    output.data(),
                    input_size,
                    output_size,
                    model.layers[layer_index].getActivation());
}

Vector forward(
    const CPUModel& model,
    const Vector& input
)
{
    Vector output = input;
    
    for (int i = 0; i < model.layers.size(); i++)
    {
        int output_size = model.layers[i].getUnits();
        Vector next_output(output_size, 0.0f);

        std::vector<const float*> W_ptrs(output_size);
        for (int j = 0; j < output_size; ++j)
            W_ptrs[j] = model.weights[i][j].data();

        forward_layer(W_ptrs.data(),
                      model.biases[i].data(),
                      output.data(),
                      next_output.data(),
                      output.size(),
                      output_size,
                      model.layers[i].getActivation());

        output = std::move(next_output);
    }

    return output;
}

// -------------------- Forward Batch --------------------
Matrix forward_batch(const CPUModel& model, const Matrix& input_batch)
{
    Matrix output(input_batch.size());

    for (int i = 0; i < input_batch.size(); i++)
    {
        output[i] = forward(model, input_batch[i]);
    }

    return output;
}

// -------------------- Fit Batch --------------------
float fit_batch(CPUModel& model, const Matrix& input_batch, const Matrix& target_batch,
                float learning_rate, float momentum)
{
    if (input_batch.size() != target_batch.size())
        throw std::invalid_argument("Input and target batch size mismatch");

    float total_loss = 0.0f;

    // -------------------- Initialize velocities --------------------
    std::vector<Matrix> weight_velocities(model.weights.size());
    std::vector<Vector> bias_velocities(model.biases.size());

    for (size_t l = 0; l < model.layers.size(); ++l)
    {
        int neurons_out = model.layers[l].getUnits();
        int neurons_in  = model.weights[l][0].size();

        weight_velocities[l].resize(neurons_out, Vector(neurons_in, 0.0f));
        bias_velocities[l].resize(neurons_out, 0.0f);
    }

    // -------------------- Loop over samples (SGD) --------------------
    for (size_t sample_idx = 0; sample_idx < input_batch.size(); ++sample_idx)
    {
        const Vector& x = input_batch[sample_idx];
        const Vector& y_true = target_batch[sample_idx];

        // -------------------- Forward pass --------------------
        std::vector<Vector> activations(model.layers.size() + 1);
        activations[0] = x;

        for (size_t l = 0; l < model.layers.size(); ++l)
        {
            activations[l + 1].resize(model.layers[l].getUnits());
            forward_layer(model, l, activations[l], activations[l + 1]);

            // Apply softmax if the user set SOFTMAX for this layer
            if (model.layers[l].getActivation() == act::ActivationFunctionType::SOFTMAX)
                activations[l + 1] = eth::act::softmax(activations[l + 1]);
        }

        // -------------------- Compute output delta --------------------
        Vector delta(activations.back().size());

        if (model.layers.back().getActivation() == act::ActivationFunctionType::SOFTMAX)
        {
            // Softmax + Cross-Entropy: delta = y_pred - y_true
            delta = activations.back();
            for (size_t i = 0; i < delta.size(); ++i)
            {
                delta[i] -= y_true[i];
                total_loss += -y_true[i] * std::log(std::max(1e-15f, activations.back()[i])); // CE loss
            }
        }
        else
        {
            // Other activations: MSE
            for (size_t i = 0; i < delta.size(); ++i)
            {
                delta[i] = activations.back()[i] - y_true[i];
                total_loss += 0.5f * delta[i] * delta[i];
            }
        }

        // -------------------- Backward pass --------------------
        for (int l = model.layers.size() - 1; l >= 0; --l)
        {
            int neurons_out = model.layers[l].getUnits();
            int neurons_in  = activations[l].size();

            // Multiply delta by derivative of activation (skip for softmax output, already included)
            if (!(l == (int)model.layers.size() - 1 &&
                  model.layers[l].getActivation() == act::ActivationFunctionType::SOFTMAX))
            {
                for (int i = 0; i < neurons_out; ++i)
                {
                    delta[i] *= eth::act::forward_activation_derivative(
                                    model.layers[l].getActivation(),
                                    activations[l + 1][i]);
                }
            }

            // Compute weight gradients
            Matrix weight_gradients = eth::alg::outerproduct(delta, activations[l]);
            Vector bias_gradients = delta;

            // Update weights and biases with momentum
            for (int i = 0; i < neurons_out; ++i)
            {
                for (int j = 0; j < neurons_in; ++j)
                {
                    weight_velocities[l][i][j] = momentum * weight_velocities[l][i][j]
                                                 - learning_rate * weight_gradients[i][j];
                    model.weights[l][i][j] += weight_velocities[l][i][j];
                }

                bias_velocities[l][i] = momentum * bias_velocities[l][i]
                                        - learning_rate * bias_gradients[i];
                model.biases[l][i] += bias_velocities[l][i];
            }

            // Compute delta for previous layer
            if (l > 0)
            {
                Vector new_delta(neurons_in, 0.0f);
                for (int i = 0; i < neurons_out; ++i)
                    for (int j = 0; j < neurons_in; ++j)
                        new_delta[j] += model.weights[l][i][j] * delta[i];

                delta = new_delta;
            }
        }
    }

    return total_loss / input_batch.size();
}




// -------------------- Utilities --------------------
std::vector<Matrix> getWeights(const CPUModel& model) { return model.weights; }
std::vector<Vector> getBiases(const CPUModel& model) { return model.biases; }

} // namespace eth::mlp::cpu
