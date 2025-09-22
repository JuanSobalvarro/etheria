#include "neural_network.hpp"

namespace nfs {

static std::mt19937& global_rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

NeuralNetwork::NeuralNetwork(const NeuralNetworkConfig& cfg): config(cfg) {
    if (config.layer_sizes.size() < 2) {
        throw std::invalid_argument("Need at least input and output layer");
    }
    // He or Xavier initialization depending on activation
    weights.resize(config.layer_sizes.size() - 1);
    biases.resize(config.layer_sizes.size() - 1);

    for (size_t l = 0; l + 1 < config.layer_sizes.size(); ++l) {
        int in = config.layer_sizes[l];
        int out = config.layer_sizes[l+1];
        weights[l].assign(out, Vector(in));
        biases[l].assign(out, 0.0);

        double stddev;
        Activation::ActivationFunctionType act = (l + 1 == config.layer_sizes.size() - 1) ? config.output_activation : config.hidden_activation;
        if (act == Activation::RELU || act == Activation::SOFTPLUS) {
            stddev = std::sqrt(2.0 / in); // He
        } else if (act == Activation::SIGMOID || act == Activation::TANH) {
            stddev = std::sqrt(1.0 / in); // Xavier simple
        } else { // LINEAR
            stddev = std::sqrt(1.0 / in);
        }
        std::normal_distribution<double> dist(0.0, stddev);
        for (int r = 0; r < out; ++r) {
            for (int c = 0; c < in; ++c) {
                weights[l][r][c] = dist(global_rng());
            }
        }
    }
}

double NeuralNetwork::activation(double x, Activation::ActivationFunctionType type) {
    return Activation::forward(type, x);
}

double NeuralNetwork::activation_derivative(double x, Activation::ActivationFunctionType type) {
    return Activation::derivative(type, x);
}

Vector NeuralNetwork::applyActivation(const Vector& z, Activation::ActivationFunctionType type) {
    Vector a(z.size());
    for (size_t i = 0; i < z.size(); ++i) a[i] = activation(z[i], type);
    return a;
}

Vector NeuralNetwork::applyActivationDerivative(const Vector& z, Activation::ActivationFunctionType type) {
    Vector d(z.size());
    for (size_t i = 0; i < z.size(); ++i) d[i] = activation_derivative(z[i], type);
    return d;
}

Vector NeuralNetwork::matvec(const Matrix& W, const Vector& v) {
    Vector out(W.size(), 0.0);
    for (size_t r = 0; r < W.size(); ++r) {
        double sum = 0.0;
        const Vector& row = W[r];
        for (size_t c = 0; c < row.size(); ++c) sum += row[c] * v[c];
        out[r] = sum;
    }
    return out;
}

Matrix NeuralNetwork::outer(const Vector& a, const Vector& b) {
    Matrix m(a.size(), Vector(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) m[i][j] = a[i] * b[j];
    }
    return m;
}

Vector NeuralNetwork::add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector size mismatch");
    Vector r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

void NeuralNetwork::inplace_axpy(Vector& y, const Vector& x, double alpha) {
    if (y.size() != x.size()) throw std::invalid_argument("Vector size mismatch in axpy");
    for (size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
}

Vector NeuralNetwork::predict(const Vector& input) const {
    if (input.size() != static_cast<size_t>(config.layer_sizes.front())) {
        throw std::invalid_argument("Input size mismatch");
    }
    Vector a = input; // activation of previous layer
    for (size_t l = 0; l < weights.size(); ++l) {
        Activation::ActivationFunctionType act = (l == weights.size()-1) ? config.output_activation : config.hidden_activation;
        Vector z = matvec(weights[l], a);
        for (size_t i = 0; i < z.size(); ++i) z[i] += biases[l][i];
        a = applyActivation(z, act);
    }
    return a;
}

void NeuralNetwork::train(const std::vector<Vector>& inputs,
                          const std::vector<Vector>& targets,
                          int epochs,
                          double learning_rate,
                          bool verbose) {
    if (inputs.size() != targets.size()) throw std::invalid_argument("Inputs/targets size mismatch");
    if (inputs.empty()) return;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t sample = 0; sample < inputs.size(); ++sample) {
            const Vector& x = inputs[sample];
            const Vector& y = targets[sample];

            // ----- Forward pass store intermediates -----
            std::vector<Vector> z_values; // pre-activations
            std::vector<Vector> a_values; // activations (a0 = input)
            a_values.push_back(x);
            for (size_t l = 0; l < weights.size(); ++l) {
                Activation::ActivationFunctionType act = (l == weights.size()-1) ? config.output_activation : config.hidden_activation;
                Vector z = matvec(weights[l], a_values.back());
                for (size_t i = 0; i < z.size(); ++i) z[i] += biases[l][i];
                z_values.push_back(z);
                a_values.push_back(applyActivation(z, act));
            }

            const Vector& y_pred = a_values.back();
            if (y_pred.size() != y.size()) throw std::runtime_error("Target size mismatch");

            // MSE loss contribution
            for (size_t i = 0; i < y.size(); ++i) {
                double diff = y_pred[i] - y[i];
                total_loss += 0.5 * diff * diff; // 0.5 for simpler derivative
            }

            // ----- Backward pass -----
            std::vector<Vector> deltas(weights.size()); // delta[l] matches layer l (post-activation index l+1 in a_values)
            // Output delta
            {
                size_t L = weights.size() - 1;
                Activation::ActivationFunctionType act = config.output_activation;
                Vector d_act = applyActivationDerivative(z_values[L], act);
                deltas[L].assign(y.size(), 0.0);
                for (size_t i = 0; i < y.size(); ++i) {
                    double diff = (y_pred[i] - y[i]);
                    deltas[L][i] = diff * d_act[i];
                }
            }
            // Hidden layers
            for (int l = static_cast<int>(weights.size()) - 2; l >= 0; --l) {
                Activation::ActivationFunctionType act = (l == static_cast<int>(weights.size()) - 1) ? config.output_activation : config.hidden_activation;
                Activation::ActivationFunctionType act_hidden = (l == static_cast<int>(weights.size()) - 1) ? config.output_activation : config.hidden_activation;
                (void)act; // silence unused (act_hidden kept for clarity)
                Vector d_act = applyActivationDerivative(z_values[l], (l == static_cast<int>(weights.size()) - 1) ? config.output_activation : config.hidden_activation);
                deltas[l].assign(z_values[l].size(), 0.0);
                for (size_t i = 0; i < weights[l+1].size(); ++i) { // i indexes neurons in layer l+1 (next layer)
                    // weights[l+1][i][j] : from neuron j (layer l) to neuron i (layer l+1)
                }
                // Compute delta_l = (W_{l+1}^T * delta_{l+1}) ⊙ f'(z_l)
                for (size_t j = 0; j < deltas[l].size(); ++j) {
                    double sum = 0.0;
                    for (size_t i = 0; i < deltas[l+1].size(); ++i) {
                        sum += weights[l+1][i][j] * deltas[l+1][i];
                    }
                    deltas[l][j] = sum * d_act[j];
                }
            }

            // ----- Update weights & biases -----
            for (size_t l = 0; l < weights.size(); ++l) {
                const Vector& a_prev = a_values[l];
                for (size_t i = 0; i < weights[l].size(); ++i) { // row
                    for (size_t j = 0; j < weights[l][i].size(); ++j) { // col
                        weights[l][i][j] -= learning_rate * deltas[l][i] * a_prev[j];
                    }
                    biases[l][i] -= learning_rate * deltas[l][i];
                }
            }
        }
        if (verbose) {
            double avg_loss = total_loss / inputs.size();
            std::cout << "[MatrixNet] Epoch " << epoch << " - MSE: " << avg_loss << "\n";
        }
    }
}

/*
* Evaluate the network on a dataset and return accuracy in the range of [0, 1]
*/
void NeuralNetwork::evaluate(const std::vector<Vector>& inputs,
                             const std::vector<Vector>& targets,
                             double& loss,
                             double& accuracy) const {
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
            total_loss += 0.5 * diff * diff; // 0.5 for simpler derivative
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

} // namespace nfs
