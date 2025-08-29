#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "neural_network.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neuralscratch, m) {
    py::enum_<ActivationFunctionType>(m, "ActivationFunctionType")
        .value("LINEAR", ActivationFunctionType::LINEAR)
        .value("RELU", ActivationFunctionType::RELU)
        .value("SIGMOID", ActivationFunctionType::SIGMOID)
        .value("TANH", ActivationFunctionType::TANH)
        .value("SOFTPLUS", ActivationFunctionType::SOFTPLUS)
        .export_values();

        // Matrix-based network bindings
        py::class_<nfs::MatrixNetworkConfig>(m, "MatrixNetworkConfig")
                .def(py::init<>())
                .def_readwrite("layer_sizes", &nfs::MatrixNetworkConfig::layer_sizes)
                .def_readwrite("hidden_activation", &nfs::MatrixNetworkConfig::hidden_activation)
                .def_readwrite("output_activation", &nfs::MatrixNetworkConfig::output_activation);

        py::class_<nfs::MatrixNetwork>(m, "MatrixNetwork")
                .def(py::init([](const std::vector<int>& layer_sizes,
                                                ActivationFunctionType hidden_act = ActivationFunctionType::RELU,
                                                ActivationFunctionType output_act = ActivationFunctionType::LINEAR) {
                        nfs::MatrixNetworkConfig cfg; cfg.layer_sizes = layer_sizes; cfg.hidden_activation = hidden_act; cfg.output_activation = output_act; return new nfs::MatrixNetwork(cfg);
                }), py::arg("layer_sizes"), py::arg("hidden_activation") = ActivationFunctionType::RELU, py::arg("output_activation") = ActivationFunctionType::LINEAR,
                     R"doc(Create a matrix-based neural network.

Args:
    layer_sizes: List of layer sizes including input and output (e.g. [784, 128, 64, 10]).
    hidden_activation: Activation for hidden layers.
    output_activation: Activation for the output layer.
)doc")
                .def("predict", &nfs::MatrixNetwork::predict, py::arg("input"), R"doc(Forward pass returning output activations.)doc")
                .def("train", [](nfs::MatrixNetwork& self,
                                                    const std::vector<std::vector<double>>& data,
                                                    const std::vector<std::vector<double>>& labels,
                                                    int epochs, double lr, bool verbose) {
                        if (data.empty()) return; // no-op
                        if (data.size() != labels.size()) throw std::runtime_error("data and labels size mismatch");
                        self.train(data, labels, epochs, lr, verbose);
                }, py::arg("data"), py::arg("labels"), py::arg("epochs"), py::arg("learning_rate"), py::arg("verbose") = true,
                     R"doc(Train the network with per-sample SGD.

Args:
    data: List of input vectors.
    labels: List of target vectors (same length as data).
    epochs: Number of epochs.
    learning_rate: SGD learning rate.
    verbose: Print loss each epoch.
)doc")
                .def("evaluate", [](const nfs::MatrixNetwork& self,
                                     const std::vector<std::vector<double>>& inputs,
                                     const std::vector<std::vector<double>>& targets) {
                        double loss = 0.0, acc = 0.0;
                        self.evaluate(inputs, targets, loss, acc);
                        return py::make_tuple(loss, acc);
                }, py::arg("inputs"), py::arg("targets"),
                    R"doc(Evaluate the network on a dataset.

Returns:
    (loss, accuracy) tuple where:
        loss: Mean sample loss (MSE * 0.5 factor applied per sample in training).
        accuracy: Fraction in [0,1] for one-hot classification targets.
)doc")
                .def_property_readonly("weights", &nfs::MatrixNetwork::getWeights, R"doc(List of weight matrices (rows = out_features, cols = in_features).)doc")
                .def_property_readonly("biases", &nfs::MatrixNetwork::getBiases, R"doc(List of bias vectors per layer.)doc");
}
