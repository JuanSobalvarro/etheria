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

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<int, std::vector<int>, int, ActivationFunctionType>())
        .def("train", [](NeuralNetwork& self,
                        const std::vector<std::vector<double>>& data,
                        const std::vector<std::vector<double>>& labels,
                        int epochs, double lr) {
            try {
                self.train(data, labels, epochs, lr);
            } catch (const std::exception& e) {
                throw std::runtime_error("NeuralNetwork::train failed. Check your input shapes and parameters.");
            }
        }, py::arg("training_data"), py::arg("labels"), py::arg("epochs"), py::arg("learning_rate"))
        .def("predict", &NeuralNetwork::predict, py::arg("input_data"))
        .def("printNeuralNetwork", &NeuralNetwork::printNeuralNetwork)
        .def("test", &NeuralNetwork::test, py::arg("test_data"), py::arg("test_labels"));
}
