#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // for std::vector conversions
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>   // for numpy <-> Vector interop
#include "mlp/neural_network.hpp"
#include "cuda/cuda_helper.cuh"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(eth::Vector)
PYBIND11_MAKE_OPAQUE(eth::Matrix)

PYBIND11_MODULE(etheria, m) {
    m.doc() = "Etheria | Neural Network library (CPU/CUDA backend)";

    // --- enums ---
    py::enum_<eth::mlp::Backend>(m, "Backend")
        .value("CPU", eth::mlp::Backend::CPU)
        .value("CUDA", eth::mlp::Backend::CUDA);

    // --- NeuralNetworkConfig ---
    py::class_<eth::mlp::NeuralNetworkConfig>(m, "NeuralNetworkConfig")
        .def(py::init<>())
        .def_readwrite("input_size", &eth::mlp::NeuralNetworkConfig::input_size)
        .def_readwrite("layers", &eth::mlp::NeuralNetworkConfig::layers)
        .def_readwrite("backend", &eth::mlp::NeuralNetworkConfig::backend)
        .def_readwrite("device_id", &eth::mlp::NeuralNetworkConfig::device_id)
        .def_readwrite("verbose", &eth::mlp::NeuralNetworkConfig::verbose);

    py::enum_<eth::act::ActivationFunctionType>(m, "Activation")
        .value("LINEAR", eth::act::LINEAR)
        .value("SIGMOID", eth::act::SIGMOID)
        .value("RELU", eth::act::RELU)
        .value("TANH", eth::act::TANH)
        .value("SOFTPLUS", eth::act::SOFTPLUS)
        .value("SOFTMAX", eth::act::SOFTMAX);

    // --- Vector and Matrix as alias (std::vector<float>) ---
    // TODO: Matrix needs to have a list of eth.Vector I want to be able to just put [[1,2],[3,4]] so fix that please
    py::bind_vector<eth::Vector>(m, "Vector");
    py::bind_vector<eth::Matrix>(m, "Matrix");

    // --- LayerConfig ---
    py::class_<eth::mlp::Layer>(m, "Layer")
        .def(py::init<int, eth::act::ActivationFunctionType>(),
             py::arg("units"), py::arg("activation"))
        .def("getUnits", &eth::mlp::Layer::getUnits)
        .def("getActivation", &eth::mlp::Layer::getActivation);

    // --- NeuralNetwork ---
    py::class_<eth::mlp::NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const eth::mlp::NeuralNetworkConfig&>())
        .def("getWeights", &eth::mlp::NeuralNetwork::getWeights)
        .def("getBiases", &eth::mlp::NeuralNetwork::getBiases)
        .def("getConfig", &eth::mlp::NeuralNetwork::getConfig, py::return_value_policy::reference_internal)
        .def("setVerbose", &eth::mlp::NeuralNetwork::setVerbose, py::arg("v"))
        .def("useCUDADevice", &eth::mlp::NeuralNetwork::useCUDADevice, py::arg("device_id")=0)
        .def("useCPU", &eth::mlp::NeuralNetwork::useCPU)
        .def("predict", &eth::mlp::NeuralNetwork::predict)
        .def("fit", &eth::mlp::NeuralNetwork::fit,
             py::arg("inputs"), py::arg("targets"),
             py::arg("epochs"), py::arg("learning_rate"),
             py::arg("momentum")=0.9f, py::arg("verbose")=true)
        .def("evaluate", [](const eth::mlp::NeuralNetwork &nn,
                            const std::vector<eth::Vector>& inputs,
                            const std::vector<eth::Vector>& targets) {
                double loss=0, acc=0;
                nn.evaluate(inputs, targets, loss, acc);
                return py::make_tuple(loss, acc);
        });

    // --- CUDA Helper ---
    m.def("is_cuda_available", &eth::cuda::isCUDAAvailable, "Check if CUDA is available");
    m.def("number_cuda_devices", &eth::cuda::numberCUDADevices, "Get the number of CUDA devices");
    m.def("is_cuda_compatible", &eth::cuda::isCUDACompatible, "Check if a CUDA device is compatible");
    m.def("list_cuda_devices", &eth::cuda::listCUDADevices, "List all available CUDA devices");
}
