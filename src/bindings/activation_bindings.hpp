#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "activation/activation.hpp"
#include "tensor/tensor.hpp"

namespace py = pybind11;
namespace eth::bind
{

void bindActivation(py::module_& m)
{
    py::enum_<eth::ActivationFunction>(m, "ActivationFunction")
        .value("LINEAR", eth::ActivationFunction::LINEAR)
        .value("RELU", eth::ActivationFunction::RELU)
        .value("SIGMOID", eth::ActivationFunction::SIGMOID)
        .value("TANH", eth::ActivationFunction::TANH)
        .export_values();

    m.def("activation", &eth::Activation::apply, py::arg("input"), py::arg("func"));
    m.def("activation_derivative", &eth::Activation::derivative, py::arg("input"), py::arg("func"));
}

} // namespace eth::bind
