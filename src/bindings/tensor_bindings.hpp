#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor/tensor.hpp"

namespace py = pybind11;
namespace eth::bind
{

void bindTensor(py::module_& m)
{
    py::class_<eth::Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<std::vector<int>, bool>(), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<float, bool>(), py::arg("value"), py::arg("requires_grad") = false)
        .def(py::init<std::vector<float>, bool>(), py::arg("data"), py::arg("requires_grad") = false)
        .def(py::init<std::vector<float>, std::vector<int>, bool>(), py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false)

        // Properties
        .def_readwrite("requires_grad", &eth::Tensor::requires_grad)
        .def("get_shape", &eth::Tensor::get_shape)
        .def("size", &eth::Tensor::size)
        .def("to_vector", &eth::Tensor::to_vector)

        // Multi-index access
        .def("get", py::overload_cast<const std::vector<int>&>(&eth::Tensor::get, py::const_))
        .def("set", py::overload_cast<const std::vector<int>&, float>(&eth::Tensor::set))

        // Flat index access
        .def("get_flat", py::overload_cast<int>(&eth::Tensor::get, py::const_))
        .def("set_flat", py::overload_cast<int, float>(&eth::Tensor::set))

        // Math operations
        .def("add", &eth::Tensor::add)
        .def("multiply", &eth::Tensor::multiply)
        .def("outer_product", &eth::Tensor::outer_product)
        .def("dot_product", &eth::Tensor::dot_product)
        .def("transpose", &eth::Tensor::transpose)
        .def("contraction", &eth::Tensor::contraction)
        
        // device management
        .def("move_to_gpu", &eth::Tensor::move_to_gpu, py::arg("device_id"))
        .def("move_to_cpu", &eth::Tensor::move_to_cpu)
        .def("current_device", &eth::Tensor::current_device)
        ;
}

} // namespace eth::bind
