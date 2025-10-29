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
        .def(py::init<std::vector<int>, bool, int>(), py::arg("shape"), py::arg("requires_grad") = false, py::arg("device_id") = -1)
        .def(py::init<float, bool, int>(), py::arg("data"), py::arg("requires_grad") = false, py::arg("device_id") = -1)
        .def(py::init<std::vector<float>, bool, int>(), py::arg("data"), py::arg("requires_grad") = false, py::arg("device_id") = -1)
        .def(py::init<std::vector<float>, std::vector<int>, bool, int>(), py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false, py::arg("device_id") = -1)

        // Properties
        .def_readwrite("requires_grad", &eth::Tensor::requires_grad)
        .def("get_shape", &eth::Tensor::get_shape)
        .def("get_rank", &eth::Tensor::get_rank)
        .def("size", &eth::Tensor::size)
        .def("to_vector", &eth::Tensor::to_vector)

        // get sub tensor
        .def("get_subtensor", py::overload_cast<size_t>(&eth::Tensor::get_subtensor, py::const_))
        .def("get_subtensor_multi", py::overload_cast<const std::vector<size_t>>(&eth::Tensor::get_subtensor, py::const_))
        
        // Flat index access
        .def("get_flat", py::overload_cast<int>(&eth::Tensor::get, py::const_))
        .def("set_flat", py::overload_cast<int, float>(&eth::Tensor::set))
        
        // Multi-index access
        .def("get_flat_multi", py::overload_cast<const std::vector<int>&>(&eth::Tensor::get, py::const_))
        .def("set_flat_multi", py::overload_cast<const std::vector<int>&, float>(&eth::Tensor::set))

        // Math operations
        .def("add", &eth::Tensor::add)
        .def("add_scalar", &eth::Tensor::add_scalar)
        .def("subtract", &eth::Tensor::subtract)
        .def("multiply", &eth::Tensor::multiply)
        .def("scalar_multiply", &eth::Tensor::scalar_multiply)
        .def("outer_product", &eth::Tensor::outer_product)
        .def("dot_product", &eth::Tensor::dot_product)
        .def("transpose", &eth::Tensor::transpose)
        .def("contraction", &eth::Tensor::contraction)
        
        // device management
        .def("move_to_gpu", &eth::Tensor::move_to_gpu, py::arg("device_id"))
        .def("move_to_cpu", &eth::Tensor::move_to_cpu)
        .def("current_device", &eth::Tensor::current_device)

        .def("reshape", &eth::Tensor::reshape, py::arg("new_shape"))
        ;
}

} // namespace eth::bind
