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
        .def(py::init<>())
        .def(py::init<std::vector<int>>(), py::arg("shape"))
        .def(py::init<std::vector<int>, float*>(), py::arg("shape"), py::arg("data"))
        .def("get", &eth::Tensor::get, py::arg("indices"))
        .def("set", &eth::Tensor::set, py::arg("indices"), py::arg("value"))
        .def("get_shape", &eth::Tensor::get_shape)
        .def("get_num_elements", &eth::Tensor::get_num_elements)
        .def("get_current_device_id", &eth::Tensor::get_current_device_id)
        .def("to_cpu", &eth::Tensor::to_cpu)
        .def("to_gpu", &eth::Tensor::to_gpu, py::arg("device_id") = 0)
        .def("add", &eth::Tensor::add, py::arg("other"))
        .def("multiply", &eth::Tensor::multiply, py::arg("other"));
}

} // namespace eth::bind
