#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor/tensor.hpp"
#include "tensor/ops.hpp"

namespace py = pybind11;
namespace eth::bind
{

void bindTensor(py::module_& m)
{
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>>())
        .def("fill", &Tensor::fill)
        .def("add", &Tensor::add)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("shape", &Tensor::shape)
        .def("__getitem__", &Tensor::get_item)
        .def("__setitem__", &Tensor::set_item);

    // add bindings to inplace operations
    m.def("add", &add, "In-place addition of tensor b to tensor a");
    m.def("matmul", &matmul, "In-place matrix multiplication of tensor a by tensor b");
    m.def("activation", &activation, "Apply activation function to tensor",
          py::arg("input"), py::arg("activation"));
}

} // namespace eth::bind
