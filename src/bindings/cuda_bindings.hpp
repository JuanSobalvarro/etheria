#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include "cuda/cuda_helper.cuh"

namespace py = pybind11;

namespace eth::bindings
{

void bind_cuda(py::module_ &m)
{
    // --- CUDA Helper ---
    m.def("is_cuda_available", &cuda::isCUDAAvailable, "Check if CUDA is available");
    m.def("number_cuda_devices", &cuda::numberCUDADevices, "Get the number of CUDA devices");
    m.def("is_cuda_compatible", &cuda::isCUDACompatible, "Check if a CUDA device is compatible");
    m.def("list_cuda_devices", &cuda::listCUDADevices, "List all available CUDA devices");
    m.def("current_cuda_device", &cuda::currentCUDADevice, "Get the current active CUDA device");
    m.def("device_details", &cuda::deviceDetails, "Get details of a CUDA device", py::arg("device_id"));
    m.def("set_cuda_device", &cuda::setDevice, "Set the active CUDA device", py::arg("device_id"));
}

} // namespace eth::bindings
