#include <pybind11/pybind11.h>
#include "cuda_bindings.hpp"
#include "tensor_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_etheria, m) {
    m.doc() = "Etheria Neural Network Library";

    // CUDA submodule
    py::module_ cuda_sub = m.def_submodule("cuda", "CUDA helpers");
    
    eth::bind::bind_cuda(cuda_sub);

    py::module_ tensor_sub = m.def_submodule("tensor", "Tensor operations");
    eth::bind::bindTensor(tensor_sub);

}
