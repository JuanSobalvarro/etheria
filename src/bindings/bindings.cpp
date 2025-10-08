#include <pybind11/pybind11.h>
#include "cuda_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_etheria, m) {
    m.doc() = "Etheria Neural Network Library";

    // CUDA submodule
    py::module_ cuda_sub = m.def_submodule("cuda", "CUDA helpers");
    eth::bindings::bind_cuda(cuda_sub);
}
