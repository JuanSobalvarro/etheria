#include "algebra/ops.hpp"
#include "algebra/ops_cpu.hpp"
#include "algebra/ops_cuda.hpp"
#include <stdexcept>


namespace eth::algebra
{

Vector vectoradd(const Vector& a, const Vector& b, Backend backend)
{
    if (backend == Backend::CPU)
        return cpu::vectoradd(a, b);
    else if (backend == Backend::CUDA)
        throw std::runtime_error("vectoradd: CUDA backend not implemented");
    else
        throw std::runtime_error("vectoradd: Unknown backend");
}

}