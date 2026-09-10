#include <pybind11/pybind11.h>

// Only the backend for the configured platform can be included: the CUDA
// backend asserts CUDA >= 12.4 and binds the `cuLibrary*` entry points,
// neither of which holds in a MACA build (the cu-bridge compatibility
// headers report CUDA 11.6 and expose no `cuLibrary*`).  CMake defines
// `DEEP_JIT_BACKEND_MACA` for `DEEP_JIT_PLATFORM=maca`.
#if defined(DEEP_JIT_BACKEND_MACA)
#include <deep_jit/backend/maca/backend.hpp>
#else
#include <deep_jit/backend/cuda/backend.hpp>
#endif

#include <deep_jit/python_api.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    // Consumer libraries register their configured runtime with `register_python_api`.
}
