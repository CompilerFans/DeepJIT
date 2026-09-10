#pragma once

#include <mc_runtime.h>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/lazy.hpp>

namespace deep_jit::maca::driver {

// MACA keeps the module/launch entry points in the runtime library: the
// native `mc*` spelling lives in `libmcruntime.so` (the CUDA-namespace
// `cu*` spelling is a bridge that `libruntime_cu.so` rewrites to `wcu*`,
// which is not what the rest of this project links against).  We bind the
// `mc*` names directly so the backend depends on exactly the library the
// host project's own kernel JIT uses.
DJ_DECL_LAZY_DL_HANDLE(get_maca_handle, "libmcruntime.so");

DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcGetErrorName);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcGetErrorString);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcFuncSetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleLoad);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleUnload);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleGetFunction);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleLaunchKernel);

// NOTE: MACA has no `mcLibrary*` and no module-level function enumeration
// (`mcModuleGetFunctionCount` / `mcModuleEnumerateFunctions` do not exist),
// so the loaded artifact must report exactly one kernel and its name is
// discovered out-of-band by `llvm-nm` on the device artifact -- see
// `maca::Kernel::load`.  The launch path likewise uses the exported
// `mcModuleLaunchKernel` rather than a `*LaunchKernelEx` variant: only
// `mcLaunchKernelExC` exists and it is not an exported symbol, so the
// attribute-based launch config is flattened onto the classic entry point
// (see `maca/kernel.hpp`).

inline void check_maca_driver(const mcError_t error, const char* expression) {
    if (error == mcSuccess)
        return;

    // NOTE: unlike CUDA's `cuGetErrorName(error, &name)` out-parameter form,
    // the MACA entry points return the string directly.
    const char* error_name = lazy_mcGetErrorName(error);
    const char* error_description = lazy_mcGetErrorString(error);
    DJ_PANIC("{} failed with MACA error {} ({}): {}",
             expression,
             static_cast<int>(error),
             error_name == nullptr ? "unknown" : error_name,
             error_description == nullptr ? "unknown" : error_description);
}

#ifndef DJ_MACA_DRIVER_CHECK
#define DJ_MACA_DRIVER_CHECK(expr) ::deep_jit::maca::driver::check_maca_driver((expr), #expr)
#endif

}  // namespace deep_jit::maca::driver
