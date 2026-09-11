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
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleLoad);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleUnload);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleGetFunction);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleLaunchKernelEx);

// NOTE: no `mcFuncSetAttribute` / `cuFuncSetAttribute` binding here on
// purpose.  MACA does not need the maximum-dynamic-shared-memory attribute
// set before a launch (the launch itself enforces the 64 KiB
// `sharedMemPerBlockOptin` ceiling), and neither exported spelling is usable
// for a module-loaded kernel: the native `mcFuncSetAttribute(const void*,
// ...)` rejects a `mcFunction_t` with mcErrorInvalidDeviceFunction, and
// `cuFuncSetAttribute` lives in libsymbol_cu.so rather than the
// libmcruntime.so handle loaded here.  See `maca::Kernel::launch`.

// NOTE: the driver-side kernel *enumeration* used by the CUDA backend is
// genuinely absent -- `mcLibraryLoadFromFile`, `mcLibraryGetKernelCount`,
// `mcLibraryEnumerateKernels`, `mcKernelGetFunction`,
// `mcModuleGetFunctionCount` and `mcModuleEnumerateFunctions` are all
// missing from libmcruntime.so -- so the loaded artifact must report
// exactly one kernel and its name is discovered out-of-band by `llvm-nm` on
// the device artifact (see `maca::Kernel::load`).  (`mcLibraryLoadData` /
// `mcLibraryUnload` / `mcLibraryGetKernel` do exist, but `mcModuleLoad` is
// the path the host project's own kernel JIT uses, so the module-level API
// is kept.)
//
// NOTE: the CUDA backend also binds `cuTensorMapEncodeTiled` (see
// `cuda/driver.hpp`), which is how a consumer builds a TMA descriptor from
// the host.  No descriptor-encode entry point exists to bind here: scanning
// every `*.so*` in the MACA library directory finds no `TensorMap` symbol at
// all, neither under the `mc` spelling nor under the bridge's `cu*` one, so
// the TMA-descriptor facility has no MACA counterpart at this layer to
// document or expose.
//
// Launching goes through `mcModuleLaunchKernelEx`, which takes an
// `mcLaunchConfigExtension` (attribute list included) plus a
// `mcFunction_t` -- the module-level counterpart of `mcLaunchKernelExC`.
// `mcLaunchKernelExC` is exported too, but takes a host-side kernel *stub*
// pointer rather than a function handle, so it returns
// `mcErrorInvalidDeviceFunction` for a module-loaded kernel.

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
