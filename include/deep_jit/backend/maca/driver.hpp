// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#pragma once

#include <mc_runtime.h>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/lazy.hpp>

namespace deep_jit::maca::driver {

// The native module/launch entry points live in libmcruntime.so; the `cu*`
// spelling is a bridge in libruntime_cu.so.
DJ_DECL_LAZY_DL_HANDLE(get_maca_handle, "libmcruntime.so");

DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcGetErrorName);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcGetErrorString);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleLoad);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleUnload);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleGetFunction);
DJ_DECL_LAZY_DL_FUNCTION(get_maca_handle, mcModuleLaunchKernelEx);

// `mcModuleLaunchKernelEx` is the module-level entry; `mcLaunchKernelExC`
// rejects `mcFunction_t`. No attribute or TensorMap entry point exists.

inline void check_maca_driver(const mcError_t error, const char* expression) {
    if (error == mcSuccess)
        return;

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
