#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <mc_runtime.h>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit::maca {

// The xcore family identifiers: the keys of the per-family attribute table
// (deep_gemm.utils.arch_config) and the mxcc offload targets
// (xcore1000/1500/1600 = C500/C600/C600U).
enum class XcoreFamily : int {
    xcore1000 = 1000,
    xcore1500 = 1500,
    xcore1600 = 1600,
};

inline void check_maca_runtime(const mcError_t error, const char* expression) {
    if (error != mcSuccess) {
        DJ_PANIC("{} failed with MACA error {} ({}): {}",
                 expression,
                 static_cast<int>(error),
                 mcGetErrorName(error),
                 mcGetErrorString(error));
    }
}

#ifndef DJ_MACA_RUNTIME_CHECK
#define DJ_MACA_RUNTIME_CHECK(expr) ::deep_jit::maca::check_maca_runtime((expr), #expr)
#endif

class Device {
    mcDeviceProp_t prop{};
    int64_t clock_rate = 0;
    bool initialized = false;

public:
    const mcDeviceProp_t& get_prop() {
        if (not initialized) {
            // `mcFree(nullptr)` is to ensure the current MACA context exists
            // before later driver API calls.
            int device_index = 0;
            DJ_MACA_RUNTIME_CHECK(mcGetDevice(&device_index));
            DJ_MACA_RUNTIME_CHECK(mcFree(nullptr));
            DJ_MACA_RUNTIME_CHECK(mcGetDeviceProperties(&prop, device_index));
            initialized = true;
        }
        return prop;
    }

    int get_num_sms() { return get_prop().multiProcessorCount; }

    int get_num_l2_cache_bytes() { return get_prop().l2CacheSize; }

    int get_num_smem_bytes() { return static_cast<int>(get_prop().sharedMemPerBlockOptin); }

    int64_t get_clock_rate() {
        if (clock_rate == 0) {
            int device_index = 0;
            int rate = 0;
            DJ_MACA_RUNTIME_CHECK(mcGetDevice(&device_index));
            DJ_MACA_RUNTIME_CHECK(mcDeviceGetAttribute(&rate, mcDeviceAttributeClockRate, device_index));
            clock_rate = static_cast<int64_t>(rate) * 1000;
        }
        return clock_rate;
    }

    int get_arch_major() { return get_prop().major; }

    int get_arch_minor() { return get_prop().minor; }

    std::pair<int, int> get_arch_pair() { return {get_arch_major(), get_arch_minor()}; }

    // The device's xcore family.  The mc native arch major IS the family's
    // leading digits (mc major 10/15/16 -> xcore1000/1500/1600); the minor
    // is a revision inside the family and is not a judgment axis.  An
    // unknown major is a tripwire, not a speculative mapping.
    XcoreFamily get_family() {
        switch (get_arch_major()) {
            case 10: return XcoreFamily::xcore1000;
            case 15: return XcoreFamily::xcore1500;
            case 16: return XcoreFamily::xcore1600;
            default:
                DJ_PANIC("unsupported MACA architecture: mc major {}", get_arch_major());
        }
    }

    // The mxcc offload target for the judged family -- the value that goes
    // into `--offload-arch=`.  The family digits are the family identity
    // itself; never derive them by arithmetic from the mc major.
    std::string get_arch() { return std::to_string(static_cast<int>(get_family())); }
};

}  // namespace deep_jit::maca
