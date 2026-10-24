#pragma once

namespace deep_jit {

// `ptr` already points at the argument value on the host, so the driver must
// use it directly instead of taking the wrapper's address.  It must never name
// a device buffer: the driver would read the value out of device memory.
struct NoRefPtr {
    void* ptr = nullptr;
};

}  // namespace deep_jit
