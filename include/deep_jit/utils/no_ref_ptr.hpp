#pragma once

namespace deep_jit {

// Wrap an argument so that the driver takes `ptr` itself as the location of
// the argument value instead of taking the address of the wrapper.  `ptr` must
// therefore point at host memory holding the argument, never at the device
// buffer the argument happens to name: passing a device address makes the
// driver read the argument value out of device memory and faults.
struct NoRefPtr {
    void* ptr = nullptr;
};

}  // namespace deep_jit
