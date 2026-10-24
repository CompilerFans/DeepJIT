#pragma once

#include <memory>
#include <utility>

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/lazy.hpp>

namespace deep_jit {

// Opaque Python-visible handle: pins the runtime, exposes no fields or methods.
template <typename Backend>
class RuntimeHandle final : public tvm::ffi::Object {
public:
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL(Backend::runtime_type_key, RuntimeHandle, tvm::ffi::Object);

    explicit RuntimeHandle(std::shared_ptr<Runtime<Backend>> runtime)
        : runtime(std::move(runtime)) {}

    std::shared_ptr<Runtime<Backend>> runtime;
};

template <typename Backend>
inline tvm::ffi::ObjectRef make_runtime_handle(std::shared_ptr<Runtime<Backend>> runtime) {
    return tvm::ffi::ObjectRef(tvm::ffi::make_object<RuntimeHandle<Backend>>(std::move(runtime)));
}

// For C++ harnesses: get the runtime back out of a get_jit() result.
template <typename Backend>
inline std::shared_ptr<Runtime<Backend>> unwrap_runtime_handle(const tvm::ffi::Any& value) {
    const auto* handle = value.as<RuntimeHandle<Backend>>();
    DJ_HOST_ASSERT(handle != nullptr, "get_jit did not return a deep_jit runtime handle");
    return handle->runtime;
}

}  // namespace deep_jit

// Export `get_jit` from the DSO that owns `jit`, a namespace-scope LazyInit.
// Invoke exactly once per library, at global scope (outside any namespace).
// The wrapper tvm-ffi generates leaves its `self` parameter unused, so the
// expansion is wrapped in a diagnostic guard for consumers that build with
// -Werror without treating tvm-ffi as a system include.
#define DJ_EXPORT_GET_JIT(jit)                                            \
    _Pragma("GCC diagnostic push")                                        \
    _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")              \
    TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_jit, []() -> tvm::ffi::ObjectRef {  \
        return ::deep_jit::make_runtime_handle((jit).get());              \
    })                                                                    \
    _Pragma("GCC diagnostic pop")
