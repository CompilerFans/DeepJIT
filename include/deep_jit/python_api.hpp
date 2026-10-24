// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include <memory>
#include <utility>

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

#include <deep_jit/runtime/runtime.hpp>
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

}  // namespace deep_jit

// Export `get_jit` from the DSO owning `jit`, once per library at global
// scope. The diagnostic guard silences the generated wrapper's unused `self`.
#define DJ_EXPORT_GET_JIT(jit)                                            \
    _Pragma("GCC diagnostic push")                                        \
    _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")              \
    TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_jit, []() -> tvm::ffi::ObjectRef {  \
        return ::deep_jit::make_runtime_handle((jit).get());              \
    })                                                                    \
    _Pragma("GCC diagnostic pop")
