// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <dlfcn.h>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit {

template <typename T>
class LazyInit {
    mutable std::mutex mutex;
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;

    T* ensure_initialized_locked() {
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
        if (ptr == nullptr)
            ptr = factory();
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
        return ptr.get();
    }

public:
    explicit LazyInit(std::nullptr_t) {}

    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}

    LazyInit(const LazyInit& other) {
        std::lock_guard lock(other.mutex);
        ptr = other.ptr;
        factory = other.factory;
    }

    LazyInit(LazyInit&& other) noexcept {
        std::lock_guard lock(other.mutex);
        ptr = std::move(other.ptr);
        factory = std::move(other.factory);
    }

    LazyInit& operator=(const LazyInit& other) {
        if (this != &other) {
            std::scoped_lock lock(mutex, other.mutex);
            ptr = other.ptr;
            factory = other.factory;
        }
        return *this;
    }

    LazyInit& operator=(LazyInit&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex, other.mutex);
            ptr = std::move(other.ptr);
            factory = std::move(other.factory);
        }
        return *this;
    }

    T* operator->() {
        std::lock_guard lock(mutex);
        return ensure_initialized_locked();
    }

    std::shared_ptr<T> get() {
        std::lock_guard lock(mutex);
        (void)ensure_initialized_locked();
        return ptr;
    }
};

}  // namespace deep_jit

#define DJ_DECL_LAZY_DL_HANDLE(handle_func_name, lib)                                         \
    inline void* handle_func_name() {                                                         \
        static void* handle = [] {                                                            \
            ::dlerror();                                                                      \
            void* value = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);                                \
            if (value == nullptr) {                                                           \
                const char* error = ::dlerror();                                              \
                DJ_PANIC("failed to load {}: {}", lib, error == nullptr ? "unknown" : error); \
            }                                                                                 \
            return value;                                                                     \
        }();                                                                                  \
        return handle;                                                                        \
    }

#define DJ_STRINGIFY(name) #name

#define DJ_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                                                   \
    template <typename... Args>                                                                                            \
    static auto lazy_##name(Args&&... args) {                                                                              \
        static const auto func = []() {                                                                                    \
            void* symbol = ::dlsym(handle_func_name(), DJ_STRINGIFY(name));                                                \
            if (symbol == nullptr) {                                                                                       \
                const char* error = ::dlerror();                                                                           \
                DJ_PANIC("failed to load {} from {}: {}", DJ_STRINGIFY(name),                                              \
                         #handle_func_name, error == nullptr ? "unknown" : error);                                         \
            }                                                                                                              \
            return reinterpret_cast<decltype(&name)>(symbol);                                                              \
        }();                                                                                                               \
        return func(std::forward<Args>(args)...);                                                                          \
    }
