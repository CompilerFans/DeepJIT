// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <filesystem>

#include <tvm/ffi/function.h>

#include <deep_jit/backend/maca/backend.hpp>
#include <deep_jit/python_api.hpp>

namespace {

auto python_api_jit = deep_jit::create_lazy_jit<deep_jit::MACA>(
    deep_jit::Config(std::filesystem::current_path(), "DEEP_JIT_DEMO"));

}  // namespace

DJ_EXPORT_GET_JIT(python_api_jit);
