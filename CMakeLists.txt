# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# NOTES: this CMake is only for debugging/IDE indexing; consumers integrate by include path.
cmake_minimum_required(VERSION 3.20)
project(deep_jit LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CMAKE_CXX_STANDARD 20)

# The MACA toolkit provides mxcc and libmcruntime; CUDA headers resolve through
# the cu-bridge compatibility root.
set(MACA_PATH "$ENV{MACA_PATH}")
if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca")
endif()
set(DEEP_JIT_TOOLKIT_INCLUDE_DIRS
    "${MACA_PATH}/include"
    "${MACA_PATH}/tools/cu-bridge/include")
set(DEEP_JIT_TOOLKIT_LIBRARY_DIRS "${MACA_PATH}/lib")
set(DEEP_JIT_TOOLKIT_LIBRARIES mcruntime)

# fmt is header-only: `<format>` is unavailable on some host toolchains behind
# mxcc. Defaults to a sibling checkout; see utils/format.hpp.
set(DEEP_JIT_FMT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../fmt" CACHE PATH "Root of an fmt checkout")

# TVM FFI ships its CMake config inside the Python package; the CLI probe does
# not import `tvm_ffi` (whose `__init__` imports torch).
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    OUTPUT_VARIABLE TVM_FFI_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(tvm_ffi CONFIG REQUIRED PATHS "${TVM_FFI_CMAKE_DIR}" NO_DEFAULT_PATH)
find_package(Torch REQUIRED)

include_directories(
    include
    ${DEEP_JIT_FMT_ROOT}/include
    ${DEEP_JIT_TOOLKIT_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
link_directories(${TORCH_INSTALL_PREFIX}/lib ${DEEP_JIT_TOOLKIT_LIBRARY_DIRS})

# Header-only consumer demo exporting `get_jit` through TVM FFI.
add_library(_C MODULE csrc/python_api.cpp)
target_link_libraries(_C PRIVATE tvm_ffi::shared ${TORCH_LIBRARIES} ${DEEP_JIT_TOOLKIT_LIBRARIES})
