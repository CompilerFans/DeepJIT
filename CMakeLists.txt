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

# exception.hpp includes <elfutils/libdwfl.h>; libdw itself is dlopen'd at run
# time, so only the headers are needed here.
find_path(DEEP_JIT_LIBDW_INCLUDE_DIR
    NAMES elfutils/libdwfl.h
    HINTS ${DEEP_JIT_TOOLKIT_INCLUDE_DIRS} "${MACA_PATH}/mxgpu_llvm/include"
    DOC "Include directory containing elfutils/libdwfl.h")
if(NOT DEEP_JIT_LIBDW_INCLUDE_DIR)
    message(FATAL_ERROR
        "DeepJIT requires the elfutils/libdw headers: install libdw-dev "
        "(Debian/Ubuntu) or elfutils-devel (RHEL/Fedora), or set "
        "DEEP_JIT_LIBDW_INCLUDE_DIR.")
endif()

# TVM FFI ships its CMake config inside the Python package; the CLI probe does
# not import `tvm_ffi` (whose `__init__` imports torch).
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    OUTPUT_VARIABLE TVM_FFI_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
find_package(tvm_ffi CONFIG REQUIRED PATHS "${TVM_FFI_CMAKE_DIR}" NO_DEFAULT_PATH)

include_directories(
    include
    ${DEEP_JIT_FMT_ROOT}/include
    ${DEEP_JIT_LIBDW_INCLUDE_DIR}
    ${DEEP_JIT_TOOLKIT_INCLUDE_DIRS}
)
link_directories(${DEEP_JIT_TOOLKIT_LIBRARY_DIRS})

# Header-only consumer demo exporting `get_jit` through TVM FFI. The headers
# are torch-free, so the demo needs no Torch package.
add_library(_C MODULE csrc/python_api.cpp)
target_link_libraries(_C PRIVATE tvm_ffi::shared ${DEEP_JIT_TOOLKIT_LIBRARIES} ${CMAKE_DL_LIBS})
