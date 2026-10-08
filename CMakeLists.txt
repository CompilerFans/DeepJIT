# NOTES: this CMake is only for debugging/IDE indexing; for setup, please use Python extension
cmake_minimum_required(VERSION 3.20)
project(deep_jit LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CMAKE_CXX_STANDARD 20)

# The toolkit is selected by the target platform.  MACA replaces the CUDA
# toolkit with the MXMACA install (mxcc + libmcruntime), so that path must
# not require CUDAToolkit.
set(DEEP_JIT_PLATFORM "cuda" CACHE STRING "Target platform: cuda or maca")
set_property(CACHE DEEP_JIT_PLATFORM PROPERTY STRINGS cuda maca)

if(DEEP_JIT_PLATFORM STREQUAL "maca")
    set(MACA_PATH "$ENV{MACA_PATH}")
    if(NOT MACA_PATH)
        set(MACA_PATH "/opt/maca")
    endif()
    set(DEEP_JIT_TOOLKIT_INCLUDE_DIRS "${MACA_PATH}/include")
    set(DEEP_JIT_TOOLKIT_LIBRARY_DIRS "${MACA_PATH}/lib")
    set(DEEP_JIT_TOOLKIT_LIBRARIES mcruntime)
    # CUDA headers still resolve through the cu-bridge compatibility root.
    list(APPEND DEEP_JIT_TOOLKIT_INCLUDE_DIRS "${MACA_PATH}/tools/cu-bridge/include")
else()
    find_package(CUDAToolkit REQUIRED)
    set(DEEP_JIT_TOOLKIT_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
    set(DEEP_JIT_TOOLKIT_LIBRARY_DIRS
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs")
    set(DEEP_JIT_TOOLKIT_LIBRARIES cuda)
endif()

# fmt supplies the formatting API.  `<format>` is not available on every host
# toolchain DeepJIT is embedded into (MACA's mxcc drives the system GCC 11,
# whose libstdc++ predates it), so the headers go through fmt instead -- see
# include/deep_jit/utils/format.hpp.  fmt is used header-only, so only the
# include path is needed.  Defaults to a sibling checkout, which is how the
# consuming repositories vendor both.
set(DEEP_JIT_FMT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../fmt" CACHE PATH "Root of an fmt checkout")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    include
    ${DEEP_JIT_FMT_ROOT}/include
    ${DEEP_JIT_TOOLKIT_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
)
link_directories(${TORCH_INSTALL_PREFIX}/lib ${DEEP_JIT_TOOLKIT_LIBRARY_DIRS})

pybind11_add_module(_C csrc/python_api.cpp)
# The CUDA and MACA backends cannot coexist in one translation unit (the
# CUDA driver header asserts CUDA >= 12.4 and binds `cuLibrary*`, which the
# cu-bridge compatibility headers do not provide), so the backend is picked
# at configure time.
if(DEEP_JIT_PLATFORM STREQUAL "maca")
    target_compile_definitions(_C PRIVATE DEEP_JIT_BACKEND_MACA)
else()
    # CUDA targets PyTorch's stable ABI with PyTorch 2.10 as the minimum runtime.
    target_compile_definitions(_C PRIVATE
        TORCH_TARGET_VERSION=0x020a000000000000
        USE_CUDA
    )
endif()
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python ${DEEP_JIT_TOOLKIT_LIBRARIES})
target_link_options(_C PRIVATE "-Wl,-Bsymbolic-functions")
