#!/usr/bin/env bash
# Build the MACA DeepJIT test harness.
#
# The host side needs a C++20 standard library: DeepJIT's headers use
# std::format throughout, and MACA's mxcc is wired to the system GCC 11,
# whose libstdc++ has no <format>.  The toolchain used here is the conda
# clang++-22 + GCC-15 libstdc++ + matching sysroot that ships in this
# environment; the device-side compilation is still mxcc, invoked by the
# JIT at run time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/tmp/djbuild/test_maca}"

MACA_PATH="${MACA_PATH:-/opt/maca}"
C="${MACA_TOOLCHAIN:-/home/compiler_gfx/gpu_model/tools/hipcc}"
SYS="$C/x86_64-conda-linux-gnu/sysroot"
G15="$C/lib/gcc/x86_64-conda-linux-gnu/15.2.0"
# GCC's own limits.h (defines _GCC_LIMITS_H_ + include_next) is what the
# sysroot's limits.h delegates to; clang does not find it on its own.
GCC_LIMITS="${GCC_LIMITS_INCLUDE:-/home/compiler_gfx/gpu_model/tools/llvm/lib/gcc/x86_64-conda-linux-gnu/15.2.0/include}"
TORCH="${TORCH_ROOT:-$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')}"
PYTHON_INCLUDE="${PYTHON_INCLUDE:-$(python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')}"
PYTHON_LIB="${PYTHON_LIB:-$(python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')}"

mkdir -p "$(dirname "$OUT")"

"$C/bin/clang++-22" "$ROOT/tests/test_maca_proj/main.cpp" -o "$OUT" \
  -std=c++20 -O2 -DUSE_MACA -D_GNU_SOURCE \
  --sysroot="$SYS" --gcc-toolchain="$C" \
  -isystem "$G15/include" -isystem "$G15/include/c++" \
  -isystem "$G15/include/c++/x86_64-conda-linux-gnu" -isystem "$G15/include/c++/backward" \
  -isystem "$SYS/usr/include" -isystem "$GCC_LIMITS" \
  -I"$ROOT/include" \
  -I"$MACA_PATH/include" -I"$MACA_PATH/include/mcr" -I"$MACA_PATH/include/mcc" \
  -I"$MACA_PATH/include/mcsparse" -I"$MACA_PATH/include/mcblas" -I"$MACA_PATH/include/mcsolver" \
  -I"$MACA_PATH/include/misc" -I"$MACA_PATH/include/common" -I"$MACA_PATH/include/mctx" \
  -I"$MACA_PATH/tools/cu-bridge/include" -I"$C/include" \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYTHON_INCLUDE" \
  -L"$MACA_PATH/lib" -L"$TORCH/lib" -L"$C/lib" -L"$PYTHON_LIB" \
  -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$MACA_PATH/lib" -Wl,-rpath,"$PYTHON_LIB" \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lpython3.10 \
  -lmcruntime -lruntime_cu -lsymbol_cu -ldw

echo "built: $OUT"
