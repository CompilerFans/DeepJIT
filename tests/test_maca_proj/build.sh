#!/usr/bin/env bash
# Build the MACA DeepJIT test harness.
#
# DeepJIT's headers format through fmt (see include/deep_jit/utils/format.hpp):
# MACA's mxcc is wired to the system GCC 11, whose libstdc++ has no <format>,
# so <format> cannot be assumed anywhere DeepJIT gets embedded.  The toolchain
# used here is the conda clang++-22 + GCC-15 libstdc++ + matching sysroot that
# ships in this environment; the device-side compilation is still mxcc, invoked
# by the JIT at run time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/tmp/djbuild/test_maca}"

MACA_PATH="${MACA_PATH:-/opt/maca}"
# Overridable from the environment, like everything else here: the in-image
# toolchain is only a default, and `tests/test_maca.py` resolves the same root
# from MACA_TOOLCHAIN / MACA_HOST_CXX / CXX.  Keep the two in step.
C="${MACA_TOOLCHAIN:-/home/compiler_gfx/gpu_model/tools/hipcc}"
# fmt is vendored as a sibling checkout by the consuming repositories.
FMT_ROOT="${FMT_ROOT:-$ROOT/../fmt}"
SYS="$C/x86_64-conda-linux-gnu/sysroot"
# The conda toolchain's GCC tree, by version glob rather than by a pinned one.
G15="$(ls -d "$C"/lib/gcc/x86_64-conda-linux-gnu/*/ 2>/dev/null | tail -1)"
G15="${G15%/}"
# GCC's own limits.h (defines _GCC_LIMITS_H_ + include_next) is what the
# sysroot's limits.h delegates to; clang does not find it on its own.  It ships
# in the `llvm` tree BESIDE the toolchain, not inside it -- the toolchain has
# the C++ headers but no limits.h of its own -- so it is derived from `$C`
# rather than pinned, and located by version glob too.
GCC_LIMITS="${GCC_LIMITS_INCLUDE:-$(ls -d "$C"/../llvm/lib/gcc/*/*/include 2>/dev/null | tail -1)}"
TORCH="${TORCH_ROOT:-$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')}"
PYTHON_INCLUDE="${PYTHON_INCLUDE:-$(python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')}"
PYTHON_LIB="${PYTHON_LIB:-$(python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')}"
# The interpreter's ABI version, for `-lpython<X.Y>`, rather than a literal.
PYTHON_ABI="${PYTHON_ABI:-$(python -c 'import sysconfig;print(sysconfig.get_config_var("LDVERSION"))')}"

for required in "$C/bin/clang++-22" "$G15/include" "$GCC_LIMITS/limits.h" \
                "$MACA_PATH/include" "$FMT_ROOT/include/fmt/format.h"; do
    if [[ ! -e "$required" ]]; then
        echo "build.sh: not found: $required" >&2
        echo "build.sh: set MACA_TOOLCHAIN (and GCC_LIMITS_INCLUDE) to a usable conda toolchain," >&2
        echo "build.sh: and MACA_PATH/FMT_ROOT to the MACA install and the fmt checkout" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUT")"

# `-DUSE_MACA` is NOT optional scaffolding: the platform's PyTorch headers gate
# on it (`ATen/Context.h` has a `static_assert(0)` in the `#else` branch), and
# the backend's own `maca/kernel.hpp` includes `<ATen/cuda/CUDAContext.h>` for
# the current stream.  Dropping it fails the build inside ATen, not here.
"$C/bin/clang++-22" "$ROOT/tests/test_maca_proj/main.cpp" -o "$OUT" \
  -std=c++20 -O2 -DUSE_MACA -D_GNU_SOURCE \
  --sysroot="$SYS" --gcc-toolchain="$C" \
  -isystem "$G15/include" -isystem "$G15/include/c++" \
  -isystem "$G15/include/c++/x86_64-conda-linux-gnu" -isystem "$G15/include/c++/backward" \
  -isystem "$SYS/usr/include" -isystem "$GCC_LIMITS" \
  -I"$ROOT/include" -I"$FMT_ROOT/include" \
  -I"$MACA_PATH/include" -I"$MACA_PATH/include/mcr" -I"$MACA_PATH/include/mcc" \
  -I"$MACA_PATH/include/mcsparse" -I"$MACA_PATH/include/mcblas" -I"$MACA_PATH/include/mcsolver" \
  -I"$MACA_PATH/include/misc" -I"$MACA_PATH/include/common" -I"$MACA_PATH/include/mctx" \
  -I"$MACA_PATH/tools/cu-bridge/include" -I"$C/include" \
  -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$PYTHON_INCLUDE" \
  -L"$MACA_PATH/lib" -L"$TORCH/lib" -L"$C/lib" -L"$PYTHON_LIB" \
  -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$MACA_PATH/lib" -Wl,-rpath,"$PYTHON_LIB" \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lpython"$PYTHON_ABI" \
  -lmcruntime -lruntime_cu -lsymbol_cu -ldw

echo "built: $OUT"
