#!/usr/bin/env bash
# Build the MACA test harness.  Device-side compilation is done by mxcc at run
# time; this only builds the host binary.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/tmp/djbuild/test_maca}"

MACA_PATH="${MACA_PATH:-/opt/maca}"
FMT_ROOT="${FMT_ROOT:-$ROOT/../fmt}"
TORCH="${TORCH_ROOT:-$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')}"

INCLUDES=(
    -I"$ROOT/include" -I"$FMT_ROOT/include"
    -I"$MACA_PATH/include" -I"$MACA_PATH/include/mcr" -I"$MACA_PATH/include/mcc"
    -I"$MACA_PATH/include/mcsparse" -I"$MACA_PATH/include/mcblas" -I"$MACA_PATH/include/mcsolver"
    -I"$MACA_PATH/include/misc" -I"$MACA_PATH/include/common" -I"$MACA_PATH/include/mctx"
    -I"$MACA_PATH/tools/cu-bridge/include"
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include"
)
# `-DUSE_MACA` is required by the platform's PyTorch headers.
COMMON=(-std=c++20 -O2 -DUSE_MACA -D_GNU_SOURCE -pthread "${INCLUDES[@]}"
        -L"$MACA_PATH/lib" -L"$TORCH/lib"
        -Wl,-rpath,"$TORCH/lib" -Wl,-rpath,"$MACA_PATH/lib")

mkdir -p "$(dirname "$OUT")"

# No MACA_TOOLCHAIN means the host compiler: the harness only needs C++20 and
# the MACA/torch headers, and GCC 11 is the toolchain mxcc itself drives.  The
# conda path below stays the canonical one when it is set.
if [[ -z "${MACA_TOOLCHAIN:-}" ]]; then
    CXX="${CXX:-g++}"
    for required in "$FMT_ROOT/include/fmt/format.h" "$MACA_PATH/include"; do
        if [[ ! -e "$required" ]]; then
            echo "build.sh: not found: $required" >&2
            echo "build.sh: set MACA_PATH/FMT_ROOT to the MACA install and the fmt checkout" >&2
            exit 1
        fi
    done
    "$CXX" "$ROOT/tests/test_maca_proj/main.cpp" -o "$OUT" "${COMMON[@]}" \
        -lmcruntime -lruntime_cu -lsymbol_cu -ldw -lc10 -lc10_cuda
    echo "built: $OUT"
    exit 0
fi

# Host toolchain (clang++ and a matching libstdc++/sysroot).
C="$MACA_TOOLCHAIN"
SYS="$C/x86_64-conda-linux-gnu/sysroot"
# Locate the GCC tree by version glob; a pinned path would fail silently.
G15="$(ls -d "$C"/lib/gcc/x86_64-conda-linux-gnu/*/ 2>/dev/null | tail -1)"
G15="${G15%/}"
# The sysroot's limits.h delegates to GCC's own, which lives beside the
# toolchain in the llvm tree rather than inside it.
GCC_LIMITS="${GCC_LIMITS_INCLUDE:-$(ls -d "$C"/../llvm/lib/gcc/*/*/include 2>/dev/null | tail -1)}"

for required in "$C/bin/clang++-22" "$G15/include" "$GCC_LIMITS/limits.h" \
                "$MACA_PATH/include" "$FMT_ROOT/include/fmt/format.h"; do
    if [[ ! -e "$required" ]]; then
        echo "build.sh: not found: $required" >&2
        echo "build.sh: set MACA_TOOLCHAIN (and GCC_LIMITS_INCLUDE) to a usable conda toolchain," >&2
        echo "build.sh: and MACA_PATH/FMT_ROOT to the MACA install and the fmt checkout" >&2
        exit 1
    fi
done

"$C/bin/clang++-22" "$ROOT/tests/test_maca_proj/main.cpp" -o "$OUT" \
  "${COMMON[@]}" -I"$C/include" -L"$C/lib" \
  --sysroot="$SYS" --gcc-toolchain="$C" \
  -isystem "$G15/include" -isystem "$G15/include/c++" \
  -isystem "$G15/include/c++/x86_64-conda-linux-gnu" -isystem "$G15/include/c++/backward" \
  -isystem "$SYS/usr/include" -isystem "$GCC_LIMITS" \
  -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -lmcruntime -lruntime_cu -lsymbol_cu -ldw

echo "built: $OUT"
