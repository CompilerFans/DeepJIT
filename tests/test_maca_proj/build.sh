#!/usr/bin/env bash
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
# Build the MACA test harness.  Device-side compilation is done by mxcc at run
# time; this only builds the host binary.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-/tmp/djbuild/test_maca}"

MACA_PATH="${MACA_PATH:-/opt/maca}"
FMT_ROOT="${FMT_ROOT:-$ROOT/../fmt}"

# The MACA SDK ships no separate `clang++`: `mxcc` is the clang driver it does
# ship, and it compiles host C++ as readily as device code.  `CXX` overrides it.
CXX="${CXX:-$MACA_PATH/mxgpu_llvm/bin/mxcc}"

for required in "$FMT_ROOT/include/fmt/format.h" "$MACA_PATH/include" "$CXX"; do
    if [[ ! -e "$required" ]]; then
        echo "build.sh: not found: $required" >&2
        echo "build.sh: set MACA_PATH/FMT_ROOT to the MACA install and the fmt checkout," >&2
        echo "build.sh: and CXX to a host C++20 compiler" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUT")"

# The headers are torch-free, so no PyTorch include or link path is needed
# here: the check stays honest if a c10/ATen dependency is reintroduced.
# `-DUSE_MACA` is consulted by the cu-bridge headers. No `-pthread`: mxcc's
# option table lacks it, and this glibc links pthread into libc.
"$CXX" "$ROOT/tests/test_maca_proj/main.cpp" -o "$OUT" \
  -std=c++20 -O2 -DUSE_MACA -D_GNU_SOURCE \
  -I"$ROOT/include" -I"$FMT_ROOT/include" \
  -I"$MACA_PATH/include" -I"$MACA_PATH/include/mcr" -I"$MACA_PATH/include/mcc" \
  -I"$MACA_PATH/include/mcsparse" -I"$MACA_PATH/include/mcblas" -I"$MACA_PATH/include/mcsolver" \
  -I"$MACA_PATH/include/misc" -I"$MACA_PATH/include/common" -I"$MACA_PATH/include/mctx" \
  -I"$MACA_PATH/tools/cu-bridge/include" \
  -L"$MACA_PATH/lib" -Wl,-rpath,"$MACA_PATH/lib" \
  -lmcruntime -lruntime_cu -lsymbol_cu

echo "built: $OUT"
