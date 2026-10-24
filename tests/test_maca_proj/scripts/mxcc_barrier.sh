#!/bin/sh
# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
# Stand-in for `mxcc` that parks the JIT inside compilation so a test can kill
# it. `--version` passes through to keep the cache key identical.
real="${DEEP_JIT_MACA_TEST_REAL_MXCC:?the real mxcc must be named}"

case " $* " in
    *" --version "*) exec "$real" "$@";;
esac

if [ -n "${DEEP_JIT_MACA_TEST_MXCC_MARKER:-}" ]; then
    : > "$DEEP_JIT_MACA_TEST_MXCC_MARKER"
fi
while [ -n "${DEEP_JIT_MACA_TEST_MXCC_BLOCK:-}" ]; do
    sleep 0.05
done

exec "$real" "$@"
