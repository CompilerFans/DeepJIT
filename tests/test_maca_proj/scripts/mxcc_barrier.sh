#!/bin/sh
# A stand-in for `mxcc` that parks the JIT inside the compiler step, so a test
# can kill a process at the moment its artifact is staged but not published.
# The MACA analogue of the CUDA driver's `nvcc_barrier.py`.
#
# `--version` is always passed through: the compiler version is part of the
# cache key, so a stand-in that answered it differently would have every
# process compile a different entry than the one it is supposed to inherit.
# With `DEEP_JIT_MACA_TEST_MXCC_BLOCK` unset the script is a plain pass-through,
# which is how the run that recovers after the kill uses it -- same compiler
# identity, same key, no block.
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
