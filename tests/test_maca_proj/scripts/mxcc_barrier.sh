#!/bin/sh
# Stand-in for `mxcc` that parks the JIT inside the compiler step so a test
# can kill it before publication.  `--version` always passes through to keep
# the cache key identical; without the block variable it is a pass-through.
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
