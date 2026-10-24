# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DeepJIT is a **header-only C++20 JIT runtime** for MACA GPUs. It compiles kernel source at runtime
with mxcc, caches the artifact in memory and on disk, loads it, and launches it. Consumer libraries
embed it by adding `include/` to their include path — there is no library to link (fmt is
header-only too); only the optional `python_api.hpp` needs TVM FFI.

This checkout is consumed as the `third-party/DeepJIT` git submodule of DeepSelect (a MACA top-K
kernel library); DeepJIT is its own repository, with `main` as the branch, and commits made here
need the consuming repository's submodule pointer bumped separately. fmt is the sibling directory
`third-party/fmt` in the consuming checkout — a plain vendored source tree (12.1.0, not a
submodule) — and is what `FMT_ROOT`/`DEEP_JIT_FMT_ROOT` default to.

`README.md` is the authoritative API reference: options, cache-key composition, toolchain
requirements, and the full environment-variable table. Read the relevant section before changing
the compile/load path.

## Commands

```bash
# Full platform test: builds the C++ harness, runs it, validates the cache artifacts on disk.
python tests/test_maca.py          # MACA: source the MACA env first (MACA_PATH, LD_LIBRARY_PATH
                                   #   must include $MACA_PATH/lib and $MACA_PATH/mxgpu_llvm/lib)
python tests/test_exception.py     # header-only check of the exception/backtrace path

# Build just the MACA harness with the MACA SDK's own clang driver (`mxcc`); `CXX` overrides it.
# The SDK ships no `clang++`.  Device-side compilation is still done by mxcc later, at run time.
bash tests/test_maca_proj/build.sh /tmp/djbuild/test_maca

# CMake is for debugging / IDE indexing only. It builds a TVM FFI `_C` module and exists to
# exercise the header-only integration; consumers integrate via include path, not via this target.
cmake -S . -B build
```

Tests are plain `assert`-based Python; run them without `-O` (`python -O` trips the guard
`if not __debug__: raise`). `test_maca.py` starts the multi-process writers in their own sessions so
that a leaked device process can be killed as a process group.

### Running one case

There is **no test filter** in the C++ harness (`run_test(...)` calls in `main()` are
unconditional). To iterate on a single case, run the built harness directly with its env, or
comment out the other `run_test` lines in `main()`:

```bash
DEEP_JIT_MACA_TEST_SOURCE_DIR=$PWD/tests/test_maca_proj \
DEEP_JIT_MACA_TEST_CACHE_DIR=/tmp/dj_cache \
LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH \
/tmp/djbuild/test_maca
```

The MACA harness also takes `--diagnostics` (one compile/load with the reports enabled),
`--compile-once <tag> <bias> <ready> <start>` (one compile behind a file barrier), and
`--publish-once <cache-root> <owner> <ready> <start>` (a direct `DiskCache` commit behind the same
kind of barrier), which the Python driver uses for the diagnostic, race, and atomic-publication
cases.

## Architecture

### One runtime, one backend

`deep_jit::Runtime<Backend>` (`include/deep_jit/runtime/runtime.hpp`) is the whole public surface,
and it is **backend-agnostic**: it owns `Config`, `Env`, `Device`, `DiskCache`, `Parser`, the memory
cache, and the backend object, and drives the shared compile/load/launch flow. The backend supplies
four nested types — `Device`, `Kernel`, `CompilerOptions`, `LaunchOptions` — plus
`compile(source, dir, env, config, options)` and `static load(dir, env)`, and a
`runtime_type_key` for the Python handle.

### The flow

```
compile(tag, source, opts)
  └─ effective options  = default_compiler_options.override_with(opts)      # unset fields inherit
  └─ cache_key          = FNV1a(extra_signature, compiler --version,
                                effective flags, post_hook path+content, parser digest)
  └─ mem_cache.get_or_create(key):
       ├─ DiskCache::entry(tag, key)   # <root>/cache/<tag>.<key>/ with a .committed marker
       │    └─ hit → load that directory
       ├─ miss → backend.compile(...) into <root>/tmp/<uuid>/
       │    └─ writes kernel.cu, kernel.devbin, optional dumps, meta.json
       └─ entry.commit()   # fsync tree, atomic directory rename, .committed marker
            └─ backend.load(dir, env) → shared_ptr<Kernel>, process-cached
launch(kernel, opts, args...)  →  kernel->launch(default_launch_options.override_with(opts), args...)
```

Key consequences:

- **The compile tag is not part of the cache key** — it only names the entry. Different tags over the
  same source and options share a digest; the tag must be letters/digits/underscore only.
- **`include_dirs` are deliberately excluded from the digest** (moving the same header contents must
  not invalidate); any `-I` you pass through `mxcc_flags` *does* count. `extra_signature` is where
  untracked dependencies (e.g. a CUTLASS version) belong.
- **`compile()` requires exactly one kernel** in the artifact; `compile_without_load()` skips that
  check and returns the artifact directory.
- Configuration is snapshotted at `Runtime` construction. `config`, `env`, `backend`, `parser`,
  `disk_cache`, `hash_base` are not to be mutated afterwards; only `default_compiler_options` and
  `default_launch_options` are supported mutable knobs.
- Disk caching is process- and node-shared: builds happen in unique temp dirs and are published by
  atomic rename, so concurrent processes racing the same key all end up at the same entry (the loser
  deletes its own temp dir — see `safe_remove_all`, deliberately not `std::filesystem::remove_all`).
  `JIT_CACHE_DIR` may be a colon-separated list: all roots are searched, misses are written to the
  first (a read-only shared lookup root behind a writable personal root).

### Include tracking (`utils/parser.hpp`)

A line-oriented scanner, **not a preprocessor**. An include is tracked only when it is a literal
`#include <...>` *and* the filename starts with one of `Config::include_prefixes`. Quoted includes are
rejected outright; macro-expanded includes, comments between tokens, and `#if 0` blocks are not
interpreted (a tracked include inside `#if 0` is still hashed). Tracked files are resolved against
`include_dirs` in order and hashed recursively; a missing tracked file is an error and cycles are
rejected. Header hashes are cached for the runtime's lifetime — recreate the runtime after editing a
tracked header.

### Environment

Every setting resolves `<PREFIX>_<SUFFIX> > DJ_<SUFFIX> > built-in default`, where `PREFIX` is
`Config::env_prefix` (`DJ` is reserved and cannot be used as a library prefix). Unprefixed names have
no effect. Suffixes (`JIT_CACHE_DIR`, `JIT_DEBUG`, `JIT_DUMP_ASM`, `JIT_CHECK_NO_SPILLS`, …) are
tabulated in README; `utils/env.hpp` implements the lookup. Snapshotting happens at runtime
construction, so set these before the first `get_jit()`/`jit->...`.

### MACA specifics worth knowing

**MACA** (`backend/maca/`) — compiles with `mxcc -x maca -device-bin` to `kernel.devbin`.
`-device-bin` is the only artifact form that serves both the load path and name discovery: MACA has
no driver-side kernel enumeration (`mcModuleGetFunctionCount` et al. are absent from
`libmcruntime.so`), so the entry point name is recovered from the binary with the toolchain's
`llvm-nm` (`Kernel::parse_kernel_names`). The driver entry points are bound lazily with
`dlopen`/`dlsym` on `libmcruntime.so` (`maca/driver.hpp`) — the `cu*` spellings are a different
library and are not used.

- Launch goes through `mcModuleLaunchKernelEx`. **Cluster launch and PDL are rejected up front**, and
  only the cooperative flag is passed as an attribute: the MACA runtime refuses every other launch
  attribute with `mcErrorInvalidConfiguration`. There is no dynamic-smem attribute call either — the
  launch itself enforces the `sharedMemPerBlockOptin` ceiling. `maca/kernel.hpp` and `maca/driver.hpp`
  carry the measured evidence; don't "fix" those paths back toward the CUDA shape.
- `check_no_spills` and `check_no_local_memory` coincide: mxcc reports a single "N bytes stack frame"
  figure and emits it only under `-resource-usage`, so either check implies that flag.
- Architecture: device major 10/15/16 → `xcore1000`/`xcore1500`/`xcore1600` (C500/C600/C600U).
  `Device::get_arch()` returns the family digits, and options render `--offload-arch=xcore<digits>`.
  An unknown major is a panic, not a guess.
- `MACA::load` is static but resolves the toolkit itself from the same `env` the `Runtime` hands it —
  MACA's load needs `llvm-nm` out of the toolkit, so unlike a pure driver load it cannot ignore the
  toolkit, and a process-global would let one `Runtime`'s toolkit decide another's loads (and race
  between two constructors).

### Python integration (TVM FFI)

`python_api.hpp` exports a single `get_jit()` through TVM FFI. It defines
`RuntimeHandle<Backend>` (a `tvm::ffi::Object` wrapping `std::shared_ptr<Runtime<Backend>>`) and
`RuntimeHandle`'s type key comes from `Backend::runtime_type_key`; `make_object` registers it lazily,
so no static-init block is needed in a header-only library. The `DJ_EXPORT_GET_JIT(jit)` macro wraps
a namespace-scope `LazyInit` and must be invoked at global scope in **exactly one translation unit
per shared library**; it always exports the name `get_jit`. Python gets an opaque handle:

```python
import tvm_ffi
jit = tvm_ffi.load_module("/path/to/library.so").get_jit()
```

The handle has no fields or methods, and each call returns a new handle over the same runtime. TVM
FFI releases the GIL around every Python-to-C++ call by default; the headers themselves are
Python-free and do not manage the GIL. `csrc/python_api.cpp` is the CMake-only demo consumer;
`deep_jit/__init__.py` is intentionally empty — the library is header-only and the Python package
exists for packaging namespace only.

## Conventions and traps

- **Format through fmt, never `std::format`, in `include/deep_jit/**`.** `<format>` predates some host
  toolchains DeepJIT gets embedded into (mxcc drives a GCC 11 libstdc++), and a header-only library's
  stdlib floor becomes its consumer's. `utils/format.hpp` defines `FMT_HEADER_ONLY` and pulls
  `<fmt/format.h>`; the consumer supplies the include path. The test harnesses under `tests/` format
  through fmt as well (they go through `deep_jit/utils/format.hpp`), so the whole tree builds with
  the host toolchain's C++20 — including the GCC 11 behind `mxcc`. The two libraries are not
  interchangeable for `{:a}`: fmt prints printf's `0x` prefix, while `std::format` follows
  `to_chars` and omits it.
- **Kernel arguments**: pass raw values (the driver takes their address) or wrap a pointer to
  host-side storage in `deep_jit::NoRefPtr{ptr}` when that address already holds the argument value
  and the wrapper must not be addressed itself. `kernel_arg_pointer` in `maca/kernel.hpp` implements
  the unwrap.
- **Errors** are `DJ_HOST_ASSERT` / `DJ_PANIC` (`utils/exception.hpp`, throws
  `deep_jit::exception::Exception`; backtrace via lazily-loaded `libdw`), and `DJ_MACA_DRIVER_CHECK` /
  `DJ_MACA_RUNTIME_CHECK`. They surface as `RuntimeError` in Python with the original `what()`
  message when called through TVM FFI.
- **Compiler, cache, include and hook paths and free-form flags go through a shell command**, so they
  must not contain whitespace or shell metacharacters.
- A **post hook** (`CompilerOptions::post_hook`) is a trusted relative path under
  `Config::python_library_root`, run after the compiler produces the artifact and before publication,
  with the artifact path as argv[1]; the hook must edit it in place and stay deterministic (its path
  and content hash are in the digest, and hashes are cached per thread — restart after editing).
- `Runtime` snapshots the current device at construction: use one runtime per device, construct and
  use it while that device is current, and don't move a loaded kernel across contexts.
- The two-state FNV-1a digest (`utils/hash.hpp`) length-prefixes every update so chained binary inputs
  have unambiguous boundaries; it is a cache checksum, **not** a cryptographic hash.

## Testing architecture

`tests/test_maca.py` builds and drives a standalone C++ harness in `tests/test_maca_proj/`. The
harness registers its cases with `run_test("name", ...)` and prints `[ RUN ] / [ OK ] / [ SKIPPED ]`.

- MACA has no Python embedding: `main.cpp` is a standalone host binary built by
  `test_maca_proj/build.sh` with the MACA SDK's `mxcc` (its clang driver; the SDK ships no
  `clang++`), and `tests/test_maca.py` uses the same compiler for the self-containment check;
  kernel sources live in `test_maca_proj/kernels/` and hooks in `test_maca_proj/scripts/`.
- The driver validates more than the harness reports: **header self-containment** (each
  `backend/maca/*.hpp` header must compile alone with `-Werror`, to an object file because
  `-fsyntax-only` is not in mxcc's option table), the exact cache-entry shape on disk
  (`kernel.cu`, `kernel.devbin`, `meta.json`, `.committed`, allowed dumps), the `meta.json` schema,
  multi-process races onto one key, diagnostic output under each env prefix, and the crashed-writer
  path.
- Failure-mode kernels are deliberate and must stay: e.g. `local_frame.cu`, `multiple_kernels.cu`,
  `no_kernel.cu`, `include_cycle/`, `include_same_content/` vs `include_changed_content/`. When
  adding an option or a cache-key input, extend the corresponding
  `*_original/ / *_same_content/ / *_changed_content/` triplet so the digest is proven to track it. The MACA triplet pins "same contents at a different path must not change the
  digest" and "changed contents must", and the untracked `third_party/` dependency pins the
  `extra_signature` path.
- `tests/test_exception.py` builds the exception/backtrace path as a TVM FFI module with
  `is_python_module=False, no_implicit_headers=True` and loads it with `tvm_ffi.load_module`; it
  asserts the exact `RuntimeError` message and demangled frames.
- Rebuild before believing a result. Kernel sources are read at *run* time (the runtime compiles
  `kernels/*.cu` on demand), but the harness's own assertions are baked into the binary — so a
  kernel-only edit needs no rebuild, while a `main.cpp` edit without one silently runs the *old*
  expectations against the *new* kernel and surfaces as a spurious mismatch in whatever the two
  compute jointly. `python tests/test_maca.py` always builds first; a hand-run of
  `/tmp/djbuild/test_maca` does not.
- Test case names are `<subject> <aspect>` with spaces (`invalid cache tag`, `secondary disk cache`,
  `compiler failure cleanup`, `macro-selected kernel ABI`, …). Keep new cases on that pattern.
