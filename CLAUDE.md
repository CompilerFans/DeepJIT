# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DeepJIT is a **header-only C++20 JIT runtime** for CUDA GPUs, MACA GPUs, and Ascend NPUs. It compiles
kernel source at runtime with the platform compiler, caches the artifact in memory and on disk, loads
it, and launches it. Consumer libraries embed it by adding `include/` to their include path — there is
no library to link (fmt is header-only too).

This checkout is the `third-party/DeepJIT` git submodule of mcDeepGEMM (xcore C500/C600/C600U MACA
project); it is its own repository (`git@github.com:CompilerFans/DeepJIT.git`, branch `main`). Commits
made here need the parent repo's submodule pointer bumped separately. fmt is the sibling submodule
`third-party/fmt` (pinned at 12.1.0) — that is what `FMT_ROOT`/`DEEP_JIT_FMT_ROOT` default to.

`README.md` (31 KB) is the authoritative API reference: options, cache-key composition, per-backend
toolchain requirements, and the full environment-variable table. Read the relevant section before
changing a backend's compile/load path.

## Commands

```bash
# Full platform test: builds the C++ harness, runs it, validates the cache artifacts on disk.
python tests/test_maca.py          # MACA: source the MACA env first (MACA_PATH, LD_LIBRARY_PATH
                                   #   must include $MACA_PATH/lib and $MACA_PATH/mxgpu_llvm/lib)
python tests/test_cuda.py          # CUDA: needs a GPU + CUDA 12.4+ headers / NVCC 12.9+
python tests/test_ascend.py        # Ascend: needs CANN + ACL/torch_npu headers
python tests/test_exception.py     # header-only check of the exception/backtrace path

# Build just the MACA harness (host side is clang++-22 + a GCC-15 conda sysroot; mxcc is only
# invoked later, at run time, by the JIT).
bash tests/test_maca_proj/build.sh /tmp/djbuild/test_maca

# CMake is for debugging / IDE indexing only. It builds a pybind11 `_C` module and exists to
# exercise the cuda|maca backend switch; consumers integrate via include path, not via this target.
cmake -S . -B build -DDEEP_JIT_PLATFORM=maca    # or cuda (default)
```

Tests are plain `assert`-based Python; run them without `-O` (`python -O` trips the guard
`if not __debug__: raise`). The CUDA/Ascend drivers re-exec themselves as `--worker` in a child
process group so that leaked device processes can be killed.

### Running one case

There is **no test filter** in the C++ harnesses (`run_test(...)` calls in `main()` are unconditional).
To iterate on a single case, run the built harness directly with its env, or comment out the other
`run_test` lines in `main()`:

```bash
DEEP_JIT_MACA_TEST_SOURCE_DIR=$PWD/tests/test_maca_proj \
DEEP_JIT_MACA_TEST_CACHE_DIR=/tmp/dj_cache \
LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH \
/tmp/djbuild/test_maca
```

The MACA harness also takes `--diagnostics` (one compile/load with the reports enabled) and
`--compile-once <tag> <bias> <ready> <start>` (one compile behind a file barrier), which the Python
driver uses for the diagnostic and multi-process race cases.

## Architecture

### One runtime, three backends

`deep_jit::Runtime<Backend>` (`include/deep_jit/runtime/runtime.hpp`) is the whole public surface, and
it is **backend-agnostic**: it owns `Config`, `Env`, `Device`, `DiskCache`, `Parser`, the memory
cache, and the backend object, and drives the shared compile/load/launch flow. A backend supplies
four nested types — `Device`, `Kernel`, `CompilerOptions`, `LaunchOptions` — plus
`compile(source, dir, env, config, options)` and `static load(dir, env)`. Include exactly one of
`backend/{cuda,maca,ascend}/backend.hpp`; CUDA and MACA cannot coexist in one translation unit (the
CUDA driver header asserts CUDA ≥ 12.4 and binds `cuLibrary*`, which MACA's cu-bridge headers lack).

Adding a backend means writing those five pieces; the caching, hashing, env precedence, and filesystem
protocols are inherited for free.

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
       │    └─ writes kernel.<ext> (cubin / devbin / rel.o), optional dumps, meta.json
       └─ entry.commit()   # fsync tree, atomic directory rename, .committed marker
            └─ backend.load(dir, env) → shared_ptr<Kernel>, process-cached
launch(kernel, opts, args...)  →  kernel->launch(default_launch_options.override_with(opts), args...)
```

Key consequences:

- **The compile tag is not part of the cache key** — it only names the entry. Different tags over the
  same source and options share a digest; the tag must be letters/digits/underscore only.
- **`include_dirs` are deliberately excluded from the digest** (moving the same header contents must
  not invalidate); any `-I` you pass through `*_flags` *does* count. `extra_signature` is where
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
no effect. Suffixes (`JIT_CACHE_DIR`, `JIT_DEBUG`, `JIT_DUMP_PTX`, `JIT_CHECK_NO_SPILLS`, …) are
tabulated in README; `utils/env.hpp` implements the lookup. Snapshotting happens at runtime
construction, so set these before the first `get_jit()`/`jit->...`.

### Backend specifics worth knowing

**MACA** (`backend/maca/`) — compiles with `mxcc -x maca -device-bin` to `kernel.devbin`. `-device-bin`
is the only artifact form that serves both the load path and name discovery: MACA has no driver-side
kernel enumeration (`mcModuleGetFunctionCount` et al. are absent from `libmcruntime.so`), so the entry
point name is recovered from the binary with the toolchain's `llvm-nm` (`Kernel::parse_kernel_names`).
The driver entry points are bound lazily with `dlopen`/`dlsym` on `libmcruntime.so`
(`maca/driver.hpp`) — the `cu*` spellings are a different library and are not used.

- Launch goes through `mcModuleLaunchKernelEx`. **Cluster launch and PDL are rejected up front**, and
  only the cooperative flag is passed as an attribute: the MACA runtime refuses every other launch
  attribute with `mcErrorInvalidConfiguration`. There is no dynamic-smem attribute call either — the
  launch itself enforces the `sharedMemPerBlockOptin` ceiling. `maca/kernel.hpp` and `maca/driver.hpp`
  carry the measured evidence; don't "fix" those paths back toward the CUDA shape.
- `check_no_spills` and `check_no_local_memory` coincide: mxcc reports a single "N bytes stack frame"
  figure and emits it only under `-resource-usage`, so either check implies that flag.
- Architecture: mc device major 10/15/16 → `xcore1000`/`xcore1500`/`xcore1600` (C500/C600/C600U).
  `Device::get_arch()` returns the family digits, and options render `--offload-arch=xcore<digits>`.
  An unknown major is a panic, not a guess.
- `MACA::load` is static (the `Runtime` call shape is shared with the other backends) but resolves the
  toolkit itself from the same `env` the `Runtime` hands it — MACA's load needs `llvm-nm` out of the
  toolkit, so unlike CUDA's pass-through it cannot ignore the toolkit, and a process-global would let
  one `Runtime`'s toolkit decide another's loads (and race between two constructors).

**CUDA** (`backend/cuda/`) — NVCC → CUBIN, loaded through the driver `cuLibrary*` API; the richest
option set (PTX/SASS dumps, `ptxas_register_usage_level`, clusters, cooperative launch, PDL).
Compile and load before CUDA Graph capture; launching an already-loaded kernel is capture-compatible.

**Ascend** (`backend/ascend/`) — Bisheng + `ld.lld` → `kernel.rel.o`, name from the unique
`.ascend.meta.*` symbol, loaded through ACL; `num_blocks` is the required launch dimension, and the
current torch_npu stream is used when none is set.

### Python integration

`python_api.hpp` registers `Runtime<Backend>` and a `get_jit()` on a pybind11 module; a consumer
module just calls `deep_jit::register_python_api(module, my_library::jit)` and Python obtains the same
process-local runtime. `deep_jit/__init__.py` is intentionally empty — the library is header-only and
the Python package exists for packaging namespace only. `csrc/python_api.cpp` is the CMake-only demo
module and picks its backend from `DEEP_JIT_PLATFORM`.

## Conventions and traps

- **Format through fmt, never `std::format`, in `include/deep_jit/**`.** `<format>` predates some host
  toolchains DeepJIT gets embedded into (mxcc drives a GCC 11 libstdc++), and a header-only library's
  stdlib floor becomes its consumer's. `utils/format.hpp` defines `FMT_HEADER_ONLY` and pulls
  `<fmt/format.h>`; the consumer supplies the include path. The test harnesses under `tests/` are
  built with a modern host toolchain and do use `std::format` — that is not a precedent for the
  headers.
- **Kernel arguments**: pass raw values (the driver takes their address) or wrap a device pointer in
  `deep_jit::NoRefPtr{ptr}` when the argument *is* the pointer value rather than a location holding
  it. `kernel_arg_pointer` in each backend's `kernel.hpp` implements the unwrap.
- **Errors** are `DJ_HOST_ASSERT` / `DJ_PANIC` (`utils/exception.hpp`, throws
  `deep_jit::exception::Exception`; backtrace via lazily-loaded `libdw`), and per-backend check macros
  (`DJ_CUDA_DRIVER_CHECK` / `DJ_CUDA_RUNTIME_CHECK`, `DJ_MACA_DRIVER_CHECK` / `DJ_MACA_RUNTIME_CHECK`,
  `DJ_ACL_CHECK`). They surface as `RuntimeError` in Python.
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

Each platform has a Python driver (`tests/test_<platform>.py`) that builds and drives a C++ harness in
`tests/test_<platform>_proj/`. The harness registers its cases with `run_test("name", ...)` and prints
`[ RUN ] / [ OK ] / [ SKIPPED ]`.

- CUDA and Ascend look like a real consumer: `main.cpp` is a pybind11 module built with
  `torch.utils.cpp_extension.load`, and the driver checks the Python-facing surface too (lazy import,
  fork-after-lazy-init, GIL release during compile, `get_jit()`).
- MACA has no Python embedding: `main.cpp` is a standalone host binary built by
  `test_maca_proj/build.sh`, which needs the conda clang++-22 / GCC-15 sysroot; kernel sources live in
  `test_maca_proj/kernels/` and hooks in `test_maca_proj/scripts/`.
- Drivers validate more than the harness reports: **header self-containment** (each backend header
  must compile alone with `-Werror`), the exact cache-entry shape on disk (`kernel.cu`,
  `kernel.devbin`/`cubin`/`rel.o`, `meta.json`, `.committed`, allowed dumps), the `meta.json` schema,
  multi-process races onto one key, diagnostic output under each env prefix, and the lazy-import /
  GIL / crashed-writer paths.
- Failure-mode kernels are deliberate and must stay: e.g. `forced_register_spill.cu`,
  `explicit_local_memory.cu`, `multiple_kernels.cu`, `no_kernel.cu`, `include_cycle/`,
  `include_same_content/` vs `include_changed_content/`. When adding an option or a cache-key input,
  extend the corresponding `*_original/ / *_same_content/ / *_changed_content/` triplet so the digest
  is proven to track it. All three harnesses carry such triplets (CUDA for `test_cuda/` includes,
  Ascend for `test_ascend/`, MACA for `test_maca/` includes and for the untracked `third_party/`
  dependency), which is what pins "same contents at a different path must not change the digest" and
  "changed contents must".
- Rebuild before believing a result. Kernel sources are read at *run* time (the runtime compiles
  `kernels/*.cu` on demand), but the harness's own assertions are baked into the binary — so a
  kernel-only edit needs no rebuild, while a `main.cpp` edit without one silently runs the *old*
  expectations against the *new* kernel and surfaces as a spurious mismatch in whatever the two
  compute jointly. `python tests/test_maca.py` always builds first; a hand-run of
  `/tmp/djbuild/test_maca` does not.
- Test case names are shared vocabulary across the three harnesses: `<subject> <aspect>` with spaces
  (`invalid cache tag`, `secondary disk cache`, `compiler failure cleanup`, `macro-selected kernel
  ABI`, …). Keep new cases on that pattern so coverage can be diffed across harnesses by name; cases
  that exist on only one platform are named for the platform (`CUDA device` / `MACA device`, `PTXAS
  checks` / `resource usage checks`).
