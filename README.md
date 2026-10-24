# DeepJIT

DeepJIT is a lightweight, header-only C++20 JIT runtime for **MACA GPUs**. It gives C++/Python extension authors a shared interface for compiling kernel source at runtime, caching the resulting binaries, loading them onto the device, and launching them with backend-specific options.

DeepJIT handles the JIT infrastructure so that kernel libraries can focus on their device code. The runtime configuration, source and include hashing, in-memory and on-disk caches, and lazy initialization are shared; kernel source and compiler/launch options remain MACA-specific.

**Main authors:** [@guyan364](https://github.com/guyan364), [@kurisu6912](https://github.com/kurisu6912), [@LyricZhao](https://github.com/LyricZhao).

## Features

- **MACA backend:** use `deep_jit::Runtime<deep_jit::MACA>` for the compile/load/launch workflow.
- **Kernel caching:** reuse loaded kernels in memory and compiled artifacts on disk. Cache keys account for source, tracked includes, compiler versions, effective compiler options, and an application-provided dependency signature.
- **Distributed filesystems and shared caches:** share one cache directory across users, processes, and nodes to reuse compiled kernels. DeepJIT uses local and distributed filesystems with the required POSIX filesystem semantics; see [Shared cache](#shared-cache) for configuration.
- **Lazy initialization:** defer device and compiler discovery until the runtime is first used.
- **PyTorch integration:** use the current PyTorch stream by default, and expose the configured runtime to Python through TVM FFI with `get_jit()`.
- **Compilation controls and diagnostics:** configure runtime defaults and per-kernel overrides, inspect compilation metadata, dump MACA assembly, and run a Python post-compilation hook.

### In development (WIP)

- **Cache warmup from history:** use historical cache entries to anticipate kernels that future runs may need and warm up their cache in advance, reducing compilation delays during execution. This feature is under development and is not yet available.
- **Python compilation API:** pass kernel source code directly from Python to compile a kernel. This feature is under development and is not yet available.

## Supported backend

| Backend | Device toolchain and runtime | Integration requirements |
| --- | --- | --- |
| **MACA** | mxcc compiles MACA source to a pre-linked device binary; `libmcruntime` loads and launches kernels. | An MXMACA install providing `mxgpu_llvm/bin/mxcc` 1.0+ and `mxgpu_llvm/bin/llvm-nm` (via `MACA_PATH`, or `/opt/maca`), plus the MACA headers and a PyTorch built for the platform. |

The host environment must provide Linux, a C++20 compiler, [fmt][fmt] (header-only), and the [elfutils][elfutils] development headers providing `<elfutils/libdwfl.h>` (`libdw-dev` on Debian/Ubuntu, `elfutils-devel` on RHEL/Fedora). `libdw` is loaded at run time with `dlopen`; only its headers are needed at build time, and consumers on older glibc should additionally link `-ldl`.

The optional `python_api.hpp` binding header also requires [TVM FFI][tvm-ffi]. The test harnesses additionally need PyTorch (`tests/test_maca.py`) and TVM FFI (`tests/test_exception.py`); both are declared in [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`). DeepJIT is intended to be embedded into your extension as a header-only dependency.

DeepJIT formats through fmt rather than `std::format`, because some host toolchains (notably the GCC 11 libstdc++ behind MACA's `mxcc`) predate `<format>`. See [`include/deep_jit/utils/format.hpp`](include/deep_jit/utils/format.hpp).

[elfutils]: https://sourceware.org/elfutils/
[fmt]: https://github.com/fmtlib/fmt
[tvm-ffi]: https://github.com/apache/tvm-ffi

See [Integration](#integration) for setup and [MACA](#maca) for backend usage.

## Shared cache

The disk-cache implementation supports local and distributed filesystems that provide atomic directory rename within a filesystem and file/directory `fsync`. Builds use unique temporary directories, synchronize their contents, and publish complete entries through an atomic rename. Concurrent processes can compile the same entry and reuse the published result.

Multiple users, processes, and nodes can point to the same cache directory:

```bash
export DJ_JIT_CACHE_DIR=/shared/deep_jit
```

Matching compilation inputs and cache tags allow users to reuse each other's compiled kernels.

For multi-user access, set the shared cache root's group and SGID bit as shown below. DeepJIT reads SGID once when the cache is initialized. If set, it applies directory permissions `0775` and file permissions `0664`, ignoring umask to prevent access problems. It preserves inherited SGID on directories. Use a trusted group and configure existing directories separately.

Have the administrator run these commands, with `$GROUP` set to a group that includes all participating users:

```bash
mkdir -p /shared/deep_jit
chgrp "$GROUP" /shared/deep_jit
chmod 2775 /shared/deep_jit
```

Without SGID, normal umask rules apply.

You can also combine a writable personal cache with a shared lookup cache:

```bash
export DJ_JIT_CACHE_DIR="$HOME/.dj:/shared/deep_jit"
```

DeepJIT searches all roots in order and writes cache misses only to the first root. The shared lookup cache can be read-only. To configure a single consumer library, use its prefix instead, for example `MYLIB_JIT_CACHE_DIR`.

## Repository layout

| Path | Contents |
| --- | --- |
| [`include/deep_jit/runtime/`](include/deep_jit/runtime/) | Shared runtime and configuration. |
| [`include/deep_jit/backend/maca/`](include/deep_jit/backend/maca/) | MACA compiler, device queries, kernel loading, and launch options. |
| [`include/deep_jit/cache/`](include/deep_jit/cache/) | In-memory and on-disk kernel caches. |
| [`include/deep_jit/python_api.hpp`](include/deep_jit/python_api.hpp) | TVM FFI export of a consumer library's runtime through `get_jit()`. |
| [`tests/`](tests/) | MACA integration tests, example extensions, and device kernels. |

The root `CMakeLists.txt` is for debugging and IDE indexing. Integrate the headers into your own extension as described below; [`tests/test_maca_proj/`](tests/test_maca_proj/) provides a standalone host binary that drives the runtime without embedding Python.

## Integration

The DeepJIT runtime headers do not touch Python or the GIL. GIL release belongs to the binding layer: TVM FFI releases the GIL around every Python-to-C++ call by default. Consumers using another binding must manage the GIL themselves around `compile()`, `load()`, and `launch()`.

Add `DeepJIT/include` to the include path of the host target, then include the MACA backend entry header:

```cmake
target_include_directories(my_target PRIVATE third-party/deep_jit/include)
```

```cpp
#include <deep_jit/backend/maca/backend.hpp>

using JIT = deep_jit::Runtime<deep_jit::MACA>;
```

`create_lazy_jit` delays construction of the runtime until its first use. This also delays device and compiler discovery:

```cpp
inline auto jit = deep_jit::create_lazy_jit<deep_jit::MACA>(
    deep_jit::Config("/absolute/path/to/my_library", "MYLIB"));
```

If configuration is only known during library initialization, start with an empty lazy object and assign its factory later:

```cpp
#include <filesystem>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/maca/backend.hpp>

namespace my_library {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::MACA>> jit(nullptr);

inline void init_jit(const std::string& library_root) {
    const auto library_root_path = std::filesystem::absolute(library_root);
    const auto include_dir = library_root_path / "include";

    jit = deep_jit::create_lazy_jit<deep_jit::MACA>(
        deep_jit::Config(
            library_root_path,
            "MYLIB",
            "cutlass-" + std::to_string(CUTLASS_VERSION),
            {include_dir},
            {"my_library/"}));
}

}  // namespace my_library
```

The lazy object must receive a factory before `jit->...` or Python `get_jit()` is called.

### Config

`deep_jit::Config` has the following constructor:

```cpp
Config(std::filesystem::path python_library_root,
       std::string env_prefix,
       std::string extra_signature = {},
       std::vector<std::filesystem::path> include_dirs = {},
       std::vector<std::string> include_prefixes = {});
```

- `python_library_root` resolves relative `post_hook` paths. It must be non-empty and absolute.
- `env_prefix` selects the library-specific environment-variable prefix. It must be non-empty and cannot be `DJ`, which is reserved for global defaults.
- `extra_signature` represents dependencies that affect generated code but are not tracked by the include parser. Change it when such a dependency changes, for example `"cutlass-" + std::to_string(CUTLASS_VERSION)`.
- `include_dirs` are passed to the compiler and searched by the include parser. Every path must be absolute.
- `include_prefixes` select which angle-bracket includes are recursively tracked. For example, `"my_library/"` tracks `#include <my_library/kernel.cuh>`.

Configuration is snapshotted when `Runtime` is constructed. Do not mutate `config`, `backend`, `parser`, `disk_cache`, or `hash_base` afterward. The supported mutable per-runtime settings are `default_compiler_options` and `default_launch_options`; configure them before sharing the runtime across threads.

Once constructed, a runtime may be used concurrently from multiple threads: lazy initialization, the device-property snapshot, the include parser, and the in-memory kernel cache are synchronized. A compile miss for one key runs the compiler once; other callers wait for that result.

Compiler, cache, include, and hook paths, as well as free-form compiler flags, are currently passed through a simple shell command and therefore must not contain whitespace or shell metacharacters.

### Registering `get_jit()`

`python_api.hpp` exports `get_jit()` from a consumer shared library through TVM FFI:

```cpp
#include <tvm/ffi/function.h>

#include <deep_jit/backend/maca/backend.hpp>
#include <deep_jit/python_api.hpp>

namespace my_library {
inline auto jit = deep_jit::create_lazy_jit<deep_jit::MACA>(
    deep_jit::Config("/absolute/path/to/my_library", "MYLIB"));
}

DJ_EXPORT_GET_JIT(my_library::jit);
```

`DJ_EXPORT_GET_JIT` defines a symbol, so invoke it at global scope in exactly one translation unit per shared library. The target must link TVM FFI's shared library (`tvm_ffi::shared`); the root `CMakeLists.txt` shows how the package is located from an installed Python distribution. The exported name is always `get_jit`, and Python obtains an opaque handle that keeps the process-local runtime alive:

```python
import tvm_ffi

module = tvm_ffi.load_module("/path/to/my_library.so")
jit = module.get_jit()
```

The handle exposes no fields or methods; each call creates a new handle over the same runtime. `tvm_ffi.load_module` must keep the module alive (the default), and `libtvm_ffi.so` must be loadable when the library is opened. Calling `get_jit()` initializes the lazy runtime if it has not already been initialized.

## MACA

The MACA backend requires an MXMACA install that provides `mxgpu_llvm/bin/mxcc` (version 1.0 or newer) and the `llvm-nm` beside it. It compiles `kernel.cu` to `kernel.devbin` with `mxcc -device-bin` and loads it through `mcModuleLoad`. Loading requires exactly one kernel unless `CompilerOptions::kernel_name` selects an entry point; `compile_without_load()` only builds the artifact.

MACA has no driver-side kernel enumeration, so the entry-point name is recovered from the device binary with `llvm-nm`. `-device-bin` is the only output that both `llvm-nm` can read and `mcModuleLoad` can load.

### Compile and launch

```cpp
inline auto jit = deep_jit::create_lazy_jit<deep_jit::MACA>(
    deep_jit::Config(
        "/absolute/path/to/my_library",
        "MYLIB",
        {},
        {"/absolute/path/to/my_library/include"},
        {"my_library/"}));

const auto kernel = jit->compile("scale", source);
jit->launch(
    kernel,
    {
        .grid_dim = dim3((count + 255) / 256, 1, 1),
        .block_dim = dim3(256, 1, 1),
    },
    output,
    input,
    count);
```

The compile tag must contain only letters, digits, and underscores. `compile()` compiles on a cache miss, loads the device binary, and returns a process-cached `std::shared_ptr<deep_jit::maca::Kernel>`. `compile_without_load()` only returns the artifact directory.

Unset `deep_jit::maca::CompilerOptions` fields inherit from the runtime defaults, which are `--offload-arch=xcore<family>`, `-O3`, and `-std=c++20`:

```cpp
deep_jit::maca::CompilerOptions options {
    .optimize_level = "3",
    .fast_math = true,
    .arch = jit->device.get_arch(),
    .extra_mxcc_flags = {"-DMY_OPTION=1"},
    .post_hook = "hooks/hook_1.py",
};

const auto kernel = jit->compile("scale", source, options);
```

`mxcc_flags` replaces the default free-form mxcc flag list; structured options such as `optimize_level` are generated separately. `extra_mxcc_flags` appends per-kernel flags such as `-D` definitions. The optimization level, fast math (`-use-fast-math`), the resource report (`-resource-usage`), the spill and local-memory checks, line information (`with_line_info`), the assembly dump, the architecture, the entry-point selection, and the post hook are all supported. To extend the runtime defaults, append flags directly to `*jit->default_compiler_options.mxcc_flags` during initialization.

`kernel_name` selects the function to load when the device binary exports more than one kernel; without it, loading requires exactly one. For example:

```cpp
const auto kernel = jit->compile("scale", source, {.kernel_name = "scale_kernel"});
```

The name must match a defined entry point in the artifact. It participates in the cache digest and is recorded in `meta.json`, so different entry points do not share a cached kernel. `compile_without_load()` returns the artifact directory; pass the same effective options to `MACA::load(dir, env, options)` to load a selected entry point from it.

The resource report is `compiler_verbose` (`JIT_PTXAS_VERBOSE` is the shared environment-variable name). `ptxas_verbose` is an override-only alias for it; `compiler_verbose` wins if an override carries both.

`check_no_spills` and `check_no_local_memory` both reject a non-empty mxcc stack-frame report: mxcc has no separate local-memory figure, and either check implies `-resource-usage`.

MACA launch options include the stream, dynamic shared-memory size, grid, block, cluster (`cluster_dim`), cooperative-launch, and PDL (`enable_pdl`) controls. Unset fields inherit from `jit->default_launch_options`. Grid and block dimensions are required and must be positive, and the dynamic shared-memory size cannot be negative. An unset effective stream uses the current PyTorch stream.

**Cluster launch** and **programmatic dependent launch** are not implemented by the MACA runtime and are rejected before the driver is reached; `mcModuleLaunchKernelEx` honours only the cooperative flag. Dynamic shared memory needs no attribute call: the launch enforces the `sharedMemPerBlockOptin` ceiling itself.

Device information is available from `jit->device`:

```cpp
const int num_sms = jit->device.get_num_sms();
const int l2_bytes = jit->device.get_num_l2_cache_bytes();
const int smem_bytes = jit->device.get_num_smem_bytes();
const int64_t clock_rate = jit->device.get_clock_rate();
const auto [major, minor] = jit->device.get_arch_pair();
const std::string family_arch = jit->device.get_arch();
```

`get_family()` and `get_arch()` map the native device major to the xcore family that names the mxcc offload target: `10`, `15`, and `16` to `xcore1000`, `xcore1500`, and `xcore1600` (C500/C600/C600U). The minor revision is a revision inside the family and is not a judgment axis; an unknown major is an error rather than a speculative mapping. A runtime snapshots the current device when it is constructed; use one runtime per device and do not move a loaded kernel across contexts.

### Post hook

`CompilerOptions::post_hook` selects one optional Python file under `Config::python_library_root`. For example:

```text
my_library/hooks/hook_1.py
```

Pass the file's relative path through the compiler options:

```cpp
const auto kernel = jit->compile("scale", source, {
    .post_hook = "hooks/hook_1.py",
});
```

`post_hook` inherits from `jit->default_compiler_options` like every other compiler option and is unset by default. Only `std::nullopt` disables it; an empty string is still treated as a configured hook. Once a default hook is set, the current override model cannot disable it for one kernel. The caller must provide a trusted relative path. When set, it runs after mxcc produces `kernel.devbin` and before the artifact is published:

```bash
cd <temporary-artifact-directory>
python <absolute-python-library-root>/hooks/hook_1.py <absolute-device-binary-path>
```

The script receives the absolute device-binary path as its only argument and must modify that file in place, leaving it loadable if the same artifact is to be loaded afterwards. The configured relative `post_hook` path and file-content hash are included in the kernel cache digest; the path is also recorded in `meta.json`. Hook hashes are cached per thread, so restart the process after changing a hook file.

A hook must be deterministic for a given input binary and tracked signature. Imported Python modules, auxiliary files, Python/package versions, environment variables, random state, time, and network data are not discovered automatically; represent every such dependency in `extra_signature` if it can affect output.

### Cache-key and artifact rules

The MACA cache digest is built, in order, from:

1. `Config::extra_signature`.
2. The hash of the complete `mxcc --version` output.
3. The effective compiler flags returned by `CompilerOptions::get_flags()`, followed by `CompilerOptions::kernel_name` when set. Paths in `Config::include_dirs` are intentionally excluded, while any `-I...` placed directly in `mxcc_flags` or `extra_mxcc_flags` remains part of the digest.
4. The selected `post_hook` path and file-content hash.
5. The parser digest of the source and its tracked include tree.

Each component is prefixed by its fixed-width byte length before it is added to the two-state FNV-1a hash, so boundaries remain unambiguous even for binary strings containing zero bytes. The final digest is a 32-character hexadecimal string. This is a fast cache checksum, not a cryptographic hash.

The compile tag is not part of the digest. A disk entry is stored as `<cache-root>/cache/<tag>.<digest>/`, carries a `.committed` marker, and contains `kernel.cu`, `kernel.devbin`, and `meta.json`, plus `kernel.s` when `dump_asm` is set. `meta.json` records the config fields, the compiler and its version, the effective compiler options, and the mxcc command arguments used in the temporary build directory. Temporary paths in that command may no longer exist after publication. Cache roots must be trusted.

`dump_asm` writes the device assembly with `-aop -S --device-obj` and does not affect the cache digest; it only runs when compilation actually occurs, so a pre-existing cache hit is not rebuilt to add a missing dump. `JIT_DUMP_PTX` and `JIT_PTXAS_VERBOSE` are accepted as aliases for the assembly dump and resource report respectively, and `JIT_DUMP_SASS` has no effect.

The cache key assumes the external compiler environment is stable. Variables such as `CPATH`, host-compiler selection, and transitive third-party header changes are not discovered automatically. Avoid such implicit inputs or represent them in `extra_signature`/explicit compiler flags. Excluding tracked include-directory paths is safe only when moving the same header contents does not change compilation behavior such as embedded `__FILE__` strings.

### Environment variables

MACA toolkit discovery resolves `mxcc` in this order: `JIT_MXCC_COMPILER`, the `MACA_CLANG_PATH` bin directory, the `MACA_PATH` install root, `mxcc` on `PATH` (with symlinks resolved), then `/opt/maca`. `MACA_HOME` and the CUDA_* pair are not consulted: `CUDA_HOME`/`CUDA_PATH` name the cu-bridge root in this ecosystem (a subdirectory with no `mxgpu_llvm/`), so as an install root they could only fail. Under an install root the compiler is `<home>/mxgpu_llvm/bin/mxcc`, and `llvm-nm` is resolved beside it unless `JIT_LLVM_NM` is set. Install roots and `MACA_CLANG_PATH` are read unprefixed; the two `JIT_*` overrides follow the library prefix.

## Include parser

The root source is always hashed. An included file is recursively tracked only when both of the following are true:

- The directive uses a literal angle-bracket include: `#include <...>`.
- The included filename starts with one of `Config::include_prefixes`.

The parser is intentionally a line-oriented scanner, not a C preprocessor. A tracked directive must use the canonical single-line form `#include <...>` (whitespace around `#`, `include`, and the filename is allowed). It does not interpret comments between tokens, macro-expanded includes, backslash continuations, or conditional compilation; an include inside `#if 0` is still scanned.

Tracked files are resolved by checking `include_dirs` in order and using the first matching path. A tracked file that cannot be found is an error. The same rules are applied recursively to includes inside tracked files.

Quoted includes such as `#include "kernel.cuh"` are rejected. Macro-based includes are not tracked. Angle-bracket includes that do not match any configured prefix are accepted by the compiler but ignored by the parser.

For example, with:

```cpp
deep_jit::Config(
    std::filesystem::path("/opt/my_library"),
    "MYLIB",
    "cutlass-40000",
    {std::filesystem::path("/opt/my_library/include")},
    {"my_library/"});
```

`#include <my_library/kernel.cuh>` is tracked, while `#include <cutlass/cutlass.h>` is not. The CUTLASS version is instead represented by `extra_signature`.

Tracked include graphs must not contain cycles. Header digests are cached for the lifetime of a runtime, so recreate the runtime after changing tracked header files.

## Environment variables

For `Config("/absolute/path/to/my_library", "MYLIB")`, every DeepJIT setting is resolved in this order:

```text
MYLIB_<SUFFIX> > DJ_<SUFFIX> > built-in default
```

For example:

```text
MYLIB_JIT_CACHE_DIR > DJ_JIT_CACHE_DIR > $HOME/.dj
```

The library prefix therefore allows one consumer to be configured independently, while the reserved `DJ_` prefix provides process-wide defaults. Boolean values accept `true`/`false`, `yes`/`no`, or any integer, case-insensitively.

Only the library-prefixed and `DJ_` forms are read. An unprefixed variable such as `JIT_CACHE_DIR` or `JIT_DEBUG` has no effect.

Set JIT environment variables before the first runtime construction (normally before the first `get_jit()` or `jit->...`). Cache roots, compiler selection, C++ standard, and default compiler options are snapshotted then. Compiler-command printing and load-time diagnostics are read again when compile/load runs; changing variables after initialization can therefore produce a mixed configuration and is unsupported.

| Suffix | Default | Behavior |
| --- | --- | --- |
| `JIT_CACHE_DIR` | `$HOME/.dj` | Cache root, or a colon-separated list. All roots are searched in order; misses are compiled into the first root. Empty values or empty list elements are rejected. |
| `JIT_DEBUG` | `0` | Enables compiler-command and load diagnostics, plus the MACA resource report, line information, and assembly dump. |
| `JIT_MXCC_COMPILER` | `<discovered-toolkit>/mxgpu_llvm/bin/mxcc` | Overrides the mxcc executable. `llvm-nm` must exist next to it. |
| `JIT_LLVM_NM` | `<mxcc-directory>/llvm-nm` | Overrides the `llvm-nm` used for kernel-name discovery. |
| `JIT_CPP_STANDARD` | `20` | Selects the C++ standard passed to the device compiler as `-std=c++<value>`. |
| `JIT_PRINT_COMPILER_COMMAND` | `0` | Prints compiler and disassembler commands. |
| `JIT_PTXAS_VERBOSE` | `0` | Adds `-resource-usage` and prints the report after compilation. |
| `JIT_CHECK_NO_SPILLS` | `0` | Adds `-resource-usage` and rejects a non-empty stack frame. |
| `JIT_CHECK_NO_LOCAL_MEMORY` | `0` | Rejects a non-empty stack frame, the only resource figure mxcc reports. |
| `JIT_PRINT_LOAD_TIME` | `0` | Prints kernel-binary loading time. |
| `JIT_WITH_LINEINFO` | `0` | Adds source line information. |
| `JIT_DUMP_ASM` | `0` | Generates a MACA assembly artifact on a cache miss. |
| `JIT_DUMP_PTX` | `0` | Alias for `JIT_DUMP_ASM`. |
| `JIT_DUMP_SASS` | `0` | Accepted for compatibility; has no effect. |

| Variable | Behavior |
| --- | --- |
| `HOME` | Required for the default `$HOME/.dj` cache path. |
| `MACA_PATH` | The MACA install root. |
| `MACA_CLANG_PATH` | Bin directory holding `mxcc`, tried before the install root. |
| `PATH` | Used by `which mxcc` when no root is configured. |
