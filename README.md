# DeepJIT

DeepJIT is a lightweight, header-only C++20 JIT runtime for **NVIDIA CUDA GPUs**, **MACA GPUs**, and **HUAWEI Ascend (昇腾) NPUs**. It gives C++/Python extension authors a shared interface for compiling kernel source at runtime, caching the resulting binaries, loading them onto the device, and launching them with backend-specific options.

DeepJIT handles the JIT infrastructure so that kernel libraries can focus on their device code. Every backend shares runtime configuration, source and include hashing, in-memory and on-disk caches, and lazy initialization. Kernel source and compiler/launch options remain specific to the selected backend.

**Main authors:** [@guyan364](https://github.com/guyan364), [@kurisu6912](https://github.com/kurisu6912), [@LyricZhao](https://github.com/LyricZhao).

## Features

- **CUDA, MACA, and Ascend backends:** use `deep_jit::Runtime<deep_jit::CUDA>`, `deep_jit::Runtime<deep_jit::MACA>`, or `deep_jit::Runtime<deep_jit::Ascend>` with the same compile/load/launch workflow.
- **Kernel caching:** reuse loaded kernels in memory and compiled artifacts on disk. Cache keys account for source, tracked includes, compiler versions, effective compiler options, and an application-provided dependency signature.
- **Distributed filesystems and shared caches:** share one cache directory across users, processes, and nodes to reuse compiled kernels. Every backend supports local and distributed filesystems with the required POSIX filesystem semantics; see [Shared cache](#shared-cache) for configuration.
- **Lazy initialization:** defer device and compiler discovery until the runtime is first used.
- **PyTorch integration:** use the current PyTorch stream by default, and expose the configured runtime through pybind11 with `get_jit()`.
- **Compilation controls and diagnostics:** configure runtime defaults and per-kernel overrides, inspect compilation metadata, and dump CUDA PTX/SASS, MACA assembly, or Ascend assembly. CUDA and MACA support a Python post-compilation hook.

### In development (WIP)

- **Cache warmup from history:** use historical cache entries to anticipate kernels that future runs may need and warm up their cache in advance, reducing compilation delays during execution. This feature is under development and is not yet available.
- **Python compilation API:** pass kernel source code directly from Python to compile a kernel for any backend. This feature is under development and is not yet available.

## Supported backends

| Backend | Device toolchain and runtime | Integration requirements |
| --- | --- | --- |
| **CUDA** | NVCC compiles CUDA source to CUBIN; the CUDA Driver API loads and launches kernels. | CUDA headers 12.4+, NVCC 12.9+, and PyTorch with CUDA support. |
| **MACA** | mxcc compiles MACA source to a pre-linked device binary; `libmcruntime` loads and launches kernels. | An MXMACA install providing `mxgpu_llvm/bin/mxcc` 1.0+ and `mxgpu_llvm/bin/llvm-nm` (via `MACA_PATH`, or `/opt/maca`), plus the MACA headers and a PyTorch built for the platform. |
| **Ascend** | Bisheng and ld.lld compile and link Ascend kernel source; ACL loads and launches kernels. | CANN with `bin/bisheng`, `bin/ld.lld`, and the Ascend `adv_api` headers; ACL and `torch_npu` headers and runtime. |

The host environment must provide Linux, a C++20 compiler, [fmt][fmt] (header-only), and the dependencies for the selected backend. The default GIL support also requires Python and pybind11; consumers that manage the GIL themselves can disable it as described under [Integration](#integration). DeepJIT is intended to be embedded into your extension as a header-only dependency. CUDA consumers should compile with `TORCH_TARGET_VERSION=0x020a000000000000` and `USE_CUDA`; this targets PyTorch's stable ABI with PyTorch 2.10 as the minimum runtime version.

DeepJIT formats through fmt rather than `std::format`. `<format>` is a C++20 *library* feature that not every host toolchain has: MACA's `mxcc`, for instance, drives the system GCC 11, whose libstdc++ predates it. Since a header-only library's standard-library requirements become its consumer's requirements, going through fmt keeps DeepJIT embeddable on those toolchains. The formatting language and the output are the same; see [`include/deep_jit/utils/format.hpp`](include/deep_jit/utils/format.hpp).

[fmt]: https://github.com/fmtlib/fmt

See [Integration](#integration) for setup, [CUDA](#cuda) for GPU usage, [MACA](#maca) for MACA GPU usage, and [Ascend](#ascend) for NPU usage.

## Shared cache

Every backend uses the same disk-cache implementation. It supports local and distributed filesystems that provide atomic directory rename within a filesystem and file/directory `fsync`. Builds use unique temporary directories, synchronize their contents, and publish complete entries through an atomic rename. Concurrent processes can compile the same entry and reuse the published result.

Multiple users, processes, and nodes can point to the same cache directory:

```bash
export DJ_JIT_CACHE_DIR=/shared/deep_jit
```

Configure directory permissions so participating users can read shared artifacts and writers can create and publish entries under the cache root. As with all DeepJIT caches, use a trusted shared directory. Matching compilation inputs and cache tags allow users to reuse each other's compiled kernels.

You can also combine a writable personal cache with a shared lookup cache:

```bash
export DJ_JIT_CACHE_DIR="$HOME/.dj:/shared/deep_jit"
```

DeepJIT searches all roots in order and writes cache misses only to the first root. The shared lookup cache can be read-only. To configure a single consumer library, use its prefix instead, for example `MYLIB_JIT_CACHE_DIR`.

## Repository layout

| Path | Contents |
| --- | --- |
| [`include/deep_jit/runtime/`](include/deep_jit/runtime/) | Shared runtime and configuration. |
| [`include/deep_jit/backend/cuda/`](include/deep_jit/backend/cuda/) | CUDA compiler, device queries, kernel loading, and launch options. |
| [`include/deep_jit/backend/maca/`](include/deep_jit/backend/maca/) | MACA compiler, device queries, kernel loading, and launch options. |
| [`include/deep_jit/backend/ascend/`](include/deep_jit/backend/ascend/) | Ascend compiler/linker integration, device queries, kernel loading, and launch options. |
| [`include/deep_jit/cache/`](include/deep_jit/cache/) | In-memory and on-disk kernel caches. |
| [`include/deep_jit/python_api.hpp`](include/deep_jit/python_api.hpp) | pybind11 registration for a consumer library's runtime. |
| [`tests/`](tests/) | CUDA, MACA, and Ascend integration tests, example extensions, and device kernels. |

The root `CMakeLists.txt` is for debugging and IDE indexing. Integrate the headers into your own extension as described below; the projects under [`tests/test_cuda_proj/`](tests/test_cuda_proj/) and [`tests/test_ascend_proj/`](tests/test_ascend_proj/) provide working pybind11 integration examples, and [`tests/test_maca_proj/`](tests/test_maca_proj/) provides a standalone host binary that drives the same runtime without embedding Python.

## Integration

GIL management is enabled by default and includes pybind11. Consumers that manage
the GIL themselves, such as kernels called through `torch.ops`, can compile with
`-DDJ_DISABLE_GIL=1` to make `deep_jit::GilScopedRelease` a no-op without including
pybind11 or the Python C API from this helper. In CMake, use
`target_compile_definitions(my_target PRIVATE DJ_DISABLE_GIL=1)`. Set the macro
consistently for every source file compiled into the consumer target.

Add `DeepJIT/include` to the include path of the host target, then include:

```cmake
target_include_directories(my_target PRIVATE third-party/deep_jit/include)
```

Include exactly one backend entry header. For CUDA:

```cpp
#include <deep_jit/backend/cuda/backend.hpp>
```

For MACA:

```cpp
#include <deep_jit/backend/maca/backend.hpp>
```

For Ascend:

```cpp
#include <deep_jit/backend/ascend/backend.hpp>
```

The selected header exposes its backend type:

```cpp
using JIT = deep_jit::Runtime<deep_jit::CUDA>;
```

Use `deep_jit::Runtime<deep_jit::MACA>` for the MACA header, or `deep_jit::Runtime<deep_jit::Ascend>` for the Ascend header.

Exactly one backend can be compiled into a translation unit: the CUDA backend requires CUDA 12.4 or newer with the `cuLibrary*` entry points, which the MACA cu-bridge compatibility headers do not provide. The root `CMakeLists.txt` encodes that choice as `DEEP_JIT_PLATFORM` (`cuda` or `maca`) and selects it for `csrc/python_api.cpp`.

`create_lazy_jit` delays construction of the runtime until its first use. This also delays device and compiler discovery:

```cpp
inline auto jit = deep_jit::create_lazy_jit<deep_jit::CUDA>(
    deep_jit::Config("/absolute/path/to/my_library", "MYLIB"));
```

If configuration is only known during library initialization, start with an empty lazy object and assign its factory later:

```cpp
#include <filesystem>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

namespace my_library {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root) {
    const auto library_root_path = std::filesystem::absolute(library_root);
    const auto include_dir = library_root_path / "include";

    jit = deep_jit::create_lazy_jit<deep_jit::CUDA>(
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

Configuration is snapshotted when `Runtime` is constructed. Do not mutate `config`, `backend`, `parser`, `disk_cache`, or `hash_base` afterward. The supported mutable per-runtime settings are `default_compiler_options` and `default_launch_options`.

Compiler, cache, include, and hook paths, as well as free-form compiler flags, are currently passed through a simple shell command and therefore must not contain whitespace or shell metacharacters.

### Registering `get_jit()`

For a pybind11 extension, DeepJIT can register the runtime type and `get_jit()` directly. The host library does not need to implement its own `get_jit` binding:

```cpp
#include <pybind11/pybind11.h>

#include <deep_jit/python_api.hpp>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    deep_jit::register_python_api(module, my_library::jit);
}
```

Python can then obtain the same process-local runtime object:

```python
jit = my_library._C.get_jit()
```

Calling `get_jit()` initializes the lazy runtime if it has not already been initialized.

## CUDA

The CUDA backend requires CUDA headers 12.4 or newer and NVCC 12.9 or newer. It uses NVCC to generate a CUBIN. Loading through `compile()` requires exactly one CUDA kernel; `compile_without_load()` only builds the artifact and does not perform that check.

### Compile and launch

```cpp
const auto kernel = jit->compile("scale", R"(
extern "C" __global__ void scale(float* output, const float* input, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count)
        output[index] = input[index] * 2.0f;
}
)");

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

The compile tag must contain only letters, digits, and underscores. `compile()` compiles on a cache miss, loads the CUBIN, and returns a process-cached `std::shared_ptr<deep_jit::cuda::Kernel>`. `compile_without_load()` only returns the artifact directory.

Unset `deep_jit::cuda::CompilerOptions` fields inherit from the runtime defaults:

```cpp
deep_jit::cuda::CompilerOptions options {
    .optimize_level = "3",
    .fast_math = true,
    .check_no_spills = true,
    .arch = jit->device.get_arch(/* use_arch_family = */ false),
    .extra_nvcc_flags = {"-DMY_OPTION=1"},
    .post_hook = "hooks/hook_1.py",
};

const auto kernel = jit->compile("scale", source, options);
```

`nvcc_flags` replaces the default free-form NVCC flag list; structured options such as `optimize_level` are generated separately. `extra_nvcc_flags` appends per-kernel flags such as `-D` definitions. To extend the runtime defaults, append flags directly to `*jit->default_compiler_options.nvcc_flags` during initialization.

CUDA launch options include the stream, dynamic shared-memory size, grid, block, cluster, cooperative-launch, PDL, and non-portable cluster controls. Unset fields inherit from `jit->default_launch_options`, following the same override model as compiler options. Do not clear the initialized runtime defaults back to `std::nullopt`. Grid and block dimensions are required and must be positive; dynamic shared-memory size cannot be negative. An unset effective stream uses the current PyTorch CUDA stream; an explicitly supplied null stream remains the CUDA default stream.

Only one-dimensional clusters are supported (`cluster_dim.y == cluster_dim.z == 1`). Enabling `nonportable_cluster_size_allowed` sets a persistent CUDA function attribute; later launches with the option disabled do not reset that attribute.

Compile and load kernels before CUDA Graph capture. Launching an already loaded kernel is capture-compatible, including when the stream is inherited from the current PyTorch stream.

Library-wide launch defaults can be changed once during initialization:

```cpp
jit->default_launch_options.enable_pdl = true;
```

Device information is available from `jit->device`:

```cpp
const int num_sms = jit->device.get_num_sms();
const int l2_bytes = jit->device.get_num_l2_cache_bytes();
const int smem_bytes = jit->device.get_num_smem_bytes();
const int64_t clock_rate = jit->device.get_clock_rate();
const auto [major, minor] = jit->device.get_arch_pair();
const std::string family_arch = jit->device.get_arch();
const std::string concrete_arch = jit->device.get_arch(false);
```

A runtime snapshots the current CUDA device when its default architecture is initialized. Use a separate runtime per CUDA device, construct and use it while that device is current, and do not move a loaded kernel between device contexts.

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

`post_hook` inherits from `jit->default_compiler_options` like every other compiler option and is unset by default. Only `std::nullopt` disables it; an empty string is still treated as a configured hook. Once a default hook is set, the current override model cannot disable it for one kernel. The caller must provide a trusted relative path. When set, it runs after NVCC produces the CUBIN and before the artifact is published:

```bash
cd <temporary-artifact-directory>
python <absolute-python-library-root>/hooks/hook_1.py <absolute-cubin-path>
```

The script receives the absolute CUBIN path as its only argument and must modify that file in place. The configured relative `post_hook` path and file-content hash are included in the kernel cache digest; the path is also recorded in `meta.json`. Hook hashes are cached per thread, so restart the process after changing a hook file.

A hook must be deterministic for a given input CUBIN and tracked signature. Imported Python modules, auxiliary files, Python/package versions, environment variables, random state, time, and network data are not discovered automatically; represent every such dependency in `extra_signature` if it can affect output.

### Cache-key and artifact rules

The CUDA cache digest is built, in order, from:

1. `Config::extra_signature`.
2. The hash of the complete `nvcc --version` output.
3. The effective compiler flags returned by `CompilerOptions::get_flags()`. Paths in `Config::include_dirs` are intentionally excluded, while any `-I...` placed directly in `nvcc_flags` or `extra_nvcc_flags` remains part of the digest.
4. The selected `post_hook` path and file-content hash.
5. The parser digest of the source and its tracked include tree.

Each component is prefixed by its fixed-width byte length before it is added to the two-state FNV-1a hash, so boundaries remain unambiguous even for binary strings containing zero bytes. The final digest is a 32-character hexadecimal string. This is a fast cache checksum, not a cryptographic hash.

The compile tag is not part of the digest. A disk entry is stored as:

```text
<cache-root>/cache/<tag>.<digest>/
```

Cache roots must be trusted. A directory carrying a `.committed` marker is treated as a completed artifact and its CUBIN may be loaded directly.

An entry contains `kernel.cu`, `kernel.cubin`, `meta.json`, `.committed`, and optional `kernel.ptx` or `kernel.sass` files. `meta.json` records the NVCC command arguments used in the temporary build directory, config fields, compiler information, and effective compiler options, including the selected `post_hook` path. Temporary paths in that command may no longer exist after publication. It does not record the surrounding `cd`, stderr redirection, post-hook command, or dump commands.

`dump_ptx` and `dump_sass` control extra artifacts but do not affect the cache digest. They only run when compilation actually occurs; a pre-existing cache hit is not rebuilt to add missing dumps.

The cache key assumes the external compiler environment is stable. Variables such as `NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS`, `CPATH`, host-compiler selection, and transitive third-party header changes are not discovered automatically. Avoid such implicit inputs or represent them in `extra_signature`/explicit compiler flags. Excluding tracked include-directory paths is safe only when moving the same header contents does not change compilation behavior such as embedded `__FILE__` strings.

### Include parser

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

### Environment variables

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
| `JIT_DEBUG` | `0` | Enables compiler-command and load diagnostics, and each backend's own extra reporting: CUDA adds PTXAS output, line info and PTX/SASS dumps, MACA adds the resource report, line info and the assembly dump. |
| `JIT_NVCC_COMPILER` | `<discovered-toolkit>/bin/nvcc` | Overrides the NVCC executable (CUDA). |
| `JIT_MXCC_COMPILER` | `<discovered-toolkit>/mxgpu_llvm/bin/mxcc` | Overrides the mxcc executable (MACA). `llvm-nm` must exist next to it. |
| `JIT_LLVM_NM` | `<mxcc-directory>/llvm-nm` | Overrides the `llvm-nm` used for MACA kernel-name discovery. |
| `JIT_CPP_STANDARD` | `20` | Selects the C++ standard passed to the device compiler as `-std=c++<value>`. |
| `JIT_KERNEL_DEBUG_INFO` | `0` | Adds Bisheng kernel debug information (Ascend). |
| `JIT_LAUNCH_TIMEOUT` | `300` | Sets the Ascend kernel launch timeout in seconds; `0` disables it. |
| `JIT_PRINT_COMPILER_COMMAND` | `0` | Prints compiler and disassembler commands. |
| `JIT_PTXAS_VERBOSE` | `0` | Adds verbose PTXAS output and prints it after compilation (CUDA); on MACA it adds `-resource-usage` and prints the report. |
| `JIT_CHECK_NO_SPILLS` | `0` | Adds `--warn-on-spills` and rejects register spills (CUDA); on MACA it adds `-resource-usage` and rejects a non-empty stack frame. |
| `JIT_CHECK_NO_LOCAL_MEMORY` | `0` | Adds `--warn-on-local-memory-usage` and rejects any local-memory usage (CUDA); on MACA it rejects a non-empty stack frame, the only resource figure mxcc reports. |
| `JIT_PRINT_LOAD_TIME` | `0` | Prints kernel-binary loading time. |
| `JIT_WITH_LINEINFO` | `0` | Adds source line information. |
| `JIT_DUMP_ASM` | `0` | Generates CUDA PTX/SASS, MACA assembly, or Ascend assembly artifacts on a cache miss. |
| `JIT_DUMP_PTX` | `0` | Generates a PTX artifact on a cache miss (CUDA); MACA treats it as `JIT_DUMP_ASM`. |
| `JIT_DUMP_SASS` | `0` | Generates a SASS artifact on a cache miss (CUDA only). |

CUDA toolkit and cache discovery also use these standard environment variables:

| Variable | Behavior |
| --- | --- |
| `HOME` | Required for the default `$HOME/.dj` cache path. |
| `CUDA_HOME` | First CUDA toolkit-root candidate. |
| `CUDA_PATH` | CUDA toolkit-root fallback when `CUDA_HOME` is unset or empty. |
| `PATH` | Used by `which nvcc` when neither CUDA root variable identifies a toolkit. |

If both CUDA root variables are unset or empty and `which nvcc` fails, DeepJIT tries `/usr/local/cuda`. A non-empty but invalid `CUDA_HOME` or `CUDA_PATH` is treated as an error rather than skipped.

`JIT_NVCC_COMPILER` overrides the executable after CUDA-home discovery. A valid CUDA root must still be discoverable through `CUDA_HOME`, `CUDA_PATH`, `PATH`, or `/usr/local/cuda`. SASS dumping additionally requires an executable `cuobjdump` under that discovered toolkit root.

## MACA

The MACA backend requires an MXMACA install that provides `mxgpu_llvm/bin/mxcc` (version 1.0 or newer) and the `llvm-nm` beside it. It compiles `kernel.cu` to `kernel.devbin` with `mxcc -device-bin`, and loads that pre-linked device binary through `mcModuleLoad`. Loading requires exactly one kernel; `compile_without_load()` only builds the artifact and does not perform that check.

MACA has no driver-side kernel enumeration (`mcModuleGetFunctionCount` and the other enumeration entry points are absent from `libmcruntime`), so the entry-point name is recovered from the device binary itself with the toolchain's `llvm-nm`. `mxcc`'s `-fatbin` bundle and `-fgpu-rdc --device-bc` bitcode form are both loadable, but neither can be read by `llvm-nm`, so `-device-bin` is the one artifact that serves both the discovery and the load path.

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

Unset `deep_jit::maca::CompilerOptions` fields inherit from the runtime defaults, which are `--offload-arch=xcore<family>`, `-O3`, and `-std=c++20`:

```cpp
deep_jit::maca::CompilerOptions options {
    .optimize_level = "3",
    .fast_math = true,
    .arch = jit->device.get_arch(),
    .extra_mxcc_flags = {"-DMY_OPTION=1"},
    .post_hook = "hooks/hook_1.py",
};
```

`mxcc_flags` replaces the default free-form mxcc flag list; structured options such as `optimize_level` are generated separately. `extra_mxcc_flags` appends per-kernel flags such as `-D` definitions. The optimization level, fast math (`-use-fast-math`), the resource report (`-resource-usage`), the spill and local-memory checks, line information (`with_line_info`), the assembly dump, the architecture, and the post hook are all supported. There is no MACA counterpart for CUDA's `ptxas_register_usage_level`, and no PTX or SASS dump axis.

The MACA structure names its own toolchain's concepts, like the Ascend one does, so the resource report is `compiler_verbose` (`JIT_PTXAS_VERBOSE` is the shared name of the environment variable behind it). `ptxas_verbose` is accepted as an input alias of `compiler_verbose` so that an option set written for the CUDA backend ports over unchanged; it is an override-only alias -- the defaults never set it, and `compiler_verbose` wins if one override carries both -- so `meta.json`, the flags and the cache digest carry a single spelling.

`check_no_spills` and `check_no_local_memory` both reject a non-empty mxcc stack-frame report. mxcc reports one frame-size figure per function and has no separate local-memory line, so the two checks coincide on MACA; enabling either one also adds the `-resource-usage` flag, which is what makes mxcc emit the report in the first place.

MACA launch options include the stream, dynamic shared-memory size, grid, block, cluster (`cluster_dim`), cooperative-launch, and PDL (`enable_pdl`) controls. Unset fields inherit from `jit->default_launch_options`. Grid and block dimensions are required and must be positive, and the dynamic shared-memory size cannot be negative. An unset effective stream uses the current PyTorch stream.

Two of those controls are not implemented by the MACA runtime: **cluster launch** and **programmatic dependent launch** are rejected with a message naming the unsupported axis, before the driver is reached, because `mcModuleLaunchKernelEx` refuses every launch attribute other than the cooperative flag with `mcErrorInvalidConfiguration`. Only one-dimensional clusters are accepted as an option at all, and `nonportable_cluster_size_allowed` has no MACA counterpart. Dynamic shared memory does not need the maximum-dynamic-shared-memory attribute that CUDA sets: the launch enforces the `sharedMemPerBlockOptin` ceiling itself, and a request above it is refused at launch.

Device information is available from `jit->device`:

```cpp
const int num_sms = jit->device.get_num_sms();
const int l2_bytes = jit->device.get_num_l2_cache_bytes();
const int smem_bytes = jit->device.get_num_smem_bytes();
const int64_t clock_rate = jit->device.get_clock_rate();
const auto [major, minor] = jit->device.get_arch_pair();
const std::string family_arch = jit->device.get_arch();
```

`get_family()` and `get_arch()` map the native device major to the xcore family that names the mxcc offload target: `10`, `15`, and `16` to `xcore1000`, `xcore1500`, and `xcore1600` (C500/C600/C600U). The minor revision is a revision inside the family and is not a judgment axis; an unknown major is an error rather than a speculative mapping.

### Post hook

`CompilerOptions::post_hook` works as it does for CUDA, with the device binary in place of the CUBIN:

```bash
cd <temporary-artifact-directory>
python <absolute-python-library-root>/hooks/hook_1.py <absolute-device-binary-path>
```

The hook runs after mxcc produces `kernel.devbin` and before the artifact is published, so it must leave that binary loadable if the same artifact is to be loaded afterwards. The configured path and file-content hash are part of the kernel cache digest, and the path is recorded in `meta.json`.

### Cache-key and artifact rules

The MACA cache digest is built, in order, from:

1. `Config::extra_signature`.
2. The hash of the complete `mxcc --version` output.
3. The effective compiler flags returned by `CompilerOptions::get_flags()`. Paths in `Config::include_dirs` are intentionally excluded, while any `-I...` placed directly in `mxcc_flags` or `extra_mxcc_flags` remains part of the digest.
4. The selected `post_hook` path and file-content hash.
5. The parser digest of the source and its tracked include tree.

The compile tag is not part of the digest. A disk entry is stored as `<cache-root>/cache/<tag>.<digest>/`, carries a `.committed` marker, and contains `kernel.cu`, `kernel.devbin`, and `meta.json`, plus `kernel.s` when `dump_asm` is set. `meta.json` records the config fields, the compiler and its version, the effective compiler options, and the mxcc command arguments used in the temporary build directory.

`dump_asm` writes the device assembly with `-aop -S --device-obj` and does not affect the cache digest; like the CUDA dumps, it only runs when compilation actually occurs. `JIT_DUMP_PTX` and `JIT_PTXAS_VERBOSE` are accepted as aliases for the MACA assembly dump and resource report respectively, and `JIT_DUMP_SASS` has no effect.

### Environment variables

MACA toolkit discovery checks `MACA_HOME`, `MACA_PATH`, `CUDA_HOME`, and `CUDA_PATH` in that order -- the first non-empty one wins, and the cu-bridge root also carries the `mxgpu_llvm` toolchain as a sibling -- then falls back to `/opt/maca`. The compiler is `<home>/mxgpu_llvm/bin/mxcc`, and `JIT_MXCC_COMPILER` overrides it; `llvm-nm` is resolved next to the selected `mxcc` unless `JIT_LLVM_NM` overrides it. The four install roots name an installation rather than a library setting, so like `CUDA_HOME`/`CUDA_PATH` on the CUDA backend they are read unprefixed; the `JIT_MXCC_COMPILER` and `JIT_LLVM_NM` overrides do follow the library prefix.

## Ascend

The Ascend backend requires ACL and torch_npu headers plus a CANN toolkit containing `bin/bisheng` and `bin/ld.lld`. It compiles `kernel.asc` to `kernel.rel.o`, links `kernel.o`, parses the unique `.ascend.meta.*` kernel name, and loads it through ACL.

```cpp
inline auto jit = deep_jit::create_lazy_jit<deep_jit::Ascend>(
    deep_jit::Config(
        "/absolute/path/to/my_library",
        "MYLIB",
        {},
        {"/absolute/path/to/my_library/include"},
        {"my_library/"}));

const auto kernel = jit->compile("scale", source);
jit->launch(kernel, {.num_blocks = num_blocks}, output, input, count);
```

An unset stream uses the current torch_npu stream. `num_blocks` is required. `num_ubuf_bytes` controls dynamic UB size, and `num_launch_timeout_secs` defaults to `JIT_LAUNCH_TIMEOUT`.

When `ASCEND_LAUNCH_BLOCKING` is enabled, DeepJIT synchronizes the device after every launch.

`deep_jit::ascend::CompilerOptions` supports the optimization level, `dav-*` architecture, debug information, assembly dumping, and replace/append lists for Bisheng and linker flags. The defaults are `-O2`, `--cce-aicore-only`, VF loop unrolling, and `ld.lld -m aicorelinux -Ttext 0 --no-mmap-output-file`. `<toolkit>/aarch64-linux/asc/include/adv_api` is required and added automatically.

Assembly dumping writes Bisheng saved intermediates under the artifact's `asm/` directory. Like CUDA's PTX/SASS dumps, it does not change the cache digest and only runs on a cache miss.

Device queries are available through `jit->device`, including `get_npu_arch()`, `get_num_sms()`/`get_num_aicore_cores()`, vector and cube core counts, UB size, and L2 size.

Toolkit discovery checks `ASCEND_HOME_PATH`, `ASCEND_TOOLKIT_HOME`, `/usr/local/Ascend/ascend-toolkit/latest`, and `/usr/local/Ascend/cann`, in that order.
