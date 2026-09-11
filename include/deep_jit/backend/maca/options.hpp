#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mc_runtime.h>

#include <deep_jit/backend/maca/device.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/format.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/str.hpp>

namespace deep_jit::maca {

// Per-compile mxcc options. Unset fields inherit from backend defaults.
struct CompilerOptions {
    std::optional<std::string> optimize_level;
    std::optional<bool> fast_math;
    std::optional<bool> compiler_verbose;
    std::optional<bool> check_no_spills;
    std::optional<bool> check_no_local_memory;
    std::optional<bool> with_line_info;
    std::optional<bool> dump_asm;
    std::optional<std::string> arch;
    std::optional<std::vector<std::string>> mxcc_flags;
    std::vector<std::string> extra_mxcc_flags;
    std::optional<std::string> post_hook;

    static CompilerOptions default_options(const Env& env, Device& device) {
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool dump_asm = debug or env.get<bool>("JIT_DUMP_ASM", false);
        return {
            .optimize_level = "3",
            .fast_math = false,
            .compiler_verbose = debug or env.get<bool>("JIT_PTXAS_VERBOSE", false),
            .check_no_spills = env.get<bool>("JIT_CHECK_NO_SPILLS", false),
            .check_no_local_memory = env.get<bool>("JIT_CHECK_NO_LOCAL_MEMORY", false),
            .with_line_info = debug or env.get<bool>("JIT_WITH_LINEINFO", false),
            .dump_asm = dump_asm or env.get<bool>("JIT_DUMP_PTX", false),
            .arch = device.get_arch(),
            .mxcc_flags = std::vector<std::string> {
                fmt::format("-std=c++{}", env.get<int>("JIT_CPP_STANDARD", 20)),
            },
            .post_hook = std::nullopt
        };
    }

    [[nodiscard]] CompilerOptions override_with(const CompilerOptions& overrides) const {
        CompilerOptions result = *this;
        if (overrides.optimize_level)
            result.optimize_level = overrides.optimize_level;
        if (overrides.fast_math)
            result.fast_math = overrides.fast_math;
        if (overrides.compiler_verbose)
            result.compiler_verbose = overrides.compiler_verbose;
        if (overrides.check_no_spills)
            result.check_no_spills = overrides.check_no_spills;
        if (overrides.check_no_local_memory)
            result.check_no_local_memory = overrides.check_no_local_memory;
        if (overrides.with_line_info)
            result.with_line_info = overrides.with_line_info;
        if (overrides.dump_asm)
            result.dump_asm = overrides.dump_asm;
        if (overrides.arch)
            result.arch = overrides.arch;
        if (overrides.mxcc_flags)
            result.mxcc_flags = overrides.mxcc_flags;
        result.extra_mxcc_flags.insert(result.extra_mxcc_flags.end(), overrides.extra_mxcc_flags.begin(), overrides.extra_mxcc_flags.end());

        if (overrides.post_hook)
            result.post_hook = overrides.post_hook;
        return result;
    }

    [[nodiscard]] std::vector<std::string> get_flags() const {
        // Arch (the xcore offload target)
        DJ_HOST_ASSERT(arch.has_value() and not arch->empty(), "MACA architecture must be specified");
        std::vector flags = {"--offload-arch=xcore" + *arch};

        // Optimization level
        DJ_HOST_ASSERT(optimize_level.has_value() and not optimize_level->empty(), "optimization level must be specified");
        flags.emplace_back("-O" + *optimize_level);

        // Global fast-math
        if (fast_math.value_or(false))
            flags.emplace_back("-use-fast-math");

        // Print per-function resource usage (the mxcc analogue of PTXAS -v).
        // mxcc emits the "N bytes stack frame" line that the spill and
        // local-memory checks read, and emits it only under this flag, so a
        // check that validates that report has to request it too.
        if (compiler_verbose.value_or(false) or check_no_spills.value_or(false) or
            check_no_local_memory.value_or(false))
            flags.emplace_back("-resource-usage");

        // Line info for debugging
        if (with_line_info.value_or(false))
            flags.emplace_back("--generate-line-info");

        // Other flags
        if (mxcc_flags)
            flags.insert(flags.end(), mxcc_flags->begin(), mxcc_flags->end());
        flags.insert(flags.end(), extra_mxcc_flags.begin(), extra_mxcc_flags.end());
        return flags;
    }

    // mxcc takes a single input file and emits a single device artifact;
    // it has no `nvcc --cubin` and no `cuobjdump --dump-sass` counterpart,
    // so there is no SASS dump axis.
    void update_hash(hash::FNV1a& hasher) const {
        hasher.update(str::join(get_flags()));
    }

    [[nodiscard]] std::string get_post_hook_hash(const Config& config) const {
        if (not post_hook)
            return "";

        // Check cache hit
        thread_local std::unordered_map<std::string, std::string> post_hook_hashes;
        const auto path = config.get_python_path(*post_hook);
        if (const auto iterator = post_hook_hashes.find(path); iterator != post_hook_hashes.end())
            return iterator->second;

        // Miss: update hash
        const auto digest = hash::FNV1a()
            .update(*post_hook).update(read(path))
            .get_hex_digest();
        return post_hook_hashes.emplace(path, digest).first->second;
    }

    [[nodiscard]] json to_json() const {
        return json::object_t {
            {"optimize_level", optimize_level},
            {"fast_math", fast_math},
            {"compiler_verbose", compiler_verbose},
            {"check_no_spills", check_no_spills},
            {"check_no_local_memory", check_no_local_memory},
            {"with_line_info", with_line_info},
            {"dump_asm", dump_asm},
            {"arch", arch},
            {"mxcc_flags", mxcc_flags},
            {"extra_mxcc_flags", extra_mxcc_flags},
            {"post_hook", post_hook},
        };
    }
};

// Per-launch MACA options. Unset fields inherit from runtime defaults.
struct LaunchOptions {
    // Use current stream if unset
    std::optional<mcStream_t> stream;

    // Dynamic shared memory size
    std::optional<int> num_smem_bytes;

    // Grid
    std::optional<dim3> grid_dim;
    std::optional<dim3> block_dim;
    std::optional<dim3> cluster_dim;

    // Cooperative launch
    std::optional<bool> cooperative;

    // Dependent kernel launch
    std::optional<bool> enable_pdl;

    // Others
    std::optional<bool> nonportable_cluster_size_allowed;

    static LaunchOptions default_options(const Env&) {
        return {
            .stream = std::nullopt,
            .num_smem_bytes = 0,
            .grid_dim = std::nullopt,
            .block_dim = std::nullopt,
            .cluster_dim = dim3(1, 1, 1),
            .cooperative = false,
            .enable_pdl = false,
            .nonportable_cluster_size_allowed = false,
        };
    }

    [[nodiscard]] LaunchOptions override_with(const LaunchOptions& overrides) const {
        LaunchOptions result = *this;
        if (overrides.stream)
            result.stream = overrides.stream;
        if (overrides.num_smem_bytes)
            result.num_smem_bytes = overrides.num_smem_bytes;
        if (overrides.grid_dim)
            result.grid_dim = overrides.grid_dim;
        if (overrides.block_dim)
            result.block_dim = overrides.block_dim;
        if (overrides.cluster_dim)
            result.cluster_dim = overrides.cluster_dim;
        if (overrides.cooperative)
            result.cooperative = overrides.cooperative;
        if (overrides.enable_pdl)
            result.enable_pdl = overrides.enable_pdl;
        if (overrides.nonportable_cluster_size_allowed)
            result.nonportable_cluster_size_allowed = overrides.nonportable_cluster_size_allowed;
        return result;
    }
};

}  // namespace deep_jit::maca
