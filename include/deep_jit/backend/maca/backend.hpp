#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <deep_jit/backend/maca/device.hpp>
#include <deep_jit/backend/maca/kernel.hpp>
#include <deep_jit/backend/maca/options.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/runtime/runtime.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/str.hpp>

namespace deep_jit {

class MACA {
public:
    using Device = maca::Device;
    using Kernel = maca::Kernel;
    using CompilerOptions = maca::CompilerOptions;
    using LaunchOptions = maca::LaunchOptions;

    struct CompilerInfo {
        std::filesystem::path path;
        std::string version;

        [[nodiscard]] std::string get_hash() const {
            return hash::get_hex_digest(version);
        }

        [[nodiscard]] json to_json() const {
            return json::object_t {
                {"path", path.string()},
                {"version", version},
            };
        }
    };

    struct Toolkit {
        std::filesystem::path mxcc;
    };

    Toolkit toolkit;
    CompilerInfo compiler_info;

    explicit MACA(const Env& env)
        : toolkit(find_maca_toolkit(env)),
          compiler_info(get_compiler_info()) {}

    [[nodiscard]] CompilerInfo get_compiler_info() const {
        const auto version = call_external_command(toolkit.mxcc.string() + " --version");

        // `mxcc version 1.0.0 (d9102a1572)`
        std::smatch match;
        DJ_HOST_ASSERT(std::regex_search(version, match, std::regex(R"(mxcc version (\d+)\.(\d+))")),
                       "failed to parse mxcc version from:\n{}", version);
        return { .path = toolkit.mxcc, .version = version };
    }

    void compile(const std::string& source,
                 std::filesystem::path dir,
                 const Env& env,
                 const Config& config,
                 const CompilerOptions& options) const {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Paths
        dir = std::filesystem::absolute(dir).lexically_normal();
        const auto source_path = dir / "kernel.cu";
        const auto binary_path = dir / "kernel.mcfb";
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_compiler_command = debug or env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);

        // Write source code
        write_file_sync(source_path, source);

        // Build the full command.  `-fatbin` emits the device-only mcfb
        // bundle that `mcModuleLoad` consumes without a device-side JIT step
        // (the `.bc` form is also loadable but pays the JIT at load time, so
        // the cached artifact is deliberately the pre-linked one).
        std::vector<std::string> args = {
            toolkit.mxcc.string(),
            source_path.string(),
            "-x",
            "maca",
            "-fatbin",
            "-o",
            binary_path.string(),
        };
        const auto option_flags = options.get_flags();
        args.insert(args.end(), option_flags.begin(), option_flags.end());
        for (const auto& include_dir: config.include_dirs) {
            args.emplace_back("-I" + include_dir.string());
        }
        const auto command = str::join(args);

        // NOTES: change directory into a temporary empty directory to prevent same name include files
        const auto cd_command = "cd " + dir.string() + " && ";

        // Compile
        const auto compiler_output = call_external_command(cd_command + command, print_compiler_command);
        if (options.compiler_verbose.value_or(false)) {
            std::fputs(compiler_output.c_str(), stdout);
            std::fflush(stdout);
        }
        DJ_HOST_ASSERT(not options.check_no_spills.value_or(false) or
                       not std::regex_search(compiler_output, std::regex(R"(bytes stack frame)", std::regex::icase)) or
                       std::regex_search(compiler_output, std::regex(R"(0 bytes stack frame)", std::regex::icase)),
                       "mxcc reported a non-empty stack frame:\n{}",
                       compiler_output);
        DJ_HOST_ASSERT(not options.check_no_local_memory.value_or(false) or
                       not std::regex_search(compiler_output, std::regex(R"(local memory)", std::regex::icase)),
                       "mxcc reported local memory usage:\n{}",
                       compiler_output);
        DJ_HOST_ASSERT(std::filesystem::is_regular_file(binary_path) and std::filesystem::file_size(binary_path) != 0,
                       "mxcc did not produce a valid device binary: {}",
                       binary_path.string());

        // Run post hook
        if (options.post_hook) {
            const auto hook_path = config.get_python_path(*options.post_hook);
            const auto hook_command = "cd " + dir.string() +
                                      " && python " + hook_path.string() + " " + binary_path.string();
            call_external_command(hook_command, print_compiler_command);
        }

        // Dump device assembly.  mxcc has no `--ptx`/`--cubin` pair; the
        // device assembly is emitted by the `-aop -S --device-obj` form.
        if (options.dump_asm.value_or(false)) {
            const auto asm_path = dir / "kernel.s";
            auto asm_args = args;
            asm_args[4] = "--device-obj";
            asm_args.insert(asm_args.begin() + 4, "-aop");
            asm_args.insert(asm_args.begin() + 5, "-S");
            asm_args[7] = asm_path.string();
            call_external_command(cd_command + str::join(asm_args), print_compiler_command);
            DJ_HOST_ASSERT(std::filesystem::is_regular_file(asm_path) and std::filesystem::file_size(asm_path) != 0,
                           "mxcc did not produce a valid device assembly: {}", asm_path.string());
        }

        // Write metadata
        const json metadata = json::object_t {
            {"command", command},
            {"config", config.to_json()},
            {"compiler_info", compiler_info.to_json()},
            {"compiler_options", options.to_json()},
        };
        write_file_sync(dir / "meta.json", metadata.dump());
    }

    [[nodiscard]] static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        return Kernel::load(dir, env);
    }

    static Toolkit find_maca_toolkit(const Env& env) {
        // Find the MACA home.  `MACA_PATH`/`MACA_HOME` are the install-root
        // spelling the host project and develop.sh use; `CUDA_HOME`/
        // `CUDA_PATH` resolve to the cu-bridge root, whose sibling install
        // layout also holds `mxgpu_llvm/bin/mxcc`, so accept them too.
        std::filesystem::path home_path;
        for (const auto* name: {"MACA_HOME", "MACA_PATH", "CUDA_HOME", "CUDA_PATH"}) {
            home_path = get_env<std::string>(name);
            if (not home_path.empty())
                break;
        }
        if (home_path.empty() and std::filesystem::exists("/opt/maca"))
            home_path = "/opt/maca";

        DJ_HOST_ASSERT(not home_path.empty() and std::filesystem::exists(home_path),
                       "MACA toolkit home was not found");
        home_path = std::filesystem::absolute(home_path).lexically_normal();

        // Find mxcc
        std::filesystem::path mxcc;
        if (const auto path = env.get<std::string>("JIT_MXCC_COMPILER"); path and not path->empty()) {
            mxcc = std::filesystem::absolute(*path).lexically_normal();
        } else {
            mxcc = home_path / "mxgpu_llvm/bin/mxcc";
        }
        DJ_HOST_ASSERT(is_executable(mxcc), "mxcc compiler is not executable: {}", mxcc.string());
        return {.mxcc = std::move(mxcc)};
    }
};

}  // namespace deep_jit
