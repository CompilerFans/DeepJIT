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
        // `llvm-nm` from the same toolchain: MACA has no driver-side kernel
        // enumeration, so entry-point discovery reads the device artifact.
        std::filesystem::path llvm_nm;
    };

    Toolkit toolkit;
    CompilerInfo compiler_info;

    explicit MACA(const Env& env)
        : toolkit(find_maca_toolkit(env)),
          compiler_info(get_compiler_info()) {
    }

    [[nodiscard]] CompilerInfo get_compiler_info() const {
        const auto version = call_external_command(toolkit.mxcc.string() + " --version");

        // `mxcc version 1.0.0 (d9102a1572)`
        std::smatch match;
        DJ_HOST_ASSERT(std::regex_search(version, match, std::regex(R"(mxcc version (\d+)\.(\d+))")),
                       "failed to parse mxcc version from:\n{}", version);

        // Minimum toolchain version, in the shape CUDA uses for NVCC.  1.0 is
        // the SUPPORTED FLOOR, not a measured feature boundary: it is the
        // toolchain this backend is validated against (MACA 3.7.0 ships
        // `mxcc version 1.0.0`), and older MACA releases are out of scope
        // rather than merely untested.  It is a check at all because a
        // too-old mxcc otherwise surfaces as an opaque compiler error or as
        // the missing-artifact assert below, never as a version statement.
        constexpr int minimum_major = 1;
        constexpr int minimum_minor = 0;
        const int major = std::stoi(match[1].str());
        const int minor = std::stoi(match[2].str());
        DJ_HOST_ASSERT(major > minimum_major or (major == minimum_major and minor >= minimum_minor),
                       "mxcc version must be at least {}.{}", minimum_major, minimum_minor);

        return { .path = toolkit.mxcc, .version = version };
    }

    // The generated flags + includes, shared by the binary and assembly
    // passes so the two can never drift apart.
    [[nodiscard]] std::vector<std::string> make_common_args(const Config& config,
                                                            const CompilerOptions& options) const {
        std::vector<std::string> args;
        const auto option_flags = options.get_flags();
        args.insert(args.end(), option_flags.begin(), option_flags.end());
        for (const auto& include_dir: config.include_dirs)
            args.emplace_back("-I" + include_dir.string());
        return args;
    }

    void compile(const std::string& source,
                 std::filesystem::path dir,
                 const Env& env,
                 const Config& config,
                 const CompilerOptions& options) const {
        // Release GIL to let other Python threads run
        [[maybe_unused]] GilScopedRelease gil_release;

        // Paths
        dir = std::filesystem::absolute(dir).lexically_normal();
        const auto source_path = dir / "kernel.cu";
        const auto binary_path = dir / "kernel.devbin";
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_compiler_command = debug or env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);

        // Write source code
        write_file_sync(source_path, source);

        // Build the full command.  mxcc has no `--cubin`; the pre-linked
        // device-only binary is `-device-bin` (already loadable by
        // `mcModuleLoad` without a device-side link step at load time).
        // NOTE: the alternative `-fatbin` bundle is also loadable but is an
        // opaque container that `llvm-nm` cannot read, and the
        // `-fgpu-rdc --device-bc` bitcode form defers the device link to
        // load time -- neither can serve both the symbol-discovery and the
        // load path, so `-device-bin` is the one artifact that does.
        std::vector<std::string> args = {
            toolkit.mxcc.string(),
            source_path.string(),
            "-x",
            "maca",
            "-device-bin",
            "-o",
            binary_path.string(),
        };
        const auto common_args = make_common_args(config, options);
        args.insert(args.end(), common_args.begin(), common_args.end());
        const auto command = str::join(args);

        // NOTES: change directory into a temporary empty directory to prevent same name include files
        const auto cd_command = "cd " + dir.string() + " && ";

        // Compile
        const auto compiler_output = call_external_command(cd_command + command, print_compiler_command);
        if (options.compiler_verbose.value_or(false)) {
            std::fputs(compiler_output.c_str(), stdout);
            std::fflush(stdout);
        }
        // `-resource-usage` reports the frame size per function as
        // "N bytes stack frame"; a non-zero value is the spill signal.  mxcc
        // has no second figure: unlike PTXAS there is no local-memory line at
        // all (measured on a kernel with an 8 KiB frame, mxcc's whole report
        // is "N bytes stack frame", the register counts and the shared-memory
        // count), so `check_no_local_memory` validates that same report and
        // the two checks coincide on MACA.  Both need `-resource-usage` to be
        // present at all -- see `CompilerOptions::get_flags`.
        if (options.check_no_spills.value_or(false) or options.check_no_local_memory.value_or(false)) {
            std::smatch match;
            const auto text = compiler_output;
            for (auto begin = text.cbegin(); std::regex_search(begin, text.cend(), match,
                                                               std::regex(R"((\d+) bytes stack frame)"));) {
                DJ_HOST_ASSERT(std::stoi(match[1].str()) == 0,
                               "mxcc reported a non-empty stack frame:\n{}", compiler_output);
                begin = match.suffix().first;
            }
        }
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

        // Dump device assembly.  `-aop -S --device-obj` is the emitting form;
        // a bare `-S` is rejected by mxcc.
        if (options.dump_asm.value_or(false)) {
            const auto asm_path = dir / "kernel.s";
            std::vector<std::string> asm_args = {
                toolkit.mxcc.string(),
                source_path.string(),
                "-x",
                "maca",
                "-aop",
                "-S",
                "--device-obj",
                "-o",
                asm_path.string(),
            };
            asm_args.insert(asm_args.end(), common_args.begin(), common_args.end());
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

    // `Runtime` invokes this statically as `Backend::load(dir, env)`, and the
    // toolkit is resolved from that same `env` here rather than carried over
    // from the constructor.  Unlike CUDA's load, this one needs a path out of
    // the toolkit (`llvm-nm`), so it cannot simply be a pass-through -- but a
    // process-global would let one Runtime's toolkit silently decide another
    // Runtime's loads, and would race if two Runtimes were constructed
    // concurrently.  Resolving per load costs a couple of `stat` calls against
    // a `dlopen` + `mcModuleLoad`.
    [[nodiscard]] static std::shared_ptr<Kernel> load(const std::filesystem::path& dir, const Env& env) {
        return Kernel::load(dir, env, find_maca_toolkit(env).llvm_nm);
    }

    static Toolkit find_maca_toolkit(const Env& env) {
        // The lookup ladder mirrors the MACA SDK's own `cucc` wrapper and the
        // host project's documented order: an explicit setting is
        // authoritative, then the compiler bin directory (`MACA_CLANG_PATH`),
        // then the install roots (`MACA_PATH`, then the cu-bridge
        // `CUDA_HOME`/`CUDA_PATH`, whose sibling layout also carries the
        // `mxgpu_llvm` toolchain), then `mxcc` on PATH, then the default
        // `/opt/maca` install.  `MACA_HOME` is deliberately NOT consulted: no
        // MXMACA install defines it, and an unrelated variable of that name
        // would shadow the real settings.
        std::filesystem::path mxcc;
        if (const auto path = env.get<std::string>("JIT_MXCC_COMPILER"); path and not path->empty()) {
            mxcc = std::filesystem::absolute(*path).lexically_normal();
        } else if (const auto clang_path = get_env<std::string>("MACA_CLANG_PATH"); not clang_path.empty()) {
            // `MACA_CLANG_PATH` names the bin directory itself (commonly
            // exported with a trailing slash); like the install roots it
            // names an installation rather than a library setting, so it is
            // read unprefixed.
            mxcc = std::filesystem::absolute(std::filesystem::path(clang_path) / "mxcc").lexically_normal();
        } else {
            std::filesystem::path home_path;
            for (const auto* name: {"MACA_PATH", "CUDA_HOME", "CUDA_PATH"}) {
                home_path = get_env<std::string>(name);
                if (not home_path.empty())
                    break;
            }
            if (home_path.empty()) {
                // `mxcc` on PATH, with symlinks resolved so the sibling
                // `llvm-nm` is looked up next to the real file (`/usr/bin/mxcc`
                // is typically a symlink into the SDK).
                try {
                    auto path = call_external_command("which mxcc");
                    while (not path.empty() and (path.back() == '\r' or path.back() == '\n'))
                        path.pop_back();
                    if (not path.empty()) {
                        std::error_code error;
                        const auto canonical = std::filesystem::canonical(path, error);
                        mxcc = error ? std::filesystem::path(path) : canonical;
                    }
                } catch (...) {}
            }
            if (mxcc.empty()) {
                if (home_path.empty() and std::filesystem::exists("/opt/maca"))
                    home_path = "/opt/maca";
                // A home that is set but does not exist is an error, not a
                // reason to try the next name down: a typo must not silently
                // select a different install.
                DJ_HOST_ASSERT(not home_path.empty() and std::filesystem::exists(home_path),
                               "MACA toolkit home was not found");
                home_path = std::filesystem::absolute(home_path).lexically_normal();
                mxcc = home_path / "mxgpu_llvm/bin/mxcc";
            }
        }
        DJ_HOST_ASSERT(is_executable(mxcc), "mxcc compiler is not executable: {}", mxcc.string());

        // `llvm-nm` sits next to mxcc in the same toolchain
        std::filesystem::path llvm_nm;
        if (const auto path = env.get<std::string>("JIT_LLVM_NM"); path and not path->empty()) {
            llvm_nm = std::filesystem::absolute(*path).lexically_normal();
        } else {
            llvm_nm = mxcc.parent_path() / "llvm-nm";
        }
        DJ_HOST_ASSERT(is_executable(llvm_nm), "llvm-nm is not executable: {}", llvm_nm.string());

        return {.mxcc = std::move(mxcc), .llvm_nm = std::move(llvm_nm)};
    }
};

}  // namespace deep_jit
