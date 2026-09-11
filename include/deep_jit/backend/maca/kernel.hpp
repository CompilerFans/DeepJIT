#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <mc_runtime.h>

#include <deep_jit/backend/maca/driver.hpp>
#include <deep_jit/backend/maca/options.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/format.hpp>
#include <deep_jit/utils/gil.hpp>
#include <deep_jit/utils/no_ref_ptr.hpp>

namespace deep_jit::maca {

template <typename T>
inline void* kernel_arg_pointer(const T& value) {
    if constexpr (std::is_base_of_v<NoRefPtr, std::decay_t<T>>) {
        return value.ptr;
    } else {
        return const_cast<void*>(static_cast<const void*>(&value));
    }
}

// Immutable MACA kernel handles with shared ownership. Driver resources are
// unloaded when the last shared owner is destroyed.
class Kernel {
public:
    mcModule_t module_handle{};
    mcFunction_t kernel_handle{};

    Kernel(const mcModule_t& module_handle, const mcFunction_t& kernel_handle)
        : module_handle(module_handle), kernel_handle(kernel_handle) {}

    ~Kernel() { unload(); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&&) = delete;
    Kernel& operator=(Kernel&&) = delete;

    // MACA has no driver-side kernel *enumeration*: `mcLibraryGetKernelCount`,
    // `mcLibraryEnumerateKernels`, `mcLibraryLoadFromFile`,
    // `mcKernelGetFunction`, `mcModuleGetFunctionCount` and
    // `mcModuleEnumerateFunctions` are all absent from libmcruntime.so
    // (verified with `nm -D`), so the entry point name is recovered from the
    // device artifact itself with the toolchain's `llvm-nm` -- the same route
    // the host project's kernel JIT takes.  Filtering mirrors that
    // implementation: the toolchain's own bookkeeping symbols and compiler
    // runtime stubs are skipped, leaving the user's `extern "C"` entry point.
    static std::vector<std::string> parse_kernel_names(const std::filesystem::path& llvm_nm,
                                                       const std::filesystem::path& path) {
        const auto output = call_external_command(
            llvm_nm.string() + " --defined-only " + path.string());

        const std::vector<std::string> illegal_names = {
            "llvm.",
            "__cxa_pure_virtual",
            "__cxa_deleted_virtual",
            "__mcImplicitDeviceSynchronize",
            "mcDeviceMemoryInfo",
            ".kd",
        };

        std::vector<std::string> kernel_names;
        std::string line;
        for (std::size_t begin = 0; begin <= output.size();) {
            const auto end = output.find('\n', begin);
            line = output.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            begin = end == std::string::npos ? output.size() + 1 : end + 1;

            // <address> <type> <name>
            std::size_t last_space = line.rfind(' ');
            if (last_space == std::string::npos)
                continue;
            const auto name = line.substr(last_space + 1);
            if (name.empty())
                continue;

            // Only strong/found definitions carry a kernel entry point.
            const auto type_position = line.rfind(' ', last_space - 1);
            const auto type = type_position == std::string::npos
                ? std::string()
                : line.substr(type_position + 1, last_space - type_position - 1);
            if (type != "T" and type != "t")
                continue;

            bool is_illegal = false;
            for (const auto& illegal: illegal_names) {
                if (name.find(illegal) != std::string::npos) {
                    is_illegal = true;
                    break;
                }
            }
            if (not is_illegal)
                kernel_names.emplace_back(name);
        }
        return kernel_names;
    }

    static std::shared_ptr<Kernel> load(const std::filesystem::path& dir,
                                        const Env& env,
                                        const std::filesystem::path& llvm_nm) {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Check existence
        const auto binary_path = dir / "kernel.devbin";
        if (not std::filesystem::is_regular_file(binary_path))
            DJ_PANIC("missing MACA device binary: {}", binary_path.string());

        // Record start time
        const bool debug = env.get<bool>("JIT_DEBUG", false);
        const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
        if (debug)
            std::fputs(fmt::format("Loading MACA device binary: {}\n", binary_path.string()).c_str(), stdout);
        const auto start_time = std::chrono::steady_clock::now();

        // Load kernel
        const auto kernel_names = parse_kernel_names(llvm_nm, binary_path);
        if (kernel_names.size() != 1)
            DJ_PANIC("expected exactly one kernel in {}, found {}: {}",
                     binary_path.string(), kernel_names.size(), str::join(kernel_names));

        mcModule_t module_handle{};
        mcFunction_t kernel_handle{};
        DJ_MACA_DRIVER_CHECK(driver::lazy_mcModuleLoad(&module_handle, binary_path.c_str()));
        try {
            DJ_MACA_DRIVER_CHECK(driver::lazy_mcModuleGetFunction(
                &kernel_handle, module_handle, kernel_names.front().c_str()));
        } catch (...) {
            driver::lazy_mcModuleUnload(module_handle);
            throw;
        }

        // Print and return
        if (print_load_time) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start_time;
            std::fputs(fmt::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(), stdout);
        }
        return std::make_shared<Kernel>(module_handle, kernel_handle);
    }

    template <typename... Args>
    void launch(const LaunchOptions& launch_options, const Args&... args) const {
        // Release GIL to let other Python threads run
        GilScopedRelease gil_release;

        // Checks
        DJ_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
        DJ_HOST_ASSERT(launch_options.num_smem_bytes.has_value(), "MACA dynamic shared-memory size must be specified");
        DJ_HOST_ASSERT(launch_options.grid_dim.has_value(), "MACA grid dimension must be specified");
        DJ_HOST_ASSERT(launch_options.block_dim.has_value(), "MACA block dimension must be specified");
        DJ_HOST_ASSERT(launch_options.cluster_dim.has_value(), "MACA cluster dimension must be specified");
        DJ_HOST_ASSERT(launch_options.cooperative.has_value(), "MACA cooperative option must be specified");
        DJ_HOST_ASSERT(launch_options.enable_pdl.has_value(), "MACA PDL option must be specified");
        DJ_HOST_ASSERT(launch_options.nonportable_cluster_size_allowed.has_value(),
                       "MACA non-portable cluster option must be specified");
        DJ_HOST_ASSERT(*launch_options.num_smem_bytes >= 0, "MACA dynamic shared-memory size must not be negative");
        DJ_HOST_ASSERT(launch_options.grid_dim->x > 0 and launch_options.grid_dim->y > 0 and launch_options.grid_dim->z > 0,
                       "MACA grid dimensions must be positive");
        DJ_HOST_ASSERT(launch_options.block_dim->x > 0 and launch_options.block_dim->y > 0 and launch_options.block_dim->z > 0,
                       "MACA block dimensions must be positive");
        DJ_HOST_ASSERT(launch_options.cluster_dim->x > 0, "MACA cluster dimension must be positive");
        DJ_HOST_ASSERT(launch_options.cluster_dim->y == 1 and launch_options.cluster_dim->z == 1,
                       "only one-dimensional MACA clusters are supported");

        // NOTE: unlike CUDA, MACA needs no `cuFuncSetAttribute` call to raise
        // the dynamic shared-memory ceiling -- measured on C500, a launch of
        // up to `sharedMemPerBlockOptin` (64 KiB) succeeds with the attribute
        // never set, and anything above it is refused at launch with
        // mcErrorInvalidValue ("Shared memory size error ... in device
        // 65536").  The ceiling is enforced by the launch itself, so the
        // maximum-dynamic-shared-memory attribute is deliberately not set:
        // the runtime entry that accepts a `mcFunction_t` is not in
        // libmcruntime.so (the `cuFuncSetAttribute` spelling lives in
        // libsymbol_cu.so, which this backend does not link), and the native
        // `mcFuncSetAttribute` rejects a function handle outright.  Adding
        // the call would buy nothing but a new link dependency.

        // Build the launch attributes.  MACA accepts a *limited* attribute
        // set: only the cooperative flag is honoured, and everything else is
        // refused unconditionally -- measured on C500, the runtime reports
        // "Only Cooperative features are supported, others are not" and
        // returns mcErrorInvalidConfiguration for any other attribute,
        // including a trivial 1x1x1 cluster.  Cluster launch and PDL are
        // therefore rejected up front with a clear message rather than being
        // handed to the driver to fail with an opaque code.  This is a MACA
        // implementation limit, not a configuration that a flag can enable,
        // so the rejection is unconditional (no escape hatch: the only
        // outcome of bypassing it would be the same failure one layer down).
        // `nonportable_cluster_size_allowed` has no MACA counterpart at all
        // (neither a launch attribute nor a function attribute).
        DJ_HOST_ASSERT(launch_options.cluster_dim->x == 1,
                       "MACA does not support cluster launch (mcErrorInvalidConfiguration)");
        DJ_HOST_ASSERT(not *launch_options.enable_pdl,
                       "MACA does not support programmatic dependent launch (mcErrorInvalidConfiguration)");

        // NOTES: please enlarge the array size if you want more attributes
        std::array<mcLaunchAttribute, 1> attributes{};
        unsigned int num_attributes = 0;
        if (*launch_options.cooperative) {
            auto& attribute = attributes[num_attributes ++];
            attribute.id = mcLaunchAttributeCooperative;
            attribute.val.cooperative = 1;
        }

        // Set launch config
        void* kernel_arg_ptrs[sizeof...(Args) + 1] = {kernel_arg_pointer(args)..., nullptr};
        mcLaunchConfigExtension config{};
        config.gridDimX = launch_options.grid_dim->x;
        config.gridDimY = launch_options.grid_dim->y;
        config.gridDimZ = launch_options.grid_dim->z;
        config.blockDimX = launch_options.block_dim->x;
        config.blockDimY = launch_options.block_dim->y;
        config.blockDimZ = launch_options.block_dim->z;
        config.sharedMemBytes = static_cast<unsigned int>(*launch_options.num_smem_bytes);
        config.hStream = launch_options.stream
            ? *launch_options.stream
            : static_cast<mcStream_t>(at::cuda::getCurrentCUDAStream().stream());
        config.attrs = num_attributes == 0 ? nullptr : attributes.data();
        config.numAttrs = num_attributes;
        DJ_MACA_DRIVER_CHECK(driver::lazy_mcModuleLaunchKernelEx(
            &config, kernel_handle,
            sizeof...(Args) == 0 ? nullptr : kernel_arg_ptrs, nullptr));
    }

    void unload() noexcept {
        if (module_handle == nullptr)
            return;

        try {
            DJ_MACA_DRIVER_CHECK(driver::lazy_mcModuleUnload(module_handle));
        } catch (...) {
        }
        module_handle = nullptr;
        kernel_handle = nullptr;
    }
};

}  // namespace deep_jit::maca
