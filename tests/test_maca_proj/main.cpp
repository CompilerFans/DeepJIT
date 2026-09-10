// MACA backend test harness.  Mirrors the structure of tests/test_cuda_proj:
// a standalone host binary that JIT-compiles kernels with mxcc, loads them
// through mcModuleLoad, and launches them on the device.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mc_runtime.h>

#include <deep_jit/backend/maca/backend.hpp>
#include <deep_jit/cache/memory.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/json.hpp>
#include <deep_jit/utils/no_ref_ptr.hpp>
#include <deep_jit/utils/parser.hpp>
#include <deep_jit/utils/str.hpp>
#include <deep_jit/utils/uuid.hpp>

namespace {

using Runtime = deep_jit::Runtime<deep_jit::MACA>;
using CompilerOptions = deep_jit::maca::CompilerOptions;
using LaunchOptions = deep_jit::maca::LaunchOptions;
namespace fs = std::filesystem;

const fs::path& get_test_maca_project_dir() {
    static const fs::path path = [] {
        const auto value = std::getenv("DEEP_JIT_MACA_TEST_SOURCE_DIR");
        DJ_HOST_ASSERT(value != nullptr, "DEEP_JIT_MACA_TEST_SOURCE_DIR must be set");
        return fs::absolute(value).lexically_normal();
    }();
    return path;
}

template <typename Function>
void run_test(const std::string_view name, Function&& function) {
    std::printf("[ RUN      ] %.*s\n", static_cast<int>(name.size()), name.data());
    std::fflush(stdout);
    if constexpr (std::is_same_v<std::invoke_result_t<Function>, bool>) {
        if (not function()) {
            std::printf("[  SKIPPED ] %.*s\n", static_cast<int>(name.size()), name.data());
            std::fflush(stdout);
            return;
        }
    } else {
        function();
    }
    std::printf("[       OK ] %.*s\n", static_cast<int>(name.size()), name.data());
    std::fflush(stdout);
}

template <typename Function>
void expect_failure(Function&& function, const std::string_view expected_message) {
    try {
        function();
    } catch (const std::exception& exception) {
        DJ_HOST_ASSERT(std::string_view(exception.what()).find(expected_message) != std::string_view::npos,
                       "unexpected exception: {}", exception.what());
        return;
    }
    DJ_PANIC("expected failure containing: {}", expected_message);
}

void set_env(const std::string& name, const std::string& value) {
    DJ_HOST_ASSERT(::setenv(name.c_str(), value.c_str(), 1) == 0, "failed to set environment variable: {}", name);
}

void unset_env(const std::string& name) {
    DJ_HOST_ASSERT(::unsetenv(name.c_str()) == 0, "failed to unset environment variable: {}", name);
}

std::string get_source(const std::string& name) {
    return deep_jit::read(get_test_maca_project_dir() / "kernels" / name);
}

std::shared_ptr<Runtime> make_runtime(const fs::path& include_dir,
                                      const std::string& extra_signature = "maca-test-signature") {
    return std::make_shared<Runtime>(deep_jit::Config(
        get_test_maca_project_dir(),
        "TEST",
        extra_signature,
        {include_dir},
        {"test_maca/"}));
}

int launch_value(Runtime& runtime, const std::shared_ptr<deep_jit::maca::Kernel>& kernel, const int input = 0) {
    int* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    try {
        runtime.launch(kernel, {.grid_dim = dim3(1, 1, 1), .block_dim = dim3(1, 1, 1)}, output, input);
        DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
        int result = 0;
        DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(int), mcMemcpyDeviceToHost));
        DJ_MACA_RUNTIME_CHECK(mcFree(output));
        return result;
    } catch (...) {
        mcFree(output);
        throw;
    }
}

void test_environment(const fs::path& cache_root) {
    set_env("TEST_ENV_TEST_VALUE", "library");
    set_env("DJ_TEST_VALUE", "global");
    const deep_jit::Env env("TEST_ENV");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "library");
    unset_env("TEST_ENV_TEST_VALUE");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "global");
    unset_env("DJ_TEST_VALUE");
    DJ_HOST_ASSERT(not env.get<std::string>("TEST_VALUE").has_value());
}

void test_config() {
    const auto root = get_test_maca_project_dir();
    const deep_jit::Config config(root, "CONFIG_TEST", "dependency-version", {root / "kernels"}, {"test_maca/"});
    DJ_HOST_ASSERT(fs::equivalent(config.python_library_root, root));
    expect_failure([] { deep_jit::Config({}, "CONFIG_TEST"); }, "Python library root must not be empty");
    expect_failure([&] { deep_jit::Config(root, "DJ"); }, "reserved for global environment variables");
}

void test_toolkit_discovery(const fs::path& cache_root) {
    const deep_jit::Env env("TOOLKIT_DISCOVERY");
    const auto toolkit = deep_jit::MACA::find_maca_toolkit(env);
    DJ_HOST_ASSERT(deep_jit::is_executable(toolkit.mxcc), "discovered mxcc is not executable");
    std::printf("           mxcc: %s\n", toolkit.mxcc.string().c_str());

    // The library-prefixed override must win over the inherited default.
    const auto fake_dir = cache_root / "fake_toolkit";
    deep_jit::make_dirs(fake_dir);
    const auto make_executable = [](const fs::path& path, const std::string& content) {
        deep_jit::write_file_sync(path, content);
        fs::permissions(path, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                        fs::perm_options::add);
    };
    const auto fake_mxcc = fake_dir / "mxcc";
    make_executable(fake_mxcc, "#!/bin/sh\necho 'mxcc version 1.0.0 (test)'\n");
    // `llvm-nm` is resolved next to mxcc; the fixture must provide both so the
    // discovery exercises the override rather than the executable check.
    make_executable(fake_dir / "llvm-nm", "#!/bin/sh\nexit 0\n");
    set_env("TOOLKIT_DISCOVERY_JIT_MXCC_COMPILER", fake_mxcc.string());
    const auto overridden = deep_jit::MACA::find_maca_toolkit(env);
    DJ_HOST_ASSERT(overridden.mxcc == fs::absolute(fake_mxcc).lexically_normal(),
                   "library mxcc override was not selected");
    DJ_HOST_ASSERT(overridden.llvm_nm == fs::absolute(fake_dir / "llvm-nm").lexically_normal(),
                   "llvm-nm was not resolved next to the overridden mxcc");
    unset_env("TOOLKIT_DISCOVERY_JIT_MXCC_COMPILER");

    set_env("INVALID_TOOLKIT_JIT_MXCC_COMPILER", (fake_dir / "missing_mxcc").string());
    expect_failure(
        [&] { deep_jit::MACA::find_maca_toolkit(deep_jit::Env("INVALID_TOOLKIT")); },
        "mxcc compiler is not executable");
    unset_env("INVALID_TOOLKIT_JIT_MXCC_COMPILER");
}

void test_device_properties(Runtime& runtime) {
    const auto family = runtime.device.get_family();
    const auto arch = runtime.device.get_arch();
    std::printf("           num_sms: %d\n", runtime.device.get_num_sms());
    std::printf("           l2_cache: %d\n", runtime.device.get_num_l2_cache_bytes());
    std::printf("           smem_per_block_optin: %d\n", runtime.device.get_num_smem_bytes());
    std::printf("           clock_rate: %lld\n", static_cast<long long>(runtime.device.get_clock_rate()));
    std::printf("           mc arch: %d.%d -> xcore%s\n",
                runtime.device.get_arch_major(), runtime.device.get_arch_minor(), arch.c_str());
    DJ_HOST_ASSERT(runtime.device.get_num_sms() > 0, "device reported no SMs");
    DJ_HOST_ASSERT(not arch.empty());
    // The family and the offload target must agree.
    DJ_HOST_ASSERT(std::to_string(static_cast<int>(family)) == arch);
}

void test_options(Runtime& runtime) {
    const auto& defaults = runtime.default_compiler_options;
    DJ_HOST_ASSERT(defaults.optimize_level == "3");
    DJ_HOST_ASSERT(defaults.fast_math == false);
    DJ_HOST_ASSERT(defaults.with_line_info == false);
    DJ_HOST_ASSERT(defaults.dump_asm == false);
    DJ_HOST_ASSERT(defaults.arch.has_value() and defaults.mxcc_flags.has_value());
    DJ_HOST_ASSERT(not defaults.post_hook.has_value());

    const std::vector<std::string> expected = {
        "--offload-arch=xcore" + *defaults.arch,
        "-O3",
        "-std=c++20",
    };
    DJ_HOST_ASSERT(defaults.get_flags() == expected, "unexpected default mxcc flag order");

    const auto& launch_defaults = runtime.default_launch_options;
    DJ_HOST_ASSERT(not launch_defaults.stream.has_value());
    DJ_HOST_ASSERT(launch_defaults.num_smem_bytes == 0);
    DJ_HOST_ASSERT(launch_defaults.cooperative == false and launch_defaults.enable_pdl == false);

    auto missing_arch = defaults;
    missing_arch.arch.reset();
    expect_failure([&] { (void)missing_arch.get_flags(); }, "MACA architecture must be specified");
}

void test_json_and_hash() {
    const deep_jit::json value = deep_jit::json::object_t {{"integer", 10}, {"optional", std::optional<int>()}};
    DJ_HOST_ASSERT(value.dump() == R"({"integer":10,"optional":null})");
    const auto digest = deep_jit::hash::get_hex_digest("maca");
    DJ_HOST_ASSERT(digest.size() == 32);
    DJ_HOST_ASSERT(deep_jit::str::join({"one", "two"}, ":") == "one:two");
}

void test_parser(const fs::path& cache_root) {
    const auto root = cache_root / "parser";
    deep_jit::make_dirs(root / "maca");
    deep_jit::write_file_sync(root / "maca/dependency.cu", "constexpr int kDependency = 1;\n");
    deep_jit::Parser parser({root}, {"maca/"});
    const auto source = "#include <maca/dependency.cu>\n";
    DJ_HOST_ASSERT(not parser.parse_into_hash(source).empty());

    const auto changed_root = cache_root / "parser_changed";
    deep_jit::make_dirs(changed_root / "maca");
    deep_jit::write_file_sync(changed_root / "maca/dependency.cu", "constexpr int kDependency = 2;\n");
    deep_jit::Parser changed({changed_root}, {"maca/"});
    DJ_HOST_ASSERT(parser.parse_into_hash(source) != changed.parse_into_hash(source),
                   "changed include contents must change the hash");
}

// The headline test: compile a kernel with mxcc, load it, run it on the GPU.
void test_jit_round_trip(Runtime& runtime) {
    const auto source = get_source("scalar_increment.cu");
    const auto kernel = runtime.compile("scalar_increment", source);
    DJ_HOST_ASSERT(kernel != nullptr);
    DJ_HOST_ASSERT(launch_value(runtime, kernel, 41) == 42, "JIT'd scalar kernel produced the wrong value");
}

void test_jit_vector_add(Runtime& runtime) {
    constexpr int size = 256;
    const auto source = get_source("vector_add.cu");
    const auto kernel = runtime.compile("vector_add", source);

    std::vector<float> left(size), right(size), expected(size);
    for (int i = 0; i < size; ++i) {
        left[i] = static_cast<float>(i);
        right[i] = static_cast<float>(2 * i);
        expected[i] = left[i] + right[i];
    }

    float* d_output = nullptr;
    float* d_left = nullptr;
    float* d_right = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&d_output), size * sizeof(float)));
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&d_left), size * sizeof(float)));
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&d_right), size * sizeof(float)));
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(d_left, left.data(), size * sizeof(float), mcMemcpyHostToDevice));
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(d_right, right.data(), size * sizeof(float), mcMemcpyHostToDevice));

    int device_size = size;
    runtime.launch(kernel, {.grid_dim = dim3(2, 1, 1), .block_dim = dim3(128, 1, 1)},
                   d_output, d_left, d_right, device_size);
    DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());

    std::vector<float> output(size);
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(output.data(), d_output, size * sizeof(float), mcMemcpyDeviceToHost));
    DJ_MACA_RUNTIME_CHECK(mcFree(d_output));
    DJ_MACA_RUNTIME_CHECK(mcFree(d_left));
    DJ_MACA_RUNTIME_CHECK(mcFree(d_right));

    for (int i = 0; i < size; ++i)
        DJ_HOST_ASSERT(output[i] == expected[i], "vector add mismatch at {}: {} != {}", i, output[i], expected[i]);
}

void test_multiple_kernels_rejected(Runtime& runtime) {
    const auto source = get_source("multiple_kernels.cu");
    expect_failure([&] { (void)runtime.compile("multiple_kernels", source); },
                   "expected exactly one kernel");
}

void test_compiler_options_affect_result(Runtime& runtime) {
    const auto source = get_source("compiler_options.cu");
    const CompilerOptions options {
        .mxcc_flags = std::vector<std::string>{"-std=c++20", "-DTEST_OPTION=64"},
    };
    DJ_HOST_ASSERT(launch_value(runtime, runtime.compile("compiler_options", source, options)) == 64,
                   "consumer macro was not compiled");
}

void test_cache_key_tracks_flags(Runtime& runtime) {
    const auto source = get_source("compiler_options.cu");
    const auto original = runtime.cache_key(source, runtime.default_compiler_options);
    auto changed = runtime.default_compiler_options;
    changed.extra_mxcc_flags.emplace_back("-DTEST_OPTION=17");
    DJ_HOST_ASSERT(runtime.cache_key(source, changed) != original,
                   "extra mxcc flags did not affect the cache key");
}

void test_cache_artifacts(Runtime& runtime, const fs::path& cache_root) {
    const auto source = get_source("scalar_increment.cu");
    const auto artifact = runtime.compile_without_load("cache_artifacts", source);
    DJ_HOST_ASSERT(fs::is_regular_file(artifact / "kernel.cu"), "missing kernel source: {}", artifact.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact / "kernel.devbin"), "missing device binary: {}", artifact.string());
    DJ_HOST_ASSERT(fs::file_size(artifact / "kernel.devbin") > 0, "empty device binary");
    DJ_HOST_ASSERT(fs::is_regular_file(artifact / "meta.json"), "missing metadata");
    DJ_HOST_ASSERT(deep_jit::read(artifact / "kernel.cu") == source, "cached source mismatch");

    const auto metadata = deep_jit::read(artifact / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"command\":") != std::string::npos);
    DJ_HOST_ASSERT(metadata.find("mxcc") != std::string::npos, "metadata does not record the compiler command");
    DJ_HOST_ASSERT(metadata.find("--offload-arch=xcore") != std::string::npos,
                   "metadata does not record the offload target");

    // A second compile of the same source must hit the same artifact.
    const auto again = runtime.compile_without_load("cache_artifacts", source);
    DJ_HOST_ASSERT(again == artifact, "disk cache did not reuse the artifact");
}

void test_dump_asm(Runtime& runtime) {
    const auto source = get_source("scalar_increment.cu");
    const CompilerOptions options {.dump_asm = true};
    const auto artifact = runtime.compile_without_load("dump_asm", source, options);
    DJ_HOST_ASSERT(fs::is_regular_file(artifact / "kernel.s"), "missing device assembly dump");
    DJ_HOST_ASSERT(fs::file_size(artifact / "kernel.s") > 0, "empty device assembly dump");
}

void test_launch_option_validation(Runtime& runtime) {
    const auto source = get_source("scalar_increment.cu");
    const auto kernel = runtime.compile("scalar_increment", source);

    // Missing grid/block must be rejected before reaching the driver.
    expect_failure([&] { runtime.launch(kernel, {.grid_dim = dim3(1, 1, 1)}, nullptr, 0); },
                   "block dimension must be specified");

    // Cluster launch and PDL are the axes MACA genuinely does not implement:
    // any attribute other than the cooperative flag makes
    // `mcModuleLaunchKernelEx` fail with mcErrorInvalidConfiguration ("Only
    // Cooperative features are supported, others are not").  They are
    // rejected up front with a clear message rather than surfacing as an
    // opaque driver error code.
    const auto launch_with = [&](const LaunchOptions& overrides) {
        auto options = runtime.default_launch_options;
        options.grid_dim = dim3(1, 1, 1);
        options.block_dim = dim3(1, 1, 1);
        if (overrides.cluster_dim)
            options.cluster_dim = overrides.cluster_dim;
        if (overrides.enable_pdl)
            options.enable_pdl = overrides.enable_pdl;
        kernel->launch(options, nullptr, 0);
    };
    expect_failure([&] { launch_with(LaunchOptions {.cluster_dim = dim3(2, 1, 1)}); },
                   "does not support cluster launch");
    expect_failure([&] { launch_with(LaunchOptions {.enable_pdl = true}); },
                   "does not support programmatic dependent launch");
    // The rejection is unconditional: no environment flag may re-enable a
    // path the runtime refuses anyway.
    set_env("DJ_MACA_ALLOW_UNSUPPORTED_LAUNCH_ATTRS", "1");
    expect_failure([&] { launch_with(LaunchOptions {.cluster_dim = dim3(2, 1, 1)}); },
                   "does not support cluster launch");
    unset_env("DJ_MACA_ALLOW_UNSUPPORTED_LAUNCH_ATTRS");

    // Dynamic shared memory and the cooperative flag *are* supported and must
    // still reach the driver successfully.
    int* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    auto options = runtime.default_launch_options;
    options.grid_dim = dim3(1, 1, 1);
    options.block_dim = dim3(1, 1, 1);
    options.num_smem_bytes = 1024;
    options.cooperative = true;
    kernel->launch(options, output, 7);
    DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
    int result = 0;
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(int), mcMemcpyDeviceToHost));
    DJ_MACA_RUNTIME_CHECK(mcFree(output));
    DJ_HOST_ASSERT(result == 8, "cooperative launch produced the wrong value: {}", result);
}

void test_large_dynamic_shared_memory(Runtime& runtime) {
    // A kernel whose dynamic shared-memory request exceeds the default
    // per-block limit must launch on the `mcLaunchConfigExtension::
    // sharedMemBytes` field alone -- no function-attribute call is needed on
    // MACA (the launch enforces the `sharedMemPerBlockOptin` ceiling itself).
    // 64 KiB is the device maximum, so this is the strongest form of the
    // proof: if the field were ignored, or if a ceiling attribute were
    // required, this launch would be refused.
    const auto source = get_source("large_dynamic_shared_memory.cu");
    const auto kernel = runtime.compile("large_smem", source);
    const int device_max = runtime.device.get_num_smem_bytes();
    DJ_HOST_ASSERT(device_max > 0, "device reported no opt-in shared memory");
    std::printf("           sharedMemPerBlockOptin: %d, requesting: %d\n", device_max, device_max);

    auto options = runtime.default_launch_options;
    options.grid_dim = dim3(1, 1, 1);
    options.block_dim = dim3(32, 1, 1);
    options.num_smem_bytes = device_max;

    int* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    kernel->launch(options, output, 41);
    DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
    int result = 0;
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(int), mcMemcpyDeviceToHost));
    DJ_MACA_RUNTIME_CHECK(mcFree(output));
    DJ_HOST_ASSERT(result == 42, "large shared-memory launch produced the wrong value: {}", result);

    // One byte over the ceiling must be refused by the launch itself.
    options.num_smem_bytes = device_max + 1;
    bool refused = false;
    try {
        kernel->launch(options, nullptr, 0);
    } catch (const std::exception&) {
        refused = true;
    }
    DJ_HOST_ASSERT(refused, "a request above sharedMemPerBlockOptin was not refused");
}

void test_launch_options_override(Runtime& runtime) {
    auto defaults = LaunchOptions::default_options(runtime.env);
    defaults.grid_dim = dim3(2, 3, 4);
    defaults.cooperative = true;
    const auto overridden = defaults.override_with(LaunchOptions {.grid_dim = dim3(1, 1, 1), .cooperative = false});
    DJ_HOST_ASSERT(overridden.grid_dim->x == 1 and overridden.cooperative == false,
                   "explicit false launch options must override true defaults");
    DJ_HOST_ASSERT(defaults.cooperative == true, "launch option override mutated the defaults");
}

bool test_multiple_devices() {
    int num_devices = 0;
    DJ_MACA_RUNTIME_CHECK(mcGetDeviceCount(&num_devices));
    std::printf("           visible devices: %d\n", num_devices);
    return num_devices >= 2;
}

}  // namespace

int main() {
    const auto cache_root = fs::path(std::getenv("DEEP_JIT_MACA_TEST_CACHE_DIR") == nullptr
                                         ? "/tmp/deep_jit_maca_test"
                                         : std::getenv("DEEP_JIT_MACA_TEST_CACHE_DIR"));
    deep_jit::safe_remove_all(cache_root);
    deep_jit::make_dirs(cache_root);
    set_env("TEST_JIT_CACHE_DIR", (cache_root / "cache").string());

    const auto runtime = make_runtime(get_test_maca_project_dir() / "kernels");

    run_test("environment", [&] { test_environment(cache_root); });
    run_test("config", [&] { test_config(); });
    run_test("toolkit_discovery", [&] { test_toolkit_discovery(cache_root); });
    run_test("device_properties", [&] { test_device_properties(*runtime); });
    run_test("options", [&] { test_options(*runtime); });
    run_test("json_and_hash", [&] { test_json_and_hash(); });
    run_test("parser", [&] { test_parser(cache_root); });
    run_test("launch_options_override", [&] { test_launch_options_override(*runtime); });
    run_test("jit_round_trip", [&] { test_jit_round_trip(*runtime); });
    run_test("jit_vector_add", [&] { test_jit_vector_add(*runtime); });
    run_test("multiple_kernels_rejected", [&] { test_multiple_kernels_rejected(*runtime); });
    run_test("compiler_options_affect_result", [&] { test_compiler_options_affect_result(*runtime); });
    run_test("cache_key_tracks_flags", [&] { test_cache_key_tracks_flags(*runtime); });
    run_test("cache_artifacts", [&] { test_cache_artifacts(*runtime, cache_root); });
    run_test("dump_asm", [&] { test_dump_asm(*runtime); });
    run_test("launch_option_validation", [&] { test_launch_option_validation(*runtime); });
    run_test("large_dynamic_shared_memory", [&] { test_large_dynamic_shared_memory(*runtime); });
    run_test("multiple_devices", [&] { return test_multiple_devices(); });

    std::printf("\nAll MACA DeepJIT tests passed.\n");
    return 0;
}
