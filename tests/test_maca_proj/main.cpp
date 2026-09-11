// MACA backend test harness.  Mirrors the structure of tests/test_cuda_proj:
// a standalone host binary that JIT-compiles kernels with mxcc, loads them
// through mcModuleLoad, and launches them on the device.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <mc_runtime.h>

#include <deep_jit/backend/maca/backend.hpp>
#include <deep_jit/cache/disk.hpp>
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

// For failures whose message comes from the driver rather than from DeepJIT's
// own checks: the shape of the error text is the platform's, so only the
// refusal itself can be asserted on.
template <typename Function>
void expect_any_failure(Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    DJ_PANIC("expected a failure");
}

// For failures that a caller can reach through more than one layer, where each
// layer has its own message: any of the named messages is a correct refusal,
// anything else is a different failure being mistaken for the expected one.
template <typename Function>
void expect_failure_any_of(Function&& function, const std::initializer_list<std::string_view> messages) {
    try {
        function();
    } catch (const std::exception& exception) {
        const std::string_view text(exception.what());
        for (const auto message: messages) {
            if (text.find(message) != std::string_view::npos)
                return;
        }
        DJ_PANIC("expected one of the listed failures, got: {}", text);
    }
    DJ_PANIC("expected a failure");
}

// For a case that has to disturb the ambient environment and put it back.
std::optional<std::string> get_raw_env(const std::string& name) {
    const auto value = std::getenv(name.c_str());
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
}

void restore_env(const std::string& name, const std::optional<std::string>& value) {
    if (value)
        DJ_HOST_ASSERT(::setenv(name.c_str(), value->c_str(), 1) == 0, "failed to restore {}", name);
    else
        DJ_HOST_ASSERT(::unsetenv(name.c_str()) == 0, "failed to unset {}", name);
}

void write_executable(const fs::path& path, const std::string& content) {
    deep_jit::write_file_sync(path, content);
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec |
            std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add);
}

std::size_t count_different_bytes(const std::string_view first, const std::string_view second) {
    DJ_HOST_ASSERT(first.size() == second.size(),
                   "byte comparison requires equal sizes: {} vs {}", first.size(), second.size());
    std::size_t count = 0;
    for (std::size_t index = 0; index < first.size(); ++index)
        count += first[index] != second[index];
    return count;
}

void check_artifact(const fs::path& artifact_dir, const std::string& source) {
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / ".committed"), "missing commit marker: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "kernel.cu"), "missing MACA source: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "kernel.devbin"), "missing device binary: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::file_size(artifact_dir / "kernel.devbin") > 0, "empty device binary: {}", artifact_dir.string());
    DJ_HOST_ASSERT(fs::is_regular_file(artifact_dir / "meta.json"), "missing metadata: {}", artifact_dir.string());
    DJ_HOST_ASSERT(deep_jit::read(artifact_dir / "kernel.cu") == source, "cached source mismatch: {}", artifact_dir.string());
}

void check_tmp_is_empty(const fs::path& cache_root) {
    const auto tmp_dir = cache_root / "tmp";
    DJ_HOST_ASSERT(not fs::exists(tmp_dir) or fs::directory_iterator(tmp_dir) == fs::directory_iterator(),
                   "temporary artifacts were not cleaned: {}", tmp_dir.string());
}

// Launch a single-argument kernel and read the one int it writes back.
int launch_single_argument(Runtime& runtime, const std::shared_ptr<deep_jit::maca::Kernel>& kernel) {
    int* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    try {
        runtime.launch(kernel, {.grid_dim = dim3(1, 1, 1), .block_dim = dim3(1, 1, 1)}, output);
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

void set_env(const std::string& name, const std::string& value) {
    DJ_HOST_ASSERT(::setenv(name.c_str(), value.c_str(), 1) == 0, "failed to set environment variable: {}", name);
}

void unset_env(const std::string& name) {
    DJ_HOST_ASSERT(::unsetenv(name.c_str()) == 0, "failed to unset environment variable: {}", name);
}

std::string get_source(const std::string& name) {
    return deep_jit::read(get_test_maca_project_dir() / "kernels" / name);
}

std::shared_ptr<Runtime> make_runtime_with_prefix(const std::string& prefix,
                                                  const fs::path& include_dir,
                                                  const std::string& extra_signature = "maca-test-signature") {
    return std::make_shared<Runtime>(deep_jit::Config(
        get_test_maca_project_dir(),
        prefix,
        extra_signature,
        {include_dir},
        {"test_maca/"}));
}

std::shared_ptr<Runtime> make_runtime(const fs::path& include_dir,
                                      const std::string& extra_signature = "maca-test-signature") {
    return make_runtime_with_prefix("TEST", include_dir, extra_signature);
}

// The compile-time constant lives in the source text, so two biases are two
// different cache entries by construction.
std::string get_template_source(const int bias) {
    return std::format(
        "extern \"C\" __global__ void template_add_kernel(int* output, const int input) {{\n"
        "    output[0] = input + {};\n"
        "}}\n",
        bias);
}

fs::path get_cache_root() {
    const auto value = std::getenv("DEEP_JIT_MACA_TEST_CACHE_DIR");
    return fs::path(value == nullptr ? "/tmp/deep_jit_maca_test" : value);
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
    set_env("DJ_TEST_VALUE", "global");
    set_env("TEST_ENV_TEST_VALUE", "library");
    set_env("TEST_VALUE", "unprefixed");
    set_env("TEST_ENV_BOOL", "YeS");

    const deep_jit::Env env("TEST_ENV");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "library");
    DJ_HOST_ASSERT(env.get<bool>("BOOL") == true);
    unset_env("TEST_ENV_TEST_VALUE");
    DJ_HOST_ASSERT(env.get<std::string>("TEST_VALUE") == "global");
    unset_env("DJ_TEST_VALUE");
    DJ_HOST_ASSERT(not env.get<std::string>("TEST_VALUE").has_value());
    unset_env("TEST_ENV_BOOL");
    unset_env("TEST_VALUE");

    set_env("TEST_ENV_BOOL_TRUE", "TrUe");
    set_env("TEST_ENV_BOOL_FALSE", "nO");
    set_env("TEST_ENV_BOOL_INTEGER", "-2");
    set_env("TEST_ENV_INTEGER", "-17");
    DJ_HOST_ASSERT(env.get<bool>("BOOL_TRUE") == true);
    DJ_HOST_ASSERT(env.get<bool>("BOOL_FALSE") == false);
    DJ_HOST_ASSERT(env.get<bool>("BOOL_INTEGER") == true);
    DJ_HOST_ASSERT(env.get<int>("INTEGER") == -17);
    unset_env("TEST_ENV_BOOL_TRUE");
    unset_env("TEST_ENV_BOOL_FALSE");
    unset_env("TEST_ENV_BOOL_INTEGER");
    unset_env("TEST_ENV_INTEGER");

    set_env("TEST_ENV_INVALID_BOOL", "sometimes");
    expect_failure([&] { env.get<bool>("INVALID_BOOL"); }, "invalid value");
    unset_env("TEST_ENV_INVALID_BOOL");
    set_env("TEST_ENV_INVALID_INTEGER", "12x");
    expect_failure([&] { env.get<int>("INVALID_INTEGER"); }, "invalid value");
    unset_env("TEST_ENV_INVALID_INTEGER");
    set_env("TEST_ENV_EMPTY_BOOL", "");
    expect_failure([&] { env.get<bool>("EMPTY_BOOL"); }, "invalid value");
    unset_env("TEST_ENV_EMPTY_BOOL");
    set_env("TEST_ENV_UNSIGNED", "-1");
    expect_failure([&] { env.get<unsigned>("UNSIGNED"); }, "invalid value");
    unset_env("TEST_ENV_UNSIGNED");
    set_env("TEST_ENV_OVERFLOW", "999999999999999999999999999999");
    expect_failure([&] { env.get<int64_t>("OVERFLOW"); }, "invalid value");
    unset_env("TEST_ENV_OVERFLOW");

    expect_failure([] { deep_jit::Env(""); }, "prefix must not be empty");
    expect_failure([] { deep_jit::Env("DJ"); }, "reserved for global environment variables");

    // A colon-separated cache list is how a consumer points the runtime at a
    // write-through chain; the order is the lookup order.
    const auto first_cache = cache_root / "first";
    const auto second_cache = cache_root / "second";
    set_env("TEST_ENV_JIT_CACHE_DIR", first_cache.string() + ":" + second_cache.string());
    const auto disk_cache = deep_jit::DiskCache::from_env(env);
    DJ_HOST_ASSERT(disk_cache.paths.size() == 2);
    DJ_HOST_ASSERT(disk_cache.paths[0] == first_cache);
    DJ_HOST_ASSERT(disk_cache.paths[1] == second_cache);

    // A list with a hole in it is not a list with a default in it.
    for (const auto& invalid_paths : std::vector<std::string> {
             "",
             ":" + first_cache.string(),
             first_cache.string() + ":",
             first_cache.string() + "::" + second_cache.string(),
         }) {
        set_env("TEST_ENV_JIT_CACHE_DIR", invalid_paths);
        expect_failure([&] { (void)deep_jit::DiskCache::from_env(env); }, "contains an empty path");
    }
    unset_env("TEST_ENV_JIT_CACHE_DIR");

    // Below the library prefix: `DJ_` is the global fallback, an unprefixed
    // name does nothing, and the built-in default is `$HOME/.dj`.
    const auto saved_home = get_raw_env("HOME");
    const auto saved_global_cache = get_raw_env("DJ_JIT_CACHE_DIR");
    const auto global_fallback = cache_root / "global_fallback";
    set_env("DJ_JIT_CACHE_DIR", global_fallback.string());
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(env).paths == std::vector<fs::path> {global_fallback},
                   "DJ cache directory was not used as the global fallback");
    set_env("JIT_CACHE_DIR", (cache_root / "ignored_unprefixed").string());
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(env).paths == std::vector<fs::path> {global_fallback},
                   "an unprefixed cache directory unexpectedly took effect");
    unset_env("JIT_CACHE_DIR");
    unset_env("DJ_JIT_CACHE_DIR");
    DJ_HOST_ASSERT(saved_home.has_value(), "the test environment must provide HOME");
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(env).paths == std::vector<fs::path> {fs::path(*saved_home) / ".dj"},
                   "the built-in cache directory is not $HOME/.dj");
    unset_env("HOME");
    expect_failure([&] { (void)deep_jit::DiskCache::from_env(env); },
                   "HOME environment variable must not be empty");
    restore_env("HOME", saved_home);
    restore_env("DJ_JIT_CACHE_DIR", saved_global_cache);
}

void test_config() {
    const auto root = get_test_maca_project_dir();
    const auto include_dir = root / "kernels" / ".";
    const deep_jit::Config config(root / "scripts" / "..", "CONFIG_TEST", "dependency-version",
                                 {include_dir}, {"test_maca/"});
    DJ_HOST_ASSERT(fs::equivalent(config.python_library_root, root));
    DJ_HOST_ASSERT(config.include_dirs.size() == 1 and fs::equivalent(config.include_dirs.front(), root / "kernels"),
                   "include directories were not normalised");
    DJ_HOST_ASSERT(fs::equivalent(config.get_python_path("scripts/../scripts/post_hook.py"),
                                  root / "scripts/post_hook.py"));
    const auto serialized = config.to_json().dump();
    DJ_HOST_ASSERT(serialized.find("\"extra_signature\":\"dependency-version\"") != std::string::npos);
    DJ_HOST_ASSERT(serialized.find("\"include_prefixes\":[\"test_maca/\"]") != std::string::npos);

    expect_failure([] { deep_jit::Config({}, "CONFIG_TEST"); }, "Python library root must not be empty");
    expect_failure([] { deep_jit::Config("relative", "CONFIG_TEST"); }, "Python library root must be absolute");
    expect_failure([&] { deep_jit::Config(root, ""); }, "prefix must not be empty");
    expect_failure([&] { deep_jit::Config(root, "DJ"); }, "reserved for global environment variables");
    expect_failure([&] { deep_jit::Config(root, "CONFIG_TEST", {}, {fs::path {}}); },
                   "include directory must not be empty");
    expect_failure([&] { deep_jit::Config(root, "CONFIG_TEST", {}, {"relative"}); },
                   "include directory must be absolute");
}

void test_toolkit_discovery(const fs::path& cache_root) {
    const deep_jit::Env env("TOOLKIT_DISCOVERY");
    const auto toolkit = deep_jit::MACA::find_maca_toolkit(env);
    DJ_HOST_ASSERT(deep_jit::is_executable(toolkit.mxcc), "discovered mxcc is not executable");
    std::printf("           mxcc: %s\n", toolkit.mxcc.string().c_str());

    // The library-prefixed override must win over the inherited default.
    const auto fake_dir = cache_root / "fake_toolkit";
    deep_jit::make_dirs(fake_dir);
    const auto fake_mxcc = fake_dir / "mxcc";
    write_executable(fake_mxcc, "#!/bin/sh\necho 'mxcc version 1.0.0 (test)'\n");
    // `llvm-nm` is resolved next to mxcc; the fixture must provide both so the
    // discovery exercises the override rather than the executable check.
    write_executable(fake_dir / "llvm-nm", "#!/bin/sh\nexit 0\n");
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

    // The install-root precedence chain: `MACA_HOME > MACA_PATH > CUDA_HOME >
    // CUDA_PATH`, with `/opt/maca` only as the last resort.  These name an
    // installation rather than a library setting, so discovery reads them
    // UNPREFIXED -- the test drives the raw environment and restores it.
    const std::array<std::string, 4> home_names = {"MACA_HOME", "MACA_PATH", "CUDA_HOME", "CUDA_PATH"};
    std::array<std::optional<std::string>, 4> saved_homes;
    for (std::size_t index = 0; index < home_names.size(); ++index)
        saved_homes[index] = get_raw_env(home_names[index]);

    const auto make_home = [&](const std::string& name) {
        const auto bin = cache_root / name / "mxgpu_llvm" / "bin";
        deep_jit::make_dirs(bin);
        write_executable(bin / "mxcc", "#!/bin/sh\necho 'mxcc version 1.0.0 (test)'\n");
        write_executable(bin / "llvm-nm", "#!/bin/sh\nexit 0\n");
        return cache_root / name;
    };
    const auto expect_home = [&](const fs::path& expected, const std::string_view message) {
        const auto discovered = deep_jit::MACA::find_maca_toolkit(env);
        DJ_HOST_ASSERT(discovered.mxcc == fs::absolute(expected / "mxgpu_llvm" / "bin" / "mxcc").lexically_normal(),
                       "{}", message);
    };

    const auto home_a = make_home("home_a");
    const auto home_b = make_home("home_b");
    const auto home_c = make_home("home_c");
    const auto home_d = make_home("home_d");
    for (const auto& name: home_names)
        unset_env(name);

    set_env("CUDA_PATH", home_d.string());
    expect_home(home_d, "CUDA_PATH was not used when nothing above it was set");
    set_env("CUDA_HOME", home_c.string());
    expect_home(home_c, "CUDA_HOME must win over CUDA_PATH");
    set_env("MACA_PATH", home_b.string());
    expect_home(home_b, "MACA_PATH must win over the CUDA_* pair");
    set_env("MACA_HOME", home_a.string());
    expect_home(home_a, "MACA_HOME must win over MACA_PATH");

    // A home that is set but does not exist is an error, not a reason to try
    // the next name down: a typo must not silently select a different install.
    set_env("MACA_HOME", (cache_root / "missing_home").string());
    expect_failure([&] { (void)deep_jit::MACA::find_maca_toolkit(env); },
                   "MACA toolkit home was not found");

    for (const auto& name: home_names)
        unset_env(name);
    if (fs::exists("/opt/maca"))
        expect_home("/opt/maca", "the built-in /opt/maca fallback was not used");

    for (std::size_t index = 0; index < home_names.size(); ++index)
        restore_env(home_names[index], saved_homes[index]);
}

void test_device_properties(Runtime& runtime) {
    // A fresh, default-constructed device object must initialize itself on
    // first use: MACA's `Device` reads the properties only when asked.
    deep_jit::maca::Device sms_device;
    DJ_HOST_ASSERT(sms_device.get_num_sms() > 0,
                   "get_num_sms did not initialize MACA device properties lazily");
    deep_jit::maca::Device l2_device;
    DJ_HOST_ASSERT(l2_device.get_num_l2_cache_bytes() > 0,
                   "get_num_l2_cache_bytes did not initialize MACA device properties lazily");
    deep_jit::maca::Device smem_device;
    DJ_HOST_ASSERT(smem_device.get_num_smem_bytes() > 0,
                   "get_num_smem_bytes did not initialize MACA device properties lazily");
    deep_jit::maca::Device clock_device;
    DJ_HOST_ASSERT(clock_device.get_clock_rate() > 0, "get_clock_rate failed on a fresh device object");

    // One properties block serves the whole capability surface, so it is read
    // once and handed out by reference.
    const auto& first_prop = runtime.device.get_prop();
    const auto& second_prop = runtime.device.get_prop();
    DJ_HOST_ASSERT(&first_prop == &second_prop, "MACA device properties were not cached");

    std::printf("           num_sms: %d\n", runtime.device.get_num_sms());
    std::printf("           l2_cache: %d\n", runtime.device.get_num_l2_cache_bytes());
    std::printf("           smem_per_block_optin: %d\n", runtime.device.get_num_smem_bytes());
    std::printf("           clock_rate: %lld\n", static_cast<long long>(runtime.device.get_clock_rate()));
    std::printf("           mc arch: %d.%d -> xcore%s\n",
                runtime.device.get_arch_major(), runtime.device.get_arch_minor(),
                runtime.device.get_arch().c_str());

    const auto [major, minor] = runtime.device.get_arch_pair();
    DJ_HOST_ASSERT(runtime.device.get_arch_major() == major and runtime.device.get_arch_minor() == minor,
                   "the MACA architecture pair disagrees with the major/minor accessors");
    DJ_HOST_ASSERT(runtime.device.get_num_sms() > 0, "MACA device has no SMs");
    DJ_HOST_ASSERT(runtime.device.get_num_l2_cache_bytes() > 0, "MACA device has no L2 cache");
    DJ_HOST_ASSERT(runtime.device.get_num_smem_bytes() > 0, "MACA device has no shared memory");
    DJ_HOST_ASSERT(runtime.device.get_clock_rate() > 0, "MACA device clock rate is invalid");
    DJ_HOST_ASSERT(runtime.device.get_clock_rate() == runtime.device.get_clock_rate(),
                   "MACA clock rate was not cached");

    // The mc-major -> xcore-family mapping is spelled out HERE as literals.
    // Comparing the offload target to `get_family()` would be `f(x) == f(x)`:
    // `get_arch()` is defined as the family's digits, so a wrong `case` label
    // in that switch would satisfy both sides and go unnoticed.
    const auto expected_arch = [](const int mc_major) -> std::string_view {
        switch (mc_major) {
            case 10: return "1000";
            case 15: return "1500";
            case 16: return "1600";
            default: return {};
        }
    }(major);
    DJ_HOST_ASSERT(not expected_arch.empty(), "the device reported an unsupported mc major: {}", major);
    DJ_HOST_ASSERT(runtime.device.get_arch() == expected_arch,
                   "mc major {} must render the offload target xcore{}, got xcore{}",
                   major, expected_arch, runtime.device.get_arch());
    DJ_HOST_ASSERT(static_cast<int>(runtime.device.get_family()) == std::stoi(std::string(expected_arch)),
                   "the device family does not match mc major {}", major);
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

    // Each structured option has to emit the flag it means to.  The digest
    // axis only proves that setting an option CHANGES the flags, so a mangled
    // spelling would satisfy every other case while the kernel silently
    // stopped being built with the option.  `-use-fast-math` is the spelling
    // mxcc documents (`mxcc --help`) and honours -- measured on this platform,
    // a float kernel with 5 divides keeps 1 of them under the flag and its
    // device binary differs (24168 against 24160 bytes).
    const std::vector<std::string> expected_fast_math = {
        "--offload-arch=xcore" + *defaults.arch,
        "-O3",
        "-use-fast-math",
        "-std=c++20",
    };
    const auto fast_math_flags = defaults.override_with(CompilerOptions {.fast_math = true}).get_flags();
    DJ_HOST_ASSERT(fast_math_flags == expected_fast_math, "fast_math did not emit -use-fast-math");

    // `ptxas_verbose` is CUDA's spelling of `compiler_verbose`: overriding
    // with it has to survive the port and produce the same flag.
    const auto by_alias = defaults.override_with(CompilerOptions {.ptxas_verbose = true});
    const auto by_canonical = defaults.override_with(CompilerOptions {.compiler_verbose = true});
    const auto alias_flags = by_alias.get_flags();
    const auto canonical_flags = by_canonical.get_flags();
    DJ_HOST_ASSERT(by_alias.compiler_verbose == true,
                   "the ptxas_verbose alias did not reach compiler_verbose");
    DJ_HOST_ASSERT(alias_flags == canonical_flags,
                   "the ptxas_verbose alias and compiler_verbose must render the same flags");
    DJ_HOST_ASSERT(std::ranges::find(alias_flags, "-resource-usage") != alias_flags.end(),
                   "the ptxas_verbose alias did not request the resource report");
    // The canonical field wins when one override carries both spellings.
    const auto both_spellings = defaults.override_with(
        CompilerOptions {.compiler_verbose = false, .ptxas_verbose = true});
    const auto both_flags = both_spellings.get_flags();
    DJ_HOST_ASSERT(both_spellings.compiler_verbose == false and
                   std::ranges::find(both_flags, "-resource-usage") == both_flags.end(),
                   "compiler_verbose must win over its ptxas_verbose alias");
    // The alias is an input only: the defaults never populate it, so the
    // digest and `meta.json` keep exactly one spelling of the option.
    DJ_HOST_ASSERT(not defaults.ptxas_verbose.has_value(), "default_options must not set the alias");

    const auto& launch_defaults = runtime.default_launch_options;
    DJ_HOST_ASSERT(not launch_defaults.stream.has_value());
    DJ_HOST_ASSERT(launch_defaults.num_smem_bytes == 0);
    DJ_HOST_ASSERT(launch_defaults.cooperative == false and launch_defaults.enable_pdl == false);

    auto missing_arch = defaults;
    missing_arch.arch.reset();
    expect_failure([&] { (void)missing_arch.get_flags(); }, "MACA architecture must be specified");
}

void test_json() {
    static_assert(std::is_constructible_v<deep_jit::json, int>);
    static_assert(std::is_constructible_v<deep_jit::json, uint64_t>);
    static_assert(not std::is_constructible_v<deep_jit::json, double>);
    static_assert(not std::is_constructible_v<deep_jit::json, void*>);

    const deep_jit::json value = deep_jit::json::object_t {
        {"integer", 10},
        {"negative", -7},
        {"minimum", std::numeric_limits<int64_t>::min()},
        {"unsigned", std::numeric_limits<uint64_t>::max()},
        {"boolean", false},
        {"string", "line\n\"quoted\""},
        {"optional", std::optional<int>()},
        {"array", std::vector<int>{1, 2, 3}},
    };
    DJ_HOST_ASSERT(value.dump() == R"({"integer":10,"negative":-7,"minimum":-9223372036854775808,"unsigned":18446744073709551615,"boolean":false,"string":"line\n\"quoted\"","optional":null,"array":[1,2,3]})");
    DJ_HOST_ASSERT(deep_jit::json(std::string("\b\f\n\r\t\\\x01", 7)).dump() == "\"\\b\\f\\n\\r\\t\\\\\\u0001\"");
    DJ_HOST_ASSERT(deep_jit::json(std::string("a\0b", 3)).dump() == "\"a\\u0000b\"");
    DJ_HOST_ASSERT(deep_jit::json(deep_jit::json::object_t{{"key\n", 1}}).dump() == "{\"key\\n\":1}");

    // The metadata a cache entry publishes is built from this; an unset option
    // must serialize as null and an empty flag list as [].
    const auto empty_options = CompilerOptions().to_json().dump();
    DJ_HOST_ASSERT(empty_options.find("\"optimize_level\":null") != std::string::npos);
    DJ_HOST_ASSERT(empty_options.find("\"extra_mxcc_flags\":[]") != std::string::npos);
}

void test_hash_boundaries() {
    const std::array<std::string, 6> inputs = {"", "a", "ab", "abc", "abcd", "message digest"};
    std::unordered_set<std::string> input_digests;
    for (const auto& input : inputs) {
        const auto digest = deep_jit::hash::get_hex_digest(input);
        DJ_HOST_ASSERT(digest == deep_jit::hash::get_hex_digest(input), "hash output is not deterministic");
        DJ_HOST_ASSERT(digest.size() == 32 and std::ranges::all_of(digest, [](const char value) {
                           return (value >= '0' and value <= '9') or (value >= 'a' and value <= 'f');
                       }),
                       "hash digest is not 128-bit lowercase hexadecimal: {}", digest);
        DJ_HOST_ASSERT(input_digests.emplace(digest).second, "distinct basic inputs collided: {}", input);
    }

    const auto first = deep_jit::hash::FNV1a().update("ab").update("c").get_hex_digest();
    const auto second = deep_jit::hash::FNV1a().update("a").update("bc").get_hex_digest();
    DJ_HOST_ASSERT(first != second, "hash updates must be separated");
    const std::string left_with_zero("a\0", 2);
    const std::string right_with_zero("\0b", 2);
    DJ_HOST_ASSERT(deep_jit::hash::FNV1a().update(left_with_zero).update("b").get_hex_digest() !=
                       deep_jit::hash::FNV1a().update("a").update(right_with_zero).get_hex_digest(),
                   "hash framing must remain unambiguous with embedded null bytes");
    DJ_HOST_ASSERT(deep_jit::hash::FNV1a().update("").update("a").get_hex_digest() !=
                       deep_jit::hash::FNV1a().update("a").update("").get_hex_digest(),
                   "empty hash components must preserve their position");
    DJ_HOST_ASSERT(deep_jit::hash::get_hex_digest(std::string("a\0b", 3)) != deep_jit::hash::get_hex_digest("a"),
                   "embedded null bytes must participate in hashing");
    const std::array binary_inputs = {
        std::string("\0", 1),
        std::string("\0tail", 5),
        std::string("head\0", 5),
        std::string("head\0tail", 9),
        std::string("a\0b\0c", 5),
    };
    for (const auto& input : binary_inputs) {
        const auto digest = deep_jit::hash::get_hex_digest(input);
        DJ_HOST_ASSERT(digest ==
                           deep_jit::hash::get_hex_digest(std::string_view(input.data(), input.size())),
                       "string and explicit string_view hashing diverged");
        const auto null_position = input.find('\0');
        DJ_HOST_ASSERT(digest != deep_jit::hash::get_hex_digest(
                                     std::string_view(input.data(), null_position)),
                       "binary hash input was truncated at an embedded null");
    }
    const std::string after_null_a("prefix\0A", 8);
    const std::string after_null_b("prefix\0B", 8);
    DJ_HOST_ASSERT(deep_jit::hash::get_hex_digest(after_null_a) != deep_jit::hash::get_hex_digest(after_null_b),
                   "bytes after an embedded null must affect the hash");

    const auto segmented_digest = deep_jit::hash::FNV1a()
        .update(std::string("a\0", 2)).update(std::string("\0b", 2))
        .get_hex_digest();
    DJ_HOST_ASSERT(segmented_digest != deep_jit::hash::get_hex_digest(std::string("a\0\0b", 4)),
                   "chained hash boundaries were lost");

    auto base = deep_jit::hash::FNV1a().update("base");
    auto copied = base;
    DJ_HOST_ASSERT(base.update("left").get_hex_digest() != copied.update("right").get_hex_digest(),
                   "copied hash states must evolve independently");

    DJ_HOST_ASSERT(deep_jit::str::join({}).empty());
    DJ_HOST_ASSERT(deep_jit::str::join({"one"}) == "one");
    DJ_HOST_ASSERT(deep_jit::str::join({"one", "two", "three"}, ":") == "one:two:three");
}

void test_hash_robustness() {
    std::string all_bytes;
    all_bytes.reserve(256);
    for (int value = 0; value < 256; ++value)
        all_bytes.push_back(static_cast<char>(value));
    auto changed_all_bytes = all_bytes;
    changed_all_bytes.back() ^= 1;
    DJ_HOST_ASSERT(deep_jit::hash::get_hex_digest(all_bytes) != deep_jit::hash::get_hex_digest(changed_all_bytes),
                   "changing one byte in a binary input did not change the hash");

    std::unordered_set<std::string> one_byte_digests;
    for (int value = 0; value < 256; ++value) {
        const char byte = static_cast<char>(value);
        one_byte_digests.emplace(deep_jit::hash::get_hex_digest(std::string_view(&byte, 1)));
    }
    DJ_HOST_ASSERT(one_byte_digests.size() == 256, "single-byte inputs collided");

    {
        constexpr uint64_t num_collision_inputs = 1'000'000;
        std::mt19937_64 collision_generator(0x741bc92du);
        std::unordered_set<std::string> collision_digests;
        collision_digests.max_load_factor(0.8f);
        collision_digests.reserve(num_collision_inputs);
        const auto begin = std::chrono::steady_clock::now();
        for (uint64_t index = 0; index < num_collision_inputs; ++index) {
            const std::array<uint64_t, 4> input = {
                index, collision_generator(), collision_generator(), collision_generator()};
            const auto bytes = std::string_view(
                reinterpret_cast<const char*>(input.data()), sizeof(input));
            DJ_HOST_ASSERT(collision_digests.emplace(deep_jit::hash::get_hex_digest(bytes)).second,
                           "hash collision at randomized input {}", index);
        }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::printf("Hash collision test: %llu inputs in %.2f seconds\n",
                    static_cast<unsigned long long>(num_collision_inputs), elapsed);
    }

    constexpr int num_inputs = 10000;
    std::mt19937_64 generator(0x2d4a7f19u);
    uint64_t total_bit_distance = 0;
    uint64_t total_lane_distance = 0;
    std::array<int, 128> output_one_counts{};
    std::array<int, 128> changed_bit_counts{};
    std::array<int, 64> lane_difference_counts{};
    std::unordered_set<uint64_t> high_digests, low_digests;
    high_digests.reserve(num_inputs);
    low_digests.reserve(num_inputs);
    for (uint64_t index = 0; index < num_inputs; ++index) {
        std::string data(reinterpret_cast<const char*>(&index), sizeof(index));
        const auto random_size = static_cast<std::size_t>(generator() % 128);
        data.reserve(data.size() + random_size);
        for (std::size_t i = 0; i < random_size; ++i)
            data.push_back(static_cast<char>(generator()));

        const auto digest = deep_jit::hash::get_hex_digest(data);
        data[index % data.size()] ^= static_cast<char>(1u << (index % 8));
        const auto changed_digest = deep_jit::hash::get_hex_digest(data);
        const auto high = std::stoull(digest.substr(0, 16), nullptr, 16);
        const auto low = std::stoull(digest.substr(16), nullptr, 16);
        const auto changed_high = std::stoull(changed_digest.substr(0, 16), nullptr, 16);
        const auto changed_low = std::stoull(changed_digest.substr(16), nullptr, 16);
        const auto high_difference = high ^ changed_high;
        const auto low_difference = low ^ changed_low;
        total_bit_distance += std::popcount(high_difference) + std::popcount(low_difference);
        total_lane_distance += std::popcount(high ^ low);
        DJ_HOST_ASSERT(high_digests.emplace(high).second, "high hash lane collided at input {}", index);
        DJ_HOST_ASSERT(low_digests.emplace(low).second, "low hash lane collided at input {}", index);
        for (int bit = 0; bit < 64; ++bit) {
            output_one_counts[bit] += static_cast<int>((high >> bit) & 1);
            output_one_counts[64 + bit] += static_cast<int>((low >> bit) & 1);
            changed_bit_counts[bit] += static_cast<int>((high_difference >> bit) & 1);
            changed_bit_counts[64 + bit] += static_cast<int>((low_difference >> bit) & 1);
            lane_difference_counts[bit] += static_cast<int>(((high ^ low) >> bit) & 1);
        }
    }

    const auto average_bit_distance = static_cast<double>(total_bit_distance) / num_inputs;
    const auto average_lane_distance = static_cast<double>(total_lane_distance) / num_inputs;
    DJ_HOST_ASSERT(average_bit_distance > 60.0 and average_bit_distance < 68.0,
                   "weak hash diffusion: average bit distance is {}", average_bit_distance);
    DJ_HOST_ASSERT(average_lane_distance > 28.0 and average_lane_distance < 36.0,
                   "hash lanes are correlated: average bit distance is {}", average_lane_distance);
    const auto check_balanced_bits = [](const auto& counts, const std::string_view name) {
        for (std::size_t bit = 0; bit < counts.size(); ++bit) {
            DJ_HOST_ASSERT(counts[bit] > 3500 and counts[bit] < 6500,
                           "{} bit {} is biased: {} occurrences", name, bit, counts[bit]);
        }
    };
    check_balanced_bits(output_one_counts, "hash output");
    check_balanced_bits(changed_bit_counts, "hash avalanche");
    check_balanced_bits(lane_difference_counts, "hash lane difference");

    std::string benchmark_data(1 << 20, '\0');
    for (auto& value : benchmark_data)
        value = static_cast<char>(generator());
    constexpr int benchmark_iterations = 64;
    std::string benchmark_digest;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < benchmark_iterations; ++i)
        benchmark_digest = deep_jit::hash::get_hex_digest(benchmark_data);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    DJ_HOST_ASSERT(benchmark_digest.size() == 32);
    std::printf("Hash throughput: %.2f GiB/s\n",
                static_cast<double>(benchmark_data.size()) * benchmark_iterations / elapsed / (1ull << 30));
}

void test_memory_cache() {
    struct Value {
        int number;
    };

    deep_jit::MemCache<std::string, Value> cache;
    int num_factory_calls = 0;
    const auto first = cache.get_or_create("same", [&] {
        ++num_factory_calls;
        return std::make_shared<Value>(Value{7});
    });
    const auto second = cache.get_or_create("same", [&] {
        ++num_factory_calls;
        return std::make_shared<Value>(Value{8});
    });
    DJ_HOST_ASSERT(first == second and first->number == 7);
    DJ_HOST_ASSERT(num_factory_calls == 1, "memory cache called the factory on a hit");

    expect_failure(
        [&] {
            cache.get_or_create("failure", []() -> std::shared_ptr<Value> {
                throw std::runtime_error("factory failed");
            });
        },
        "factory failed");
    DJ_HOST_ASSERT(not cache.cache.contains("failure"), "failed factory populated the memory cache");
}

void test_filesystem(const fs::path& cache_root) {
    deep_jit::make_dirs(cache_root);
    const auto path = cache_root / "binary_data";
    const std::string expected("a\0b", 3);
    deep_jit::write_file_sync(path, expected);
    DJ_HOST_ASSERT(deep_jit::read(path) == expected, "binary file contents changed while reading");
    DJ_HOST_ASSERT(fs::remove(path), "failed to remove binary test file: {}", path.string());
    expect_failure([&] { deep_jit::read(path); }, "failed to open for reading");

    std::string all_bytes;
    all_bytes.reserve(256 * 257);
    for (int repeat = 0; repeat < 257; ++repeat) {
        for (int value = 0; value < 256; ++value)
            all_bytes.push_back(static_cast<char>(value));
    }
    const auto all_bytes_path = cache_root / "all_byte_values";
    deep_jit::write_file_sync(all_bytes_path, all_bytes);
    DJ_HOST_ASSERT(deep_jit::read(all_bytes_path) == all_bytes,
                   "binary file read stopped or changed data at an embedded null byte");
    DJ_HOST_ASSERT(fs::remove(all_bytes_path));

    const auto empty_path = cache_root / "empty_file";
    deep_jit::write_file_sync(empty_path, "");
    DJ_HOST_ASSERT(deep_jit::read(empty_path).empty(), "empty file was not read correctly");
    DJ_HOST_ASSERT(fs::remove(empty_path));

    DJ_HOST_ASSERT(not deep_jit::normalize_path(std::nullopt).has_value());
    DJ_HOST_ASSERT(not deep_jit::normalize_path(fs::path{}).has_value());
    DJ_HOST_ASSERT(deep_jit::normalize_path(fs::path("relative/../normalized")) ==
                   fs::absolute("normalized").lexically_normal());
    DJ_HOST_ASSERT(deep_jit::is_executable("/bin/sh"));

    const auto remove_root = cache_root / "safe_remove_all";
    deep_jit::make_dirs(remove_root / "nested");
    deep_jit::write_file_sync(remove_root / "nested/file", "payload");
    deep_jit::safe_remove_all(remove_root);
    DJ_HOST_ASSERT(not fs::exists(remove_root), "safe_remove_all did not remove the directory tree");

    const auto single_file = cache_root / "single_file";
    deep_jit::write_file_sync(single_file, "payload");
    DJ_HOST_ASSERT(not deep_jit::is_executable(single_file), "regular data file was reported as executable");
    deep_jit::safe_remove_all(single_file);
    DJ_HOST_ASSERT(not fs::exists(single_file), "safe_remove_all did not remove a single file");
    DJ_HOST_ASSERT(not deep_jit::try_update_mtime(single_file), "mtime update unexpectedly succeeded for a missing file");
    deep_jit::safe_remove_all(single_file);
}

void test_command_and_uuid() {
    DJ_HOST_ASSERT(deep_jit::call_external_command("sh -c 'printf stdout; printf stderr >&2'") == "stdoutstderr",
                   "external command did not capture stdout and stderr");
    DJ_HOST_ASSERT(deep_jit::call_external_command("printf '%01024d' 0").size() == 1024,
                   "external command output was truncated");
    expect_failure([] { deep_jit::call_external_command(""); }, "command must not be empty");
    expect_failure([] { deep_jit::call_external_command("sh -c 'printf failure; exit 7'"); },
                   "command failed with exit code 7");
    expect_failure([] { deep_jit::call_external_command("kill -TERM $$"); }, "exit code 143");

    const auto prefix = std::to_string(::getpid()) + "-";
    for (int i = 0; i < 16; ++i) {
        const auto uuid = deep_jit::get_uuid();
        DJ_HOST_ASSERT(uuid.starts_with(prefix), "UUID does not contain the process id: {}", uuid);
        DJ_HOST_ASSERT(uuid.size() == prefix.size() + 26, "unexpected UUID length: {}", uuid);
        const auto suffix = std::string_view(uuid).substr(prefix.size());
        DJ_HOST_ASSERT(suffix[8] == '-' and suffix[17] == '-', "UUID separators are misplaced: {}", uuid);
        for (std::size_t index = 0; index < suffix.size(); ++index) {
            if (index == 8 or index == 17)
                continue;
            const auto value = suffix[index];
            DJ_HOST_ASSERT((value >= '0' and value <= '9') or (value >= 'a' and value <= 'f'),
                           "UUID contains an invalid character: {}", uuid);
        }
    }
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

    // A cycle must be refused, and refused without leaving the visited set
    // dirty -- a stale entry there would make the NEXT parse of the same
    // source silently take the already-visited branch.
    const auto cycle_dir = get_test_maca_project_dir() / "include_cycle";
    deep_jit::Parser cycle_parser({cycle_dir}, {"test_maca/"});
    expect_failure([&] { cycle_parser.parse_into_hash("#include <test_maca/circular_include_entry.hpp>\n"); },
                   "circular include");
    DJ_HOST_ASSERT(cycle_parser.visiting.empty(), "circular include left stale parser state");
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
    // The load path names the entry point, so both ends of the count matter:
    // an artifact with two kernels and one with none.  Compiling such a source
    // is not itself an error -- the artifact is written either way -- so the
    // refusal belongs to the load, and each rejection is preceded by the
    // successful compile that proves it.
    const auto no_kernel_source = get_source("no_kernel.cu");
    check_artifact(runtime.compile_without_load("no_kernel", no_kernel_source), no_kernel_source);
    expect_failure([&] { (void)runtime.compile("no_kernel", no_kernel_source); },
                   "expected exactly one kernel");

    const auto source = get_source("multiple_kernels.cu");
    check_artifact(runtime.compile_without_load("multiple_kernels", source), source);
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

    // Every option that describes the launch must actually be named: unset is
    // not "use a default" on this path, it is a refusal, and each one is
    // refused by name.  `stream` is the deliberate exception -- unset means the
    // current stream -- so it is the seven below, not eight.
    const auto complete = [] {
        return LaunchOptions {
            .num_smem_bytes = 0,
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(1, 1, 1),
            .cluster_dim = dim3(1, 1, 1),
            .cooperative = false,
            .enable_pdl = false,
            .nonportable_cluster_size_allowed = false,
        };
    };
    const auto expect_missing = [&]<typename Member>(Member LaunchOptions::* field, const std::string_view message) {
        auto options = complete();
        (options.*field).reset();
        expect_failure([&] { kernel->launch(options, nullptr, 0); }, message);
    };
    expect_missing(&LaunchOptions::num_smem_bytes, "dynamic shared-memory size must be specified");
    expect_missing(&LaunchOptions::grid_dim, "grid dimension must be specified");
    expect_missing(&LaunchOptions::block_dim, "block dimension must be specified");
    expect_missing(&LaunchOptions::cluster_dim, "cluster dimension must be specified");
    expect_missing(&LaunchOptions::cooperative, "cooperative option must be specified");
    expect_missing(&LaunchOptions::enable_pdl, "PDL option must be specified");
    expect_missing(&LaunchOptions::nonportable_cluster_size_allowed, "non-portable cluster option must be specified");

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
        if (overrides.num_smem_bytes)
            options.num_smem_bytes = overrides.num_smem_bytes;
        if (overrides.grid_dim)
            options.grid_dim = overrides.grid_dim;
        if (overrides.block_dim)
            options.block_dim = overrides.block_dim;
        if (overrides.cluster_dim)
            options.cluster_dim = overrides.cluster_dim;
        if (overrides.enable_pdl)
            options.enable_pdl = overrides.enable_pdl;
        kernel->launch(options, nullptr, 0);
    };

    // The scalar validation that guards every launch: an option that does not
    // describe a launchable configuration is refused before it reaches the
    // driver, whoever set it.
    expect_failure([&] { launch_with(LaunchOptions {.num_smem_bytes = -1}); },
                   "must not be negative");
    expect_failure([&] { launch_with(LaunchOptions {.grid_dim = dim3(0, 1, 1)}); },
                   "grid dimensions must be positive");
    expect_failure([&] { launch_with(LaunchOptions {.block_dim = dim3(1, 0, 1)}); },
                   "block dimensions must be positive");
    expect_failure([&] { launch_with(LaunchOptions {.cluster_dim = dim3(1, 2, 1)}); },
                   "only one-dimensional MACA clusters are supported");
    expect_failure([&] { launch_with(LaunchOptions {.cluster_dim = dim3(2, 1, 1)}); },
                   "does not support cluster launch");
    expect_failure([&] { launch_with(LaunchOptions {.enable_pdl = true}); },
                   "does not support programmatic dependent launch");
    // Both rejections are unconditional: the backend reads no environment flag
    // here, because the only outcome of bypassing the check would be the same
    // driver failure (`mcErrorInvalidConfiguration`) one layer down.

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

void test_lazy_init() {
    struct Value {
        int value;
    };

    int num_initializations = 0;
    deep_jit::LazyInit<Value> lazy([&] {
        ++num_initializations;
        return std::make_shared<Value>(Value{17});
    });
    DJ_HOST_ASSERT(num_initializations == 0, "lazy value was initialized eagerly");
    DJ_HOST_ASSERT(lazy->value == 17 and lazy.get()->value == 17, "lazy value is incorrect");
    DJ_HOST_ASSERT(num_initializations == 1, "lazy value was initialized more than once");

    const auto first_value = lazy.get();
    lazy = deep_jit::LazyInit<Value>([] { return std::make_shared<Value>(Value{29}); });
    DJ_HOST_ASSERT(lazy->value == 29, "reassigned lazy factory was not used");
    DJ_HOST_ASSERT(first_value->value == 17, "reassigning lazy initialization invalidated an existing shared owner");

    deep_jit::LazyInit<Value> empty(nullptr);
    expect_failure([&] { (void)empty.get(); }, "lazy object must be initialized before use");

    deep_jit::LazyInit<Value> null_factory([] { return std::shared_ptr<Value>(); });
    expect_failure([&] { (void)null_factory.get(); }, "lazy factory must not return nullptr");

    int num_attempts = 0;
    deep_jit::LazyInit<Value> retry([&] {
        if (++num_attempts == 1)
            throw std::runtime_error("initialization failed");
        return std::make_shared<Value>(Value{41});
    });
    expect_failure([&] { (void)retry.get(); }, "initialization failed");
    DJ_HOST_ASSERT(retry->value == 41 and num_attempts == 2, "lazy initialization did not recover after an exception");

    // Creating a lazy runtime must not discover the toolchain, read the
    // environment or construct the runtime: the invalid cache-root list is
    // only observed on first use.
    set_env("LAZY_JIT_CACHE_DIR", "first::second");
    auto lazy_jit = deep_jit::create_lazy_jit<deep_jit::MACA>(deep_jit::Config(
        get_test_maca_project_dir(), "LAZY", "maca-test-signature",
        {get_test_maca_project_dir() / "kernels"}, {"test_maca/"}));
    expect_failure([&] { (void)lazy_jit.get(); }, "disk cache path list contains an empty path");
    unset_env("LAZY_JIT_CACHE_DIR");

    int value = 0;
    DJ_HOST_ASSERT(deep_jit::maca::kernel_arg_pointer(value) == &value);
    const deep_jit::NoRefPtr no_ref {&value};
    DJ_HOST_ASSERT(deep_jit::maca::kernel_arg_pointer(no_ref) == &value);
    const deep_jit::NoRefPtr null_no_ref;
    DJ_HOST_ASSERT(deep_jit::maca::kernel_arg_pointer(null_no_ref) == nullptr);
}

void test_mixed_arguments(Runtime& runtime) {
    const auto kernel = runtime.compile("mixed_arguments", get_source("mixed_arguments.cu"));

    int* device_input = nullptr;
    float* device_output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&device_input), sizeof(int)));
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&device_output), sizeof(float)));
    const int base = 64;
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(device_input, &base, sizeof(int), mcMemcpyHostToDevice));

    // Every term is exactly representable in binary32, so the sum has to match
    // bit for bit: 1 + 2.5 + 4.25 + 8 + 16 + 32 + 1 (bool) + 256 + 64 = 384.75.
    runtime.launch(kernel, {.grid_dim = dim3(1, 1, 1), .block_dim = dim3(1, 1, 1)},
                   device_output, 1, 2.5f, 4.25, static_cast<long long>(8), static_cast<short>(16),
                   static_cast<unsigned char>(32), true, static_cast<unsigned long long>(256), device_input);
    DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());

    float result = 0.0f;
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, device_output, sizeof(float), mcMemcpyDeviceToHost));
    DJ_HOST_ASSERT(result == 384.75f, "mixed kernel arguments produced the wrong value: {}", result);

    // The same call through `NoRefPtr`, which hands the driver the pointer
    // itself as the argument's storage.  For a pointer-sized argument this is
    // observably the same value as the plain form -- the wrapper's address
    // holds the same bytes the plain form's storage does -- so what this
    // launch proves is that the wrapper is ACCEPTED here, not that it changes
    // what the driver reads.  That property needs an argument wider than the
    // wrapper, and is proven in `large kernel arguments`.
    int* input_argument = device_input;
    runtime.launch(kernel, {.grid_dim = dim3(1, 1, 1), .block_dim = dim3(1, 1, 1)},
                   device_output, 1, 2.5f, 4.25, static_cast<long long>(8), static_cast<short>(16),
                   static_cast<unsigned char>(32), true, static_cast<unsigned long long>(256),
                   deep_jit::NoRefPtr {&input_argument});
    DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
    DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, device_output, sizeof(float), mcMemcpyDeviceToHost));

    DJ_MACA_RUNTIME_CHECK(mcFree(device_input));
    DJ_MACA_RUNTIME_CHECK(mcFree(device_output));
    DJ_HOST_ASSERT(result == 384.75f, "a NoRefPtr kernel argument produced the wrong value: {}", result);
}

void test_resource_usage_checks(Runtime& runtime) {
    const CompilerOptions spill_check {.check_no_spills = true};
    const CompilerOptions local_memory_check {.check_no_local_memory = true};
    const CompilerOptions both_checks {.check_no_spills = true, .check_no_local_memory = true};

    // mxcc only reports "N bytes stack frame" under `-resource-usage`, so a
    // check that validates that report has to request it; otherwise it would
    // silently pass on a kernel that spills.
    const std::vector<std::string> expected = {
        "--offload-arch=xcore" + *runtime.default_compiler_options.arch,
        "-O3",
        "-resource-usage",
        "-std=c++20",
    };
    for (const auto* options: {&spill_check, &local_memory_check, &both_checks})
        DJ_HOST_ASSERT(runtime.default_compiler_options.override_with(*options).get_flags() == expected,
                       "a resource check did not request the report it validates");

    // A kernel that does not spill passes...
    const CompilerOptions effective = runtime.default_compiler_options.override_with(both_checks);
    const auto clean = runtime.compile_without_load("resource_checks_clean", get_source("scalar_increment.cu"), both_checks);
    DJ_HOST_ASSERT(fs::is_regular_file(clean / "kernel.devbin"));

    // ... and a kernel with a large per-thread frame is rejected, by either
    // knob.  The two coincide on MACA, and this is measured, not assumed: an
    // 8 KiB-frame kernel's whole `-resource-usage` report is
    //
    //   maca info  : Function properties for  spill_kernel
    //     8200 bytes stack frame
    //   maca info  : Used  36 MTregisters, 38 STregisters, 0 bytes shared mem
    //   maca info  : staticMaxWarps/PEU : 8
    //
    // -- there is no local-memory line for a CUDA-style regex to match, which
    // is why `check_no_local_memory` validates the frame size too.
    const auto spill_source = get_source("local_frame.cu");
    expect_failure([&] { (void)runtime.compile_without_load("resource_checks_spill", spill_source, spill_check); },
                   "mxcc reported a non-empty stack frame");
    expect_failure([&] { (void)runtime.compile_without_load("resource_checks_local", spill_source, local_memory_check); },
                   "mxcc reported a non-empty stack frame");

    // The check is part of the cache key: enabling it selects another entry.
    DJ_HOST_ASSERT(runtime.cache_key(spill_source, effective) !=
                       runtime.cache_key(spill_source, runtime.default_compiler_options),
                   "enabling a resource check did not affect the cache key");
}

void test_post_hook(Runtime& runtime, const fs::path& cache_root) {
    const auto source = get_source("scalar_increment.cu");
    DJ_HOST_ASSERT(runtime.default_compiler_options.get_post_hook_hash(runtime.config).empty(),
                   "the runtime default must not have a post hook");

    const auto without_hook = runtime.compile_without_load("post_hook", source);
    DJ_HOST_ASSERT(not fs::exists(without_hook / "post_hook.marker"),
                   "the hook marker exists without a post hook");

    const CompilerOptions options {.post_hook = "scripts/post_hook.py"};
    const auto expected_hook_hash = deep_jit::hash::FNV1a()
        .update("scripts/post_hook.py")
        .update(deep_jit::read(runtime.config.get_python_path("scripts/post_hook.py")))
        .get_hex_digest();
    const auto hook_hash = options.get_post_hook_hash(runtime.config);
    DJ_HOST_ASSERT(hook_hash == expected_hook_hash and options.get_post_hook_hash(runtime.config) == hook_hash,
                   "post hook hash was not stable");

    const auto artifact = runtime.compile_without_load("post_hook", source, options);
    DJ_HOST_ASSERT(artifact != without_hook, "post hook must change the cache key");
    DJ_HOST_ASSERT(fs::is_regular_file(artifact / "post_hook.marker"),
                   "the post hook did not run inside the artifact directory");
    DJ_HOST_ASSERT(deep_jit::read(artifact / "post_hook.marker") == "first\n",
                   "the post hook recorded the wrong marker");
    DJ_HOST_ASSERT(deep_jit::read(artifact / "meta.json").find("\"post_hook\":\"scripts/post_hook.py\"") !=
                       std::string::npos,
                   "metadata does not record the post hook");

    // The hook edits the artifact in place, and that edit is what gets
    // published.  Without this the case cannot tell a runtime that runs the
    // hook from one that runs it and then throws its work away.
    const auto marker = deep_jit::read(artifact / "kernel.devbin");
    DJ_HOST_ASSERT(marker.find("DJ_POST_HOOK_MARKER") != std::string::npos,
                   "the post hook's edit of the device binary was not published");
    DJ_HOST_ASSERT(deep_jit::read(without_hook / "kernel.devbin").find("DJ_POST_HOOK_MARKER") ==
                       std::string::npos,
                   "a device binary carries a hook marker without a post hook");

    // The hook runs before publication and the artifact is loaded afterwards,
    // so the device binary must still be loadable.
    DJ_HOST_ASSERT(launch_value(runtime, runtime.compile("post_hook", source, options), 41) == 42,
                   "a kernel published through a post hook did not launch");

    const CompilerOptions alt_options {.post_hook = "scripts/post_hook_alt.py"};
    DJ_HOST_ASSERT(alt_options.get_post_hook_hash(runtime.config) != hook_hash,
                   "different post hooks must have different hashes");
    const auto alt_artifact = runtime.compile_without_load("post_hook", source, alt_options);
    DJ_HOST_ASSERT(alt_artifact != artifact, "a different post hook must change the cache key");
    DJ_HOST_ASSERT(deep_jit::read(alt_artifact / "post_hook.marker") == "second\n",
                   "the second post hook did not run");
    // ... and its artifact carries ITS marker, so the two hooks cannot be
    // confused for one another.
    const auto alt_marker = deep_jit::read(alt_artifact / "kernel.devbin");
    DJ_HOST_ASSERT(alt_marker.find("DJ_POST_HOOK_ALT_MARKER") != std::string::npos and
                       alt_marker.find("DJ_POST_HOOK_MARKER") == std::string::npos,
                   "the second post hook's edit of the device binary was not published");

    // A hook that fails takes the compilation with it: the caller sees the
    // hook's exit status, nothing is published, and nothing stays staged.
    const auto failing_cache = cache_root / "post_hook_failure";
    set_env("POST_HOOK_FAILURE_JIT_CACHE_DIR", failing_cache.string());
    const auto failing_runtime = make_runtime_with_prefix("POST_HOOK_FAILURE",
                                                          get_test_maca_project_dir() / "kernels");
    const CompilerOptions failing_options {.post_hook = "scripts/failing_post_hook.py"};
    expect_failure(
        [&] { (void)failing_runtime->compile_without_load("post_hook_failure", source, failing_options); },
        "exit code 7");
    const auto failing_entries = failing_runtime->disk_cache.paths.front() / "cache";
    if (fs::exists(failing_entries)) {
        for (const auto& entry: fs::directory_iterator(failing_entries))
            DJ_HOST_ASSERT(not entry.path().filename().string().starts_with("post_hook_failure."),
                           "a failed post hook published a cache entry");
    }
    check_tmp_is_empty(failing_runtime->disk_cache.paths.front());
    unset_env("POST_HOOK_FAILURE_JIT_CACHE_DIR");

    // The digest tracks the hook contents, not the installation path, and is
    // cached per thread for the lifetime of the process.
    const auto same_root = cache_root / "hook_root_same";
    const auto changed_root = cache_root / "hook_root_changed";
    deep_jit::make_dirs(same_root / "scripts");
    deep_jit::make_dirs(changed_root / "scripts");
    const auto hook_source = deep_jit::read(runtime.config.get_python_path("scripts/post_hook.py"));
    deep_jit::write_file_sync(same_root / "scripts/post_hook.py", hook_source);
    deep_jit::write_file_sync(changed_root / "scripts/post_hook.py", "print('changed')\n");
    const deep_jit::Config same_config(same_root, "HOOK_SAME", "maca-test-signature",
                                       {get_test_maca_project_dir() / "kernels"}, {"test_maca/"});
    const deep_jit::Config changed_config(changed_root, "HOOK_CHANGED", "maca-test-signature",
                                          {get_test_maca_project_dir() / "kernels"}, {"test_maca/"});
    DJ_HOST_ASSERT(options.get_post_hook_hash(runtime.config) == options.get_post_hook_hash(same_config),
                   "equal post hook contents must be installation-path independent");
    DJ_HOST_ASSERT(options.get_post_hook_hash(runtime.config) != options.get_post_hook_hash(changed_config),
                   "changed post hook contents must change the hash across runtimes");

    // A missing hook fails before anything is compiled.
    expect_failure([&] {
        (void)runtime.compile_without_load("missing_post_hook", source,
                                           CompilerOptions {.post_hook = "scripts/missing_post_hook.py"});
    }, "failed to open for reading");
}

void test_generated_include_graph(const fs::path& cache_root) {
    const auto root_a = cache_root / "generated_includes_a";
    const auto root_b = cache_root / "generated_includes_b";
    const auto root_changed = cache_root / "generated_includes_changed";
    for (const auto& root: {root_a, root_b, root_changed})
        deep_jit::make_dirs(root / "test_maca");

    const auto dependency = [](const int value) {
        return std::format("constexpr int kTrackedDependency = {};\n", value);
    };
    deep_jit::write_file_sync(root_a / "test_maca/tracked_dependency.cu", dependency(1));
    deep_jit::write_file_sync(root_b / "test_maca/tracked_dependency.cu", dependency(1));
    deep_jit::write_file_sync(root_changed / "test_maca/tracked_dependency.cu", dependency(2));

    const auto source = std::string(
        "#include <test_maca/tracked_dependency.cu>\n"
        "extern \"C\" __global__ void tracked_kernel(int* output, const int input) {\n"
        "    output[0] = input + kTrackedDependency;\n"
        "}\n");

    set_env("INCLUDE_A_JIT_CACHE_DIR", (cache_root / "include_shared").string());
    set_env("INCLUDE_B_JIT_CACHE_DIR", (cache_root / "include_shared").string());
    set_env("INCLUDE_CHANGED_JIT_CACHE_DIR", (cache_root / "include_shared").string());
    const auto runtime_a = make_runtime_with_prefix("INCLUDE_A", root_a);
    const auto runtime_b = make_runtime_with_prefix("INCLUDE_B", root_b);
    const auto runtime_changed = make_runtime_with_prefix("INCLUDE_CHANGED", root_changed);

    // The include directory is not part of the digest, so two installations
    // carrying the same header contents share one entry...
    const auto artifact_a = runtime_a->compile_without_load("tracked_include", source);
    DJ_HOST_ASSERT(runtime_b->compile_without_load("tracked_include", source) == artifact_a,
                   "equal tracked include contents must share a cache entry");
    // ... while changed contents produce a new entry, and a new compiled
    // behaviour rather than only a new digest.
    DJ_HOST_ASSERT(runtime_changed->compile_without_load("tracked_include", source) != artifact_a,
                   "a changed tracked include must produce a new cache entry");
    DJ_HOST_ASSERT(launch_value(*runtime_a, runtime_a->compile("tracked_include", source), 41) == 42,
                   "the tracked include was not compiled in");
    DJ_HOST_ASSERT(launch_value(*runtime_changed, runtime_changed->compile("tracked_include", source), 41) == 43,
                   "the changed tracked include was not compiled in");

    unset_env("INCLUDE_A_JIT_CACHE_DIR");
    unset_env("INCLUDE_B_JIT_CACHE_DIR");
    unset_env("INCLUDE_CHANGED_JIT_CACHE_DIR");
}

void test_compiler_failure_cleanup(Runtime& runtime) {
    const auto source = std::string("extern \"C\" __global__ void broken_kernel(int* output) { this is not C++ }\n");

    const auto memory_cache_size = runtime.mem_cache.cache.size();
    expect_failure([&] { (void)runtime.compile_without_load("compiler_failure", source); },
                   "command failed with exit code");
    expect_failure([&] { (void)runtime.compile("compiler_failure", source); },
                   "command failed with exit code");
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == memory_cache_size,
                   "a failed compilation populated the memory cache");

    const auto cache_dir = runtime.disk_cache.paths.front() / "cache";
    if (fs::exists(cache_dir)) {
        for (const auto& entry: fs::directory_iterator(cache_dir))
            DJ_HOST_ASSERT(not entry.path().filename().string().starts_with("compiler_failure."),
                           "a failed compilation was published: {}", entry.path().string());
    }
    const auto temporary_dir = runtime.disk_cache.paths.front() / "tmp";
    DJ_HOST_ASSERT(not fs::exists(temporary_dir) or fs::is_empty(temporary_dir),
                   "a failed compilation left a temporary artifact behind");
}

void test_invalid_cache_tag(Runtime& runtime) {
    const auto source = get_source("scalar_increment.cu");
    expect_failure([&] { (void)runtime.compile_without_load("invalid-tag", source); },
                   "cache tag must contain only letters, digits, or underscores");
    expect_failure([&] { (void)runtime.compile_without_load("", source); },
                   "cache tag must contain only letters, digits, or underscores");
}

void test_secondary_disk_cache(const fs::path& cache_root) {
    const auto primary_cache = cache_root / "secondary_lookup_primary";
    const auto secondary_cache = cache_root / "secondary_lookup_secondary";
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto source = get_template_source(1);

    set_env("SECONDARY_SOURCE_JIT_CACHE_DIR", secondary_cache.string());
    const auto source_runtime = make_runtime_with_prefix("SECONDARY_SOURCE", include_dir);
    const auto expected_artifact = source_runtime->compile_without_load("template_add", source);
    unset_env("SECONDARY_SOURCE_JIT_CACHE_DIR");

    // A lookup root may be read-only: replace its commit marker with a
    // symlink whose mtime cannot be updated and check that the hit still
    // resolves instead of turning into a miss or an error.
    const auto commit_path = expected_artifact / deep_jit::kCommitFileName;
    DJ_HOST_ASSERT(fs::remove(commit_path));
    fs::create_symlink("/proc/version", commit_path);
    DJ_HOST_ASSERT(not deep_jit::try_update_mtime(commit_path),
                   "read-only secondary cache marker unexpectedly allowed an mtime update");

    set_env("SECONDARY_LOOKUP_JIT_CACHE_DIR", primary_cache.string() + ":" + secondary_cache.string());
    const auto runtime = make_runtime_with_prefix("SECONDARY_LOOKUP", include_dir);
    DJ_HOST_ASSERT(runtime->compile_without_load("template_add", source) == expected_artifact,
                   "runtime did not reuse the secondary cache entry");
    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("template_add", source), 4) == 5,
                   "a kernel loaded from the secondary cache root did not launch");
    DJ_HOST_ASSERT(not fs::exists(primary_cache / "cache"),
                   "a secondary cache hit unexpectedly wrote to the primary cache root");
    unset_env("SECONDARY_LOOKUP_JIT_CACHE_DIR");
}

void test_architecture_override(Runtime& runtime) {
    const auto source = get_template_source(38);
    const auto default_arch = *runtime.default_compiler_options.arch;
    const auto default_artifact = runtime.compile_without_load("arch_override", source);
    DJ_HOST_ASSERT(launch_value(runtime, runtime.compile("arch_override", source), 1) == 39,
                   "a kernel compiled for the discovered architecture did not launch");

    // A cross-family offload target must reach mxcc and select a distinct
    // entry.  The artifact is deliberately not loaded: a binary built for
    // another xcore family cannot run on this device.
    const auto other_arch = default_arch == "1000" ? std::string("1600") : std::string("1000");
    const CompilerOptions options {.arch = other_arch};
    const auto artifact = runtime.compile_without_load("arch_override", source, options);
    DJ_HOST_ASSERT(artifact != default_artifact, "the architecture override did not change the cache key");
    const auto metadata = deep_jit::read(artifact / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"arch\":\"" + other_arch + "\"") != std::string::npos,
                   "the architecture override is missing from metadata");
    DJ_HOST_ASSERT(metadata.find("--offload-arch=xcore" + other_arch) != std::string::npos,
                   "the offload target is missing from the recorded command");
}

// Every xcore family is a separate mxcc offload target, and mxcc produces an
// artifact for any of them regardless of which device is installed -- measured
// on a C500 (xcore1000) host, `--offload-arch=xcore1500` and `=xcore1600` both
// compile.  So the whole family matrix is verifiable on a single machine; only
// the installed family's artifact can be *loaded*, which is why this case stops
// at compilation.
void test_cross_family_compilation(Runtime& runtime, const fs::path& cache_root) {
    const std::array<std::string, 3> families = {"1000", "1500", "1600"};
    const auto native_arch = runtime.device.get_arch();
    DJ_HOST_ASSERT(std::ranges::find(families, native_arch) != families.end(),
                   "the device reported an unknown xcore family: {}", native_arch);
    const auto source = get_template_source(29);

    std::vector<fs::path> artifacts;
    for (const auto& family : families) {
        const CompilerOptions options {.arch = family};
        const auto artifact = runtime.compile_without_load("cross_family", source, options);
        check_artifact(artifact, source);
        const auto metadata = deep_jit::read(artifact / "meta.json");
        DJ_HOST_ASSERT(metadata.find("--offload-arch=xcore" + family) != std::string::npos,
                       "the offload target for xcore{} is missing from the recorded command", family);
        DJ_HOST_ASSERT(metadata.find("\"arch\":\"" + family + "\"") != std::string::npos,
                       "the architecture for xcore{} is missing from metadata", family);
        artifacts.emplace_back(artifact);
    }
    // Three distinct offload targets are three distinct cache entries -- the
    // architecture is part of the flags, so it is part of the digest.
    std::vector<fs::path> sorted_artifacts = artifacts;
    std::ranges::sort(sorted_artifacts);
    DJ_HOST_ASSERT(std::ranges::adjacent_find(sorted_artifacts) == sorted_artifacts.end(),
                   "distinct xcore families shared a cache entry");

    // Recompiling the same family must reuse its entry rather than recompile.
    const CompilerOptions native_options {.arch = native_arch};
    DJ_HOST_ASSERT(runtime.compile_without_load("cross_family", source, native_options) ==
                       artifacts[std::distance(families.begin(), std::ranges::find(families, native_arch))],
                   "the native family did not reuse its cache entry");

    // The installed family's artifact is the one that must actually run.
    const auto native_kernel = runtime.compile("cross_family", source, native_options);
    DJ_HOST_ASSERT(launch_value(runtime, native_kernel, 1) == 30,
                   "the native-family artifact did not launch");
}

void test_multiple_runtimes(const fs::path& cache_root) {
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto source = get_source("compiler_options.cu");
    const auto cache_a = cache_root / "runtime_a";
    const auto cache_b = cache_root / "runtime_b";
    set_env("RUNTIME_A_JIT_CACHE_DIR", cache_a.string());
    set_env("RUNTIME_B_JIT_CACHE_DIR", cache_b.string());

    const auto runtime_a = make_runtime_with_prefix("RUNTIME_A", include_dir);
    const auto runtime_b = make_runtime_with_prefix("RUNTIME_B", include_dir);
    runtime_a->default_compiler_options.extra_mxcc_flags.emplace_back("-DTEST_OPTION=11");
    runtime_b->default_compiler_options.extra_mxcc_flags.emplace_back("-DTEST_OPTION=29");
    runtime_a->default_launch_options.cooperative = true;
    DJ_HOST_ASSERT(runtime_b->default_launch_options.cooperative == false,
                   "default launch options leaked between runtimes");
    runtime_a->default_launch_options.cooperative = false;
    DJ_HOST_ASSERT(runtime_a->disk_cache.paths.front() == cache_a and runtime_b->disk_cache.paths.front() == cache_b,
                   "runtime cache roots are not isolated");
    DJ_HOST_ASSERT(runtime_a->cache_key(source, runtime_a->default_compiler_options) !=
                       runtime_b->cache_key(source, runtime_b->default_compiler_options),
                   "runtime-specific compiler defaults must affect the cache key");
    DJ_HOST_ASSERT(launch_value(*runtime_a, runtime_a->compile("runtime_isolation", source)) == 11);
    DJ_HOST_ASSERT(runtime_b->mem_cache.cache.empty(), "one runtime populated another runtime's memory cache");
    DJ_HOST_ASSERT(launch_value(*runtime_b, runtime_b->compile("runtime_isolation", source)) == 29);

    // The toolkit is resolved per load, not carried on the backend: MACA finds
    // the entry-point name with `llvm-nm` out of the toolkit, so a runtime that
    // borrowed another's would be reading a different toolchain's binary.
    // Runtime B gets an `llvm-nm` that records that it ran and then delegates
    // to the real one, and B is constructed LAST -- the order that matters if
    // the toolkit were published to a process-global by the constructor.
    const auto toolkit_marker = cache_root / "runtime_b_llvm_nm_ran";
    const auto llvm_nm_shim = cache_root / "llvm_nm_shim";
    write_executable(llvm_nm_shim,
                     "#!/bin/sh\n"
                     "touch \"" + toolkit_marker.string() + "\"\n"
                     "exec \"" + runtime_a->backend.toolkit.llvm_nm.string() + "\" \"$@\"\n");
    set_env("RUNTIME_B_JIT_LLVM_NM", llvm_nm_shim.string());
    const auto toolkit_a = make_runtime_with_prefix("RUNTIME_A", include_dir);
    const auto toolkit_b = make_runtime_with_prefix("RUNTIME_B", include_dir);
    DJ_HOST_ASSERT(toolkit_a->backend.toolkit.llvm_nm != toolkit_b->backend.toolkit.llvm_nm,
                   "the two runtimes did not resolve different toolkits");

    const auto toolkit_source = get_template_source(23);
    (void)toolkit_a->compile_without_load("toolkit_isolation", toolkit_source);
    (void)toolkit_b->compile_without_load("toolkit_isolation", toolkit_source);
    DJ_HOST_ASSERT(not fs::exists(toolkit_marker), "llvm-nm ran without a load");
    DJ_HOST_ASSERT(launch_value(*toolkit_a, toolkit_a->compile("toolkit_isolation", toolkit_source), 1) == 24,
                   "runtime A did not load");
    DJ_HOST_ASSERT(not fs::exists(toolkit_marker), "runtime A loaded through runtime B's toolkit");
    DJ_HOST_ASSERT(launch_value(*toolkit_b, toolkit_b->compile("toolkit_isolation", toolkit_source), 1) == 24,
                   "runtime B did not load");
    DJ_HOST_ASSERT(fs::exists(toolkit_marker), "runtime B did not use its own llvm-nm");
    unset_env("RUNTIME_B_JIT_LLVM_NM");

    // Two runtimes over the same root share the disk cache but not the memory
    // cache, and the process environment is re-read rather than snapshotted
    // for anything but the runtime's own construction-time settings.
    const auto shared_cache = cache_root / "shared_runtime";
    set_env("SHARED_RUNTIME_JIT_CACHE_DIR", shared_cache.string());
    const auto shared_a = make_runtime_with_prefix("SHARED_RUNTIME", include_dir);
    const auto shared_b = make_runtime_with_prefix("SHARED_RUNTIME", include_dir);
    const auto shared_source = get_template_source(13);
    DJ_HOST_ASSERT(shared_a->compile_without_load("shared_runtime", shared_source) ==
                       shared_b->compile_without_load("shared_runtime", shared_source),
                   "equivalent runtimes did not share the disk cache");
    const auto kernel_a = shared_a->compile("shared_runtime", shared_source);
    const auto kernel_b = shared_b->compile("shared_runtime", shared_source);
    DJ_HOST_ASSERT(kernel_a != kernel_b, "different runtimes unexpectedly shared the memory cache");
    DJ_HOST_ASSERT(launch_value(*shared_a, kernel_a, 1) == 14 and launch_value(*shared_b, kernel_b, 2) == 15);

    const auto snapshot_cache_a = cache_root / "snapshot_a";
    const auto snapshot_cache_b = cache_root / "snapshot_b";
    set_env("SNAPSHOT_JIT_CACHE_DIR", snapshot_cache_a.string());
    set_env("SNAPSHOT_JIT_CPP_STANDARD", "17");
    const auto snapshot_a = make_runtime_with_prefix("SNAPSHOT", include_dir);
    set_env("SNAPSHOT_JIT_CACHE_DIR", snapshot_cache_b.string());
    set_env("SNAPSHOT_JIT_CPP_STANDARD", "23");
    const auto snapshot_b = make_runtime_with_prefix("SNAPSHOT", include_dir);
    DJ_HOST_ASSERT(snapshot_a->disk_cache.paths.front() == snapshot_cache_a and
                       snapshot_b->disk_cache.paths.front() == snapshot_cache_b,
                   "runtime construction did not snapshot its cache environment");
    DJ_HOST_ASSERT(std::ranges::find(*snapshot_a->default_compiler_options.mxcc_flags, "-std=c++17") !=
                       snapshot_a->default_compiler_options.mxcc_flags->end());
    DJ_HOST_ASSERT(std::ranges::find(*snapshot_b->default_compiler_options.mxcc_flags, "-std=c++23") !=
                       snapshot_b->default_compiler_options.mxcc_flags->end());
    DJ_HOST_ASSERT(snapshot_a->env.get<int>("JIT_CPP_STANDARD") == 23,
                   "runtime Env should continue to reflect process environment changes");

    unset_env("RUNTIME_A_JIT_CACHE_DIR");
    unset_env("RUNTIME_B_JIT_CACHE_DIR");
    unset_env("SHARED_RUNTIME_JIT_CACHE_DIR");
    unset_env("SNAPSHOT_JIT_CACHE_DIR");
    unset_env("SNAPSHOT_JIT_CPP_STANDARD");
}

void test_dump_options_on_cache_hit(const fs::path& cache_root) {
    const auto dump_cache = cache_root / "dump_cache_hit";
    set_env("DUMP_CACHE_HIT_JIT_CACHE_DIR", dump_cache.string());
    const auto runtime = make_runtime_with_prefix("DUMP_CACHE_HIT", get_test_maca_project_dir() / "kernels");
    const auto source = get_template_source(55);
    const auto artifact = runtime->compile_without_load("dump_cache_hit", source);
    DJ_HOST_ASSERT(not fs::exists(artifact / "kernel.s"), "the assembly dump ran without being requested");

    // Dumps are outside the cache digest and only run when a compilation
    // actually happens, so a cache hit is reused as it stands.
    const CompilerOptions dump_options {.dump_asm = true};
    DJ_HOST_ASSERT(runtime->compile_without_load("dump_cache_hit", source, dump_options) == artifact,
                   "dump options unexpectedly changed the cache key");
    DJ_HOST_ASSERT(not fs::exists(artifact / "kernel.s"),
                   "dump options unexpectedly rebuilt an existing cache entry");
    const auto metadata = deep_jit::read(artifact / "meta.json");
    DJ_HOST_ASSERT(metadata.find("\"dump_asm\":false") != std::string::npos,
                   "cache-hit metadata must describe the original compilation");
    unset_env("DUMP_CACHE_HIT_JIT_CACHE_DIR");
}

void test_disk_cache(const fs::path& cache_root) {
    const auto root = cache_root / "disk_cache";
    const auto primary = root / "primary";
    const auto secondary = root / "secondary";
    deep_jit::DiskCache cache({primary, secondary});
    expect_failure([] { deep_jit::DiskCache(std::vector<fs::path>{}); }, "not this->paths.empty()");

    fs::path abandoned_path;
    {
        auto entry = cache.entry("abandoned", "digest");
        DJ_HOST_ASSERT(not entry.hit);
        abandoned_path = entry.path;
        deep_jit::write_file_sync(entry.path / "partial", "partial");
    }
    DJ_HOST_ASSERT(not fs::exists(abandoned_path), "uncommitted cache entry was not cleaned");

    fs::path committed_path;
    {
        auto entry = cache.entry("committed", "digest");
        DJ_HOST_ASSERT(not entry.hit);
        deep_jit::write_file_sync(entry.path / "payload", "complete");
        committed_path = entry.commit();
        DJ_HOST_ASSERT(entry.commit() == committed_path, "cache commit must be idempotent");
    }
    DJ_HOST_ASSERT(fs::is_regular_file(committed_path / deep_jit::kCommitFileName));
    DJ_HOST_ASSERT(deep_jit::read(committed_path / "payload") == "complete");
    {
        auto entry = cache.entry("committed", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == committed_path, "committed cache entry was not reused");
        DJ_HOST_ASSERT(entry.commit() == committed_path, "committing a cache hit changed its path");
    }

    const auto secondary_path = secondary / "cache/secondary.digest";
    deep_jit::make_dirs(secondary_path);
    deep_jit::write_file_sync(secondary_path / deep_jit::kCommitFileName, "");
    deep_jit::write_file_sync(secondary_path / "payload", "secondary");
    {
        auto entry = cache.entry("secondary", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == secondary_path, "secondary cache root was not searched");
    }

    const auto primary_priority_path = primary / "cache/priority.digest";
    const auto secondary_priority_path = secondary / "cache/priority.digest";
    for (const auto& path : {primary_priority_path, secondary_priority_path}) {
        deep_jit::make_dirs(path);
        deep_jit::write_file_sync(path / deep_jit::kCommitFileName, "");
    }
    deep_jit::write_file_sync(primary_priority_path / "payload", "primary");
    deep_jit::write_file_sync(secondary_priority_path / "payload", "secondary");
    {
        auto entry = cache.entry("priority", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == primary_priority_path,
                       "the first valid cache root must take priority");
    }

    const auto incomplete_primary_path = primary / "cache/incomplete_primary.digest";
    const auto valid_secondary_path = secondary / "cache/incomplete_primary.digest";
    deep_jit::make_dirs(incomplete_primary_path);
    deep_jit::write_file_sync(incomplete_primary_path / "partial", "partial");
    deep_jit::make_dirs(valid_secondary_path);
    deep_jit::write_file_sync(valid_secondary_path / deep_jit::kCommitFileName, "");
    deep_jit::write_file_sync(valid_secondary_path / "payload", "secondary");
    {
        auto entry = cache.entry("incomplete_primary", "digest");
        DJ_HOST_ASSERT(entry.hit and entry.path == valid_secondary_path,
                       "an uncommitted primary entry hid a valid secondary entry");
    }

    const auto incomplete_secondary_path = secondary / "cache/incomplete_secondary.digest";
    deep_jit::make_dirs(incomplete_secondary_path);
    deep_jit::write_file_sync(incomplete_secondary_path / "partial", "partial");
    fs::path temporary_path;
    {
        auto entry = cache.entry("incomplete_secondary", "digest");
        DJ_HOST_ASSERT(not entry.hit and entry.path.parent_path() == primary / "tmp",
                       "an uncommitted secondary entry was treated as a cache hit");
        temporary_path = entry.path;
    }
    DJ_HOST_ASSERT(not fs::exists(temporary_path), "temporary cache miss was not cleaned");
    DJ_HOST_ASSERT(fs::is_regular_file(incomplete_secondary_path / "partial"),
                   "cache lookup modified an uncommitted secondary entry");

    const auto stale_temporary_path = primary / "tmp/stale";
    deep_jit::make_dirs(stale_temporary_path / "nested");
    deep_jit::write_file_sync(stale_temporary_path / "nested/payload", "stale");
    {
        auto entry = cache.entry("ignores_stale_tmp", "digest");
        DJ_HOST_ASSERT(not entry.hit and entry.path != stale_temporary_path,
                       "a stale temporary directory was reused");
    }
    DJ_HOST_ASSERT(fs::is_regular_file(stale_temporary_path / "nested/payload"),
                   "cache entry cleanup removed another writer's temporary directory");
    deep_jit::safe_remove_all(stale_temporary_path);

    expect_failure([&] { (void)cache.entry("", "digest"); }, "cache tag must contain only");
    expect_failure([&] { (void)cache.entry("path/tag", "digest"); }, "cache tag must contain only");
    expect_failure([&] { (void)cache.entry("dot.tag", "digest"); }, "cache tag must contain only");
}

void test_multidimensional_launch(Runtime& runtime) {
    constexpr dim3 grid_dim(2, 3, 2);
    constexpr dim3 block_dim(4, 2, 2);
    constexpr int num_blocks = grid_dim.x * grid_dim.y * grid_dim.z;
    constexpr int threads_per_block = block_dim.x * block_dim.y * block_dim.z;
    constexpr int num_values = num_blocks * threads_per_block;
    const auto kernel = runtime.compile("multidimensional_launch", get_source("multidimensional_launch.cu"));

    int* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), num_values * sizeof(int)));
    try {
        runtime.launch(
            kernel, {
                .grid_dim = grid_dim,
                .block_dim = block_dim,
            },
            output);
        DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());

        std::array<int, num_values> result{};
        DJ_MACA_RUNTIME_CHECK(mcMemcpy(result.data(), output, sizeof(result), mcMemcpyDeviceToHost));
        for (int block_index = 0; block_index < num_blocks; ++block_index) {
            for (int thread_index = 0; thread_index < threads_per_block; ++thread_index) {
                const auto index = block_index * threads_per_block + thread_index;
                DJ_HOST_ASSERT(result[index] == block_index * 1000 + thread_index,
                               "multidimensional launch produced an incorrect value at {}", index);
            }
        }

        DJ_MACA_RUNTIME_CHECK(mcFree(output));
    } catch (...) {
        mcFree(output);
        throw;
    }
}

void test_macro_selected_abi(Runtime& runtime) {
    const auto source = get_source("macro_selected_abi.cu");
    const CompilerOptions options_32 {.extra_mxcc_flags = {"-DTEST_INDEX_BITS=32"}};
    const CompilerOptions options_64 {.extra_mxcc_flags = {"-DTEST_INDEX_BITS=64"}};
    const auto kernel_32 = runtime.compile("macro_selected_abi", source, options_32);
    const auto kernel_64 = runtime.compile("macro_selected_abi", source, options_64);
    DJ_HOST_ASSERT(kernel_32 != kernel_64, "compiler macro did not select a distinct kernel ABI");

    long long* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(long long)));
    try {
        const int value_32 = 17;
        runtime.launch(
            kernel_32, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            output, value_32);
        DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
        long long result = 0;
        DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(result), mcMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == 4017, "32-bit macro-selected ABI produced {}", result);

        const long long value_64 = 23;
        runtime.launch(
            kernel_64, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1, 1, 1),
            },
            output, value_64);
        DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
        DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(result), mcMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == 8023, "64-bit macro-selected ABI produced {}", result);

        DJ_MACA_RUNTIME_CHECK(mcFree(output));
        output = nullptr;
    } catch (...) {
        if (output != nullptr)
            mcFree(output);
        throw;
    }
}

void test_kernel_lifecycle(const fs::path& cache_root) {
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto lifecycle_cache = cache_root / "kernel_lifecycle";
    set_env("KERNEL_LIFECYCLE_JIT_CACHE_DIR", lifecycle_cache.string());
    const auto runtime = make_runtime_with_prefix("KERNEL_LIFECYCLE", include_dir);

    const auto missing_dir = cache_root / "missing_devbin";
    deep_jit::make_dirs(missing_dir);
    expect_failure([&] { (void)deep_jit::MACA::load(missing_dir, runtime->env); },
                   "missing MACA device binary");

    // Bytes that are not a device binary at all: the refusal arrives from
    // whichever layer notices first -- the tool that reads the artifact exits
    // non-zero, or it finds nothing to name -- so either message is a correct
    // outcome and neither a driver-level nor an unrelated failure is.
    const auto invalid_dir = cache_root / "invalid_devbin";
    deep_jit::make_dirs(invalid_dir);
    deep_jit::write_file_sync(invalid_dir / "kernel.devbin", "not a device binary");
    expect_failure_any_of([&] { (void)deep_jit::MACA::load(invalid_dir, runtime->env); },
                          {"expected exactly one kernel", "command failed with exit code"});

    const auto source = get_template_source(34);
    const auto kernel = runtime->compile("kernel_lifecycle", source);
    DJ_HOST_ASSERT(launch_value(*runtime, kernel, 1) == 35);
    kernel->unload();
    kernel->unload();
    expect_failure(
        [&] {
            runtime->launch(
                kernel, {
                    .grid_dim = dim3(1, 1, 1),
                    .block_dim = dim3(1, 1, 1),
                });
        },
        "kernel must be loaded before launch");

    const auto reloaded_runtime = make_runtime_with_prefix("KERNEL_LIFECYCLE", include_dir);
    DJ_HOST_ASSERT(launch_value(*reloaded_runtime, reloaded_runtime->compile("kernel_lifecycle", source), 2) == 36,
                   "kernel could not be reloaded from disk cache after unload");
    unset_env("KERNEL_LIFECYCLE_JIT_CACHE_DIR");
}

void test_default_post_hook(const fs::path& cache_root) {
    const auto hook_cache = cache_root / "default_post_hook";
    set_env("DEFAULT_HOOK_JIT_CACHE_DIR", hook_cache.string());
    const auto runtime = make_runtime_with_prefix("DEFAULT_HOOK", get_test_maca_project_dir() / "kernels");
    runtime->default_compiler_options.post_hook = "scripts/post_hook.py";

    const auto source = get_template_source(71);
    const auto kernel = runtime->compile("default_post_hook", source);
    DJ_HOST_ASSERT(launch_value(*runtime, kernel, 1) == 72, "default post hook produced an unloadable binary");
    const auto artifact = runtime->compile_without_load("default_post_hook", source);
    DJ_HOST_ASSERT(deep_jit::read(artifact / "post_hook.marker") == "first\n",
                   "default post hook was not applied");
    DJ_HOST_ASSERT(deep_jit::read(artifact / "meta.json").find("\"post_hook\":\"scripts/post_hook.py\"") !=
                       std::string::npos,
                   "metadata does not record the default post hook");
    unset_env("DEFAULT_HOOK_JIT_CACHE_DIR");
}

void test_cache_key_option_matrix(Runtime& runtime) {
    const auto project_dir = get_test_maca_project_dir();
    const auto source = get_source("compiler_options.cu");
    const auto default_key = runtime.cache_key(source, runtime.default_compiler_options);
    const auto expected_base = deep_jit::hash::FNV1a()
        .update(runtime.config.extra_signature)
        .update(runtime.backend.compiler_info.get_hash())
        .get_hex_digest();
    DJ_HOST_ASSERT(runtime.hash_base.get_hex_digest() == expected_base,
                   "runtime base hash does not contain signature and compiler info in order");
    auto expected_hash = runtime.hash_base;
    expected_hash.update(deep_jit::str::join(runtime.default_compiler_options.get_flags()));
    expected_hash.update(runtime.default_compiler_options.get_post_hook_hash(runtime.config));
    expected_hash.update(runtime.parser.parse_into_hash(source));
    DJ_HOST_ASSERT(default_key == expected_hash.get_hex_digest(), "cache key components were combined in the wrong order");

    const std::string binary_source_a = source + std::string("\n// binary\0A", 12);
    const std::string binary_source_b = source + std::string("\n// binary\0B", 12);
    DJ_HOST_ASSERT(runtime.cache_key(binary_source_a, runtime.default_compiler_options) !=
                       runtime.cache_key(binary_source_b, runtime.default_compiler_options),
                   "kernel cache key ignored source bytes after a null");
    const auto expect_changed = [&](const CompilerOptions& overrides, const std::string_view name) {
        const auto options = runtime.default_compiler_options.override_with(overrides);
        DJ_HOST_ASSERT(runtime.cache_key(source, options) != default_key,
                       "compiler option did not affect cache key: {}", name);
    };

    expect_changed(CompilerOptions {.optimize_level = "0"}, "optimize_level");
    expect_changed(CompilerOptions {.fast_math = true}, "fast_math");
    expect_changed(CompilerOptions {.compiler_verbose = true}, "compiler_verbose");
    expect_changed(CompilerOptions {.check_no_spills = true}, "check_no_spills");
    expect_changed(CompilerOptions {.check_no_local_memory = true}, "check_no_local_memory");
    expect_changed(CompilerOptions {.with_line_info = true}, "with_line_info");
    // Relative to the device's own family: an arch equal to the default is not
    // a change, and which family is the default depends on the machine.
    const auto other_arch = runtime.device.get_arch() == "1000" ? "1600" : "1000";
    expect_changed(CompilerOptions {.arch = other_arch}, "arch");
    // The default `mxcc_flags` already carries the standard flag, so replacing
    // the list has to name a different one to be observable.
    expect_changed(CompilerOptions {.mxcc_flags = std::vector<std::string>{"-std=c++17"}}, "mxcc_flags");
    expect_changed(CompilerOptions {.extra_mxcc_flags = {"-DKEY_OPTION=1"}}, "extra_mxcc_flags");

    auto binary_flag_a = runtime.default_compiler_options;
    auto binary_flag_b = runtime.default_compiler_options;
    binary_flag_a.extra_mxcc_flags = {std::string("-DKEY=\0A", 8)};
    binary_flag_b.extra_mxcc_flags = {std::string("-DKEY=\0B", 8)};
    DJ_HOST_ASSERT(runtime.cache_key(source, binary_flag_a) != runtime.cache_key(source, binary_flag_b),
                   "compiler-option hash ignored bytes after a null");
    binary_flag_a.extra_mxcc_flags = {std::string("A\0-DKEY", 7)};
    binary_flag_b.extra_mxcc_flags = {std::string("B\0-DKEY", 7)};
    DJ_HOST_ASSERT(runtime.cache_key(source, binary_flag_a) != runtime.cache_key(source, binary_flag_b),
                   "compiler-option hash ignored bytes before a null");

    // The assembly dump is an artifact axis, not a compilation input: it is
    // outside the digest and only runs on a cache miss (see `test_dump_asm`).
    auto dump_options = runtime.default_compiler_options;
    dump_options.dump_asm = true;
    DJ_HOST_ASSERT(runtime.cache_key(source, dump_options) == default_key,
                   "assembly dump option must not affect the cache key");

    std::unordered_set<std::string> randomized_keys;
    randomized_keys.reserve(5000);
    std::mt19937_64 generator(0x741bc92du);
    for (int index = 0; index < 5000; ++index) {
        const auto randomized_source = source + std::format("\n// {} {}\n", index, generator());
        DJ_HOST_ASSERT(randomized_keys.emplace(
                           runtime.cache_key(randomized_source, runtime.default_compiler_options)).second,
                       "randomized cache key collision at input {}", index);
    }
}

void test_include_dirs(const fs::path& cache_root) {
    const auto project_dir = get_test_maca_project_dir();
    const auto source = get_source("tracked_include.cu");
    const auto include_original = project_dir / "include_original";
    const auto include_same_content = project_dir / "include_same_content";
    const auto include_changed_content = project_dir / "include_changed_content";
    const auto runtime_original = make_runtime(include_original);
    const auto runtime_same_content = make_runtime(include_same_content);
    const auto runtime_changed_content = make_runtime(include_changed_content);
    // The one-byte difference sits one level DOWN, in the header that the
    // tracked value header includes: the two root headers are byte-identical,
    // so the two artifacts can only differ if the parser descends recursively
    // (the compiler-side witness of the same descent is `launch_value` below,
    // 12 against 13).  A difference in the root header would make this case
    // pass without the recursion ever being exercised, so the root equality is
    // asserted too.
    DJ_HOST_ASSERT(deep_jit::read(include_original / "test_maca/tracked_include_value.hpp") ==
                       deep_jit::read(include_changed_content / "test_maca/tracked_include_value.hpp"),
                   "the tracked include fixture must differ below the root header, not in it");
    DJ_HOST_ASSERT(count_different_bytes(
                       deep_jit::read(include_original / "test_maca/detail/tracked_offset.hpp"),
                       deep_jit::read(include_changed_content / "test_maca/detail/tracked_offset.hpp")) == 1,
                   "tracked include fixture must differ by exactly one byte");

    const auto artifact_original = runtime_original->compile_without_load("tracked_include", source);
    check_artifact(artifact_original, source);
    DJ_HOST_ASSERT(launch_value(*runtime_original, runtime_original->compile("tracked_include", source), 1) == 12,
                   "tracked include file was not compiled");
    const auto artifact_same_content = runtime_same_content->compile_without_load("tracked_include", source);
    DJ_HOST_ASSERT(artifact_original == artifact_same_content, "include directory paths must not affect the cache key");
    DJ_HOST_ASSERT(launch_value(*runtime_same_content, runtime_same_content->compile("tracked_include", source), 1) == 12,
                   "equivalent include directory did not reuse a valid kernel");

    const auto artifact_changed_content = runtime_changed_content->compile_without_load("tracked_include", source);
    check_artifact(artifact_changed_content, source);
    DJ_HOST_ASSERT(artifact_original != artifact_changed_content, "changed include contents must change the cache key");
    DJ_HOST_ASSERT(launch_value(*runtime_changed_content, runtime_changed_content->compile("tracked_include", source), 1) == 13,
                   "changed include file was not compiled");

    const auto runtime_original_first = std::make_shared<Runtime>(deep_jit::Config(
        project_dir, "TEST", "maca-test-signature",
        std::vector<fs::path>{include_original, include_changed_content},
        std::vector<std::string>{"test_maca/"}));
    const auto runtime_changed_first = std::make_shared<Runtime>(deep_jit::Config(
        project_dir, "TEST", "maca-test-signature",
        std::vector<fs::path>{include_changed_content, include_original},
        std::vector<std::string>{"test_maca/"}));
    DJ_HOST_ASSERT(launch_value(*runtime_original_first, runtime_original_first->compile("tracked_include", source), 1) == 12,
                   "compiler did not use the first matching include directory");
    DJ_HOST_ASSERT(launch_value(*runtime_changed_first, runtime_changed_first->compile("tracked_include", source), 1) == 13,
                   "compiler and parser include-directory order diverged");

    const auto signature_runtime = make_runtime(include_original, "different-signature");
    DJ_HOST_ASSERT(signature_runtime->cache_key(source, signature_runtime->default_compiler_options) !=
                       runtime_original->cache_key(source, runtime_original->default_compiler_options),
                   "extra signature must change the cache key");
    const auto signature_suffix_a = make_runtime(include_original, std::string("signature\0A", 11));
    const auto signature_suffix_b = make_runtime(include_original, std::string("signature\0B", 11));
    DJ_HOST_ASSERT(signature_suffix_a->cache_key(source, signature_suffix_a->default_compiler_options) !=
                       signature_suffix_b->cache_key(source, signature_suffix_b->default_compiler_options),
                   "extra-signature hash ignored bytes after a null");
    const auto signature_prefix_a = make_runtime(include_original, std::string("A\0signature", 11));
    const auto signature_prefix_b = make_runtime(include_original, std::string("B\0signature", 11));
    DJ_HOST_ASSERT(signature_prefix_a->cache_key(source, signature_prefix_a->default_compiler_options) !=
                       signature_prefix_b->cache_key(source, signature_prefix_b->default_compiler_options),
                   "extra-signature hash ignored bytes before a null");
    const auto untracked_runtime = std::make_shared<Runtime>(deep_jit::Config(
        project_dir, "TEST", "maca-test-signature", {include_original}, {"another_library/"}));
    DJ_HOST_ASSERT(untracked_runtime->cache_key(source, untracked_runtime->default_compiler_options) !=
                       runtime_original->cache_key(source, runtime_original->default_compiler_options),
                   "include prefixes must determine whether dependency contents enter the cache key");
    // Same root as the rest of the suite: `<cache_root>/cache`.
    check_tmp_is_empty(cache_root / "cache");
}

void test_untracked_dependency_include_flags(Runtime& runtime, const fs::path& cache_root) {
    const auto project_dir = get_test_maca_project_dir();
    const auto include_dir = project_dir / "include_original";
    const auto dependency_original = project_dir / "third_party_original";
    const auto dependency_same_content = project_dir / "third_party_same_content";
    const auto dependency_changed_content = project_dir / "third_party_changed_content";
    const auto dependency_cache = cache_root / "untracked_dependency";
    set_env("UNTRACKED_DEPENDENCY_JIT_CACHE_DIR", dependency_cache.string());
    const auto dependency_runtime = make_runtime_with_prefix("UNTRACKED_DEPENDENCY", include_dir);
    const auto source = get_source("untracked_dependency.cu");
    DJ_HOST_ASSERT(count_different_bytes(
                       deep_jit::read(dependency_original / "third_party/dependency_value.cuh"),
                       deep_jit::read(dependency_changed_content / "third_party/dependency_value.cuh")) == 1,
                   "untracked dependency fixture must differ by exactly one byte");

    const auto make_options = [&](const fs::path& dependency_dir) {
        auto options = dependency_runtime->default_compiler_options;
        options.extra_mxcc_flags.emplace_back("-I" + dependency_dir.string());
        return options;
    };
    const auto original_options = make_options(dependency_original);
    const auto same_content_options = make_options(dependency_same_content);
    const auto changed_content_options = make_options(dependency_changed_content);
    DJ_HOST_ASSERT(dependency_runtime->cache_key(source, original_options) !=
                       dependency_runtime->cache_key(source, same_content_options),
                   "untracked dependency include paths must remain in the cache key");
    DJ_HOST_ASSERT(dependency_runtime->cache_key(source, original_options) !=
                       dependency_runtime->cache_key(source, changed_content_options));

    DJ_HOST_ASSERT(launch_value(*dependency_runtime, dependency_runtime->compile("untracked_dependency", source, original_options)) == 17);
    DJ_HOST_ASSERT(launch_value(*dependency_runtime, dependency_runtime->compile("untracked_dependency", source, changed_content_options)) == 18);

    const auto transitive_source = get_source("tracked_to_untracked_dependency.cu");
    deep_jit::Parser transitive_parser_original({include_dir, dependency_original}, {"test_maca/"});
    deep_jit::Parser transitive_parser_changed({include_dir, dependency_changed_content}, {"test_maca/"});
    DJ_HOST_ASSERT(transitive_parser_original.parse_into_hash(transitive_source) ==
                       transitive_parser_changed.parse_into_hash(transitive_source),
                   "parser must ignore an untracked dependency included by a tracked header");
    DJ_HOST_ASSERT(dependency_runtime->cache_key(transitive_source, original_options) !=
                       dependency_runtime->cache_key(transitive_source, changed_content_options),
                   "untracked transitive include paths must remain in the compiler-option hash");
    DJ_HOST_ASSERT(launch_value(
        *dependency_runtime,
        dependency_runtime->compile("tracked_to_untracked_dependency", transitive_source, original_options)) == 17);
    DJ_HOST_ASSERT(launch_value(
        *dependency_runtime,
        dependency_runtime->compile("tracked_to_untracked_dependency", transitive_source, changed_content_options)) == 18);
    unset_env("UNTRACKED_DEPENDENCY_JIT_CACHE_DIR");
}

void test_relocated_wheel_include_dir(const fs::path& cache_root) {
    const auto root = get_test_maca_project_dir();
    const auto wheel_cache = cache_root / "wheel_include_relocation";
    const auto environment_a_include = cache_root / "environment_a/site-packages/deep_jit/include";
    const auto environment_b_include = cache_root / "environment_b/site-packages/deep_jit/include";
    const auto header = fs::path("deep_jit/wheel_marker.cuh");
    deep_jit::make_dirs((environment_a_include / header).parent_path());
    deep_jit::make_dirs((environment_b_include / header).parent_path());
    deep_jit::write_file_sync(environment_a_include / header, "#pragma once\n");
    deep_jit::write_file_sync(environment_b_include / header, "#pragma once\n");

    const auto source = "#include <deep_jit/wheel_marker.cuh>\n" + get_template_source(73);
    const auto make_environment_runtime = [&](const fs::path& include_dir) {
        return std::make_shared<Runtime>(deep_jit::Config(
            root, "WHEEL_INCLUDE", "maca-test-signature",
            std::vector<fs::path>{include_dir},
            std::vector<std::string>{"deep_jit/"}));
    };

    set_env("WHEEL_INCLUDE_JIT_CACHE_DIR", wheel_cache.string());
    const auto environment_a = make_environment_runtime(environment_a_include);
    const auto first_artifact = environment_a->compile_without_load("wheel_include_relocation", source);
    check_artifact(first_artifact, source);
    const auto first_metadata = deep_jit::read(first_artifact / "meta.json");
    DJ_HOST_ASSERT(first_metadata.find(environment_a_include.string()) != std::string::npos);

    const auto environment_b = make_environment_runtime(environment_b_include);
    DJ_HOST_ASSERT(environment_a->cache_key(source, environment_a->default_compiler_options) ==
                       environment_b->cache_key(source, environment_b->default_compiler_options),
                   "relocating an identical wheel include directory changed the MACA cache key");
    // Point the second runtime at a compiler that must never be reached: a
    // rebuild here would fail rather than silently recompile.
    environment_b->backend.toolkit.mxcc = cache_root / "compiler_must_not_run";
    const auto second_artifact = environment_b->compile_without_load("wheel_include_relocation", source);
    DJ_HOST_ASSERT(second_artifact == first_artifact,
                   "relocating an identical wheel include directory triggered MACA compilation");
    DJ_HOST_ASSERT(deep_jit::read(second_artifact / "meta.json") == first_metadata,
                   "a MACA cache hit rewrote metadata after the wheel include directory moved");
    check_tmp_is_empty(wheel_cache);
    unset_env("WHEEL_INCLUDE_JIT_CACHE_DIR");
}

void test_lazy_runtime(const fs::path& cache_root) {
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto first_cache = cache_root / "lazy_runtime_first";
    const auto second_cache = cache_root / "lazy_runtime_second";
    set_env("LAZY_RUNTIME_JIT_CACHE_DIR", first_cache.string());
    auto lazy = deep_jit::create_lazy_jit<deep_jit::MACA>(deep_jit::Config(
        get_test_maca_project_dir(),
        "LAZY_RUNTIME",
        "lazy-runtime",
        {include_dir},
        {"test_maca/"}));
    set_env("LAZY_RUNTIME_JIT_CACHE_DIR", second_cache.string());

    const auto first = lazy.get();
    const auto second = lazy.get();
    DJ_HOST_ASSERT(first == second, "create_lazy_jit initialized more than once");
    DJ_HOST_ASSERT(first->disk_cache.paths.front() == second_cache,
                   "create_lazy_jit constructed the runtime before first use");
    DJ_HOST_ASSERT(not fs::exists(first_cache), "unused pre-initialization cache path was created");
    unset_env("LAZY_RUNTIME_JIT_CACHE_DIR");
}

void test_consumer_default_compiler_flags(const fs::path& cache_root) {
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto consumer_cache = cache_root / "consumer_defaults";
    set_env("CONSUMER_DEFAULTS_JIT_CACHE_DIR", consumer_cache.string());
    const auto runtime = make_runtime_with_prefix("CONSUMER_DEFAULTS", include_dir);
    const auto source = get_source("compiler_options.cu");
    const auto original_key = runtime->cache_key(source, runtime->default_compiler_options);

    auto& mxcc_flags = *runtime->default_compiler_options.mxcc_flags;
    mxcc_flags.emplace_back("-I" + include_dir.string());
    mxcc_flags.emplace_back("-DTEST_OPTION=64");
    DJ_HOST_ASSERT(runtime->cache_key(source, runtime->default_compiler_options) != original_key,
                   "consumer default compiler flags did not affect the cache key");

    const auto artifact = runtime->compile_without_load("consumer_defaults", source);
    const auto metadata = deep_jit::read(artifact / "meta.json");
    const std::vector<std::string> expected_flags = {
        "-I" + include_dir.string(),
        "-DTEST_OPTION=64",
    };
    for (const auto& expected : expected_flags) {
        DJ_HOST_ASSERT(metadata.find(expected) != std::string::npos,
                       "consumer compiler flag is missing from metadata: {}", expected);
    }
    DJ_HOST_ASSERT(launch_single_argument(*runtime, runtime->compile("consumer_defaults", source)) == 64,
                   "consumer default macro was not compiled");
    unset_env("CONSUMER_DEFAULTS_JIT_CACHE_DIR");
}

void test_debug_environment(const fs::path& cache_root) {
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto debug_cache = cache_root / "debug_environment";
    set_env("DEBUG_ENV_JIT_CACHE_DIR", debug_cache.string());
    set_env("DEBUG_ENV_JIT_DEBUG", "1");
    const auto runtime = make_runtime_with_prefix("DEBUG_ENV", include_dir);
    DJ_HOST_ASSERT(runtime->default_compiler_options.compiler_verbose == true);
    DJ_HOST_ASSERT(runtime->default_compiler_options.with_line_info == true);
    DJ_HOST_ASSERT(runtime->default_compiler_options.dump_asm == true);

    const auto source = get_template_source(47);
    const auto artifact = runtime->compile_without_load("debug_environment", source);
    DJ_HOST_ASSERT(fs::file_size(artifact / "kernel.s") > 0);
    DJ_HOST_ASSERT(launch_value(*runtime, runtime->compile("debug_environment", source), 1) == 48);
    unset_env("DEBUG_ENV_JIT_CACHE_DIR");
    unset_env("DEBUG_ENV_JIT_DEBUG");
}

void test_library_environment_compatibility(Runtime& runtime, const fs::path& cache_root) {
    set_env("DJ_JIT_DEBUG", "1");
    set_env("DJ_JIT_CPP_STANDARD", "17");
    set_env("DJ_JIT_CHECK_NO_LOCAL_MEMORY", "1");
    set_env("DG_JIT_DEBUG", "0");
    set_env("DG_JIT_CPP_STANDARD", "20");
    set_env("DG_JIT_CHECK_NO_SPILLS", "1");
    set_env("DG_JIT_CHECK_NO_LOCAL_MEMORY", "0");
    set_env("EP_JIT_CPP_STANDARD", "23");
    set_env("EP_JIT_CHECK_NO_LOCAL_MEMORY", "1");
    set_env("JIT_CPP_STANDARD", "14");
    set_env("JIT_CHECK_NO_SPILLS", "1");

    const auto dg_options = CompilerOptions::default_options(deep_jit::Env("DG"), runtime.device);
    const auto ep_options = CompilerOptions::default_options(deep_jit::Env("EP"), runtime.device);
    const auto other_options = CompilerOptions::default_options(deep_jit::Env("OTHER"), runtime.device);
    DJ_HOST_ASSERT(dg_options.compiler_verbose == false and dg_options.with_line_info == false and
                   dg_options.dump_asm == false,
                   "DG_JIT_DEBUG=0 must override DJ_JIT_DEBUG=1");
    DJ_HOST_ASSERT(dg_options.check_no_spills == true);
    DJ_HOST_ASSERT(dg_options.check_no_local_memory == false,
                   "DG_JIT_CHECK_NO_LOCAL_MEMORY=0 must override the DJ default");
    DJ_HOST_ASSERT(std::ranges::find(*dg_options.mxcc_flags, "-std=c++20") != dg_options.mxcc_flags->end());
    DJ_HOST_ASSERT(ep_options.compiler_verbose == true and ep_options.with_line_info == true and
                   ep_options.dump_asm == true,
                   "EP must inherit DJ_JIT_DEBUG");
    DJ_HOST_ASSERT(ep_options.check_no_local_memory == true);
    DJ_HOST_ASSERT(std::ranges::find(*ep_options.mxcc_flags, "-std=c++23") != ep_options.mxcc_flags->end());
    DJ_HOST_ASSERT(other_options.check_no_spills == false,
                   "unprefixed JIT_CHECK_NO_SPILLS must not be read");
    DJ_HOST_ASSERT(other_options.check_no_local_memory == true,
                   "DJ_JIT_CHECK_NO_LOCAL_MEMORY must be the global fallback");
    DJ_HOST_ASSERT(std::ranges::find(*other_options.mxcc_flags, "-std=c++17") != other_options.mxcc_flags->end(),
                   "DJ_JIT_CPP_STANDARD must be the global fallback");

    const auto dg_cache = cache_root / "dg_environment_cache";
    const auto ep_cache = cache_root / "ep_environment_cache";
    set_env("DG_JIT_CACHE_DIR", dg_cache.string());
    set_env("EP_JIT_CACHE_DIR", ep_cache.string());
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(deep_jit::Env("DG")).paths == std::vector<fs::path>{dg_cache});
    DJ_HOST_ASSERT(deep_jit::DiskCache::from_env(deep_jit::Env("EP")).paths == std::vector<fs::path>{ep_cache});

    // A library-prefixed compiler override is selected ahead of the discovered
    // toolkit.  `llvm-nm` is resolved next to the selected mxcc, so the stand-in
    // toolchain has to carry one too.
    const auto compiler_override_dir = cache_root / "compiler_override";
    const auto compiler_override_cache = compiler_override_dir / "cache_root";
    const auto compiler_override = compiler_override_dir / "valid_mxcc";
    deep_jit::make_dirs(compiler_override_dir);
    write_executable(compiler_override, "#!/bin/sh\nprintf '%s\\n' 'mxcc version 1.0.0 (d9102a1572)'\n");
    std::filesystem::create_symlink(runtime.backend.toolkit.llvm_nm, compiler_override_dir / "llvm-nm");
    set_env("COMPILER_OVERRIDE_JIT_CACHE_DIR", compiler_override_cache.string());
    set_env("COMPILER_OVERRIDE_JIT_MXCC_COMPILER", compiler_override.string());
    const auto override_runtime = make_runtime_with_prefix(
        "COMPILER_OVERRIDE", get_test_maca_project_dir() / "kernels");
    DJ_HOST_ASSERT(override_runtime->backend.toolkit.mxcc == fs::absolute(compiler_override).lexically_normal(),
                   "library-prefixed mxcc override was not selected");

    set_env("INVALID_COMPILER_JIT_MXCC_COMPILER", (compiler_override_dir / "missing_mxcc").string());
    expect_failure(
        [&] { make_runtime_with_prefix("INVALID_COMPILER", get_test_maca_project_dir() / "kernels"); },
        "mxcc compiler is not executable");

    const auto malformed_compiler = compiler_override_dir / "malformed_mxcc";
    write_executable(malformed_compiler, "#!/bin/sh\nprintf '%s\\n' 'not an mxcc version'\n");
    set_env("MALFORMED_COMPILER_JIT_MXCC_COMPILER", malformed_compiler.string());
    expect_failure(
        [&] { make_runtime_with_prefix("MALFORMED_COMPILER", get_test_maca_project_dir() / "kernels"); },
        "failed to parse mxcc version");

    unset_env("DJ_JIT_DEBUG");
    unset_env("DJ_JIT_CPP_STANDARD");
    unset_env("DJ_JIT_CHECK_NO_LOCAL_MEMORY");
    unset_env("DG_JIT_DEBUG");
    unset_env("DG_JIT_CPP_STANDARD");
    unset_env("DG_JIT_CHECK_NO_SPILLS");
    unset_env("DG_JIT_CHECK_NO_LOCAL_MEMORY");
    unset_env("EP_JIT_CPP_STANDARD");
    unset_env("EP_JIT_CHECK_NO_LOCAL_MEMORY");
    unset_env("DG_JIT_CACHE_DIR");
    unset_env("EP_JIT_CACHE_DIR");
    unset_env("COMPILER_OVERRIDE_JIT_CACHE_DIR");
    unset_env("COMPILER_OVERRIDE_JIT_MXCC_COMPILER");
    unset_env("INVALID_COMPILER_JIT_MXCC_COMPILER");
    unset_env("MALFORMED_COMPILER_JIT_MXCC_COMPILER");
    unset_env("JIT_CPP_STANDARD");
    unset_env("JIT_CHECK_NO_SPILLS");

    // The remaining knobs the backend reads -- the resource report, line
    // information, the assembly dump and the CUDA `JIT_DUMP_PTX` alias that
    // feeds it -- must reach the defaults through the same chain, and must
    // also reach the flags they are supposed to generate.
    set_env("DJ_JIT_PTXAS_VERBOSE", "1");
    set_env("DJ_JIT_WITH_LINEINFO", "1");
    set_env("DJ_JIT_DUMP_PTX", "1");
    const auto inherited = CompilerOptions::default_options(deep_jit::Env("INHERITED"), runtime.device);
    DJ_HOST_ASSERT(inherited.compiler_verbose == true and inherited.with_line_info == true and
                   inherited.dump_asm == true,
                   "DJ_JIT_PTXAS_VERBOSE / DJ_JIT_WITH_LINEINFO / DJ_JIT_DUMP_PTX must be global fallbacks");
    const auto inherited_flags = inherited.get_flags();
    DJ_HOST_ASSERT(std::ranges::find(inherited_flags, "-resource-usage") != inherited_flags.end(),
                   "JIT_PTXAS_VERBOSE must request the mxcc resource report");
    DJ_HOST_ASSERT(std::ranges::find(inherited_flags, "--generate-line-info") != inherited_flags.end(),
                   "JIT_WITH_LINEINFO must request line information");
    unset_env("DJ_JIT_PTXAS_VERBOSE");
    unset_env("DJ_JIT_WITH_LINEINFO");
    unset_env("DJ_JIT_DUMP_PTX");

    set_env("JIT_DUMP_ASM", "1");
    set_env("JIT_PTXAS_VERBOSE", "1");
    const auto unprefixed_knobs = CompilerOptions::default_options(deep_jit::Env("UNPREFIXED_KNOBS"), runtime.device);
    DJ_HOST_ASSERT(unprefixed_knobs.dump_asm == false and unprefixed_knobs.compiler_verbose == false,
                   "unprefixed JIT_DUMP_ASM / JIT_PTXAS_VERBOSE must not be read");
    unset_env("JIT_DUMP_ASM");
    unset_env("JIT_PTXAS_VERBOSE");

    set_env("JIT_DEBUG", "1");
    const auto unprefixed_options = CompilerOptions::default_options(deep_jit::Env("UNPREFIXED"), runtime.device);
    DJ_HOST_ASSERT(unprefixed_options.compiler_verbose == false and unprefixed_options.with_line_info == false and
                   unprefixed_options.dump_asm == false,
                   "unprefixed JIT_DEBUG must not affect compiler defaults");
    unset_env("JIT_DEBUG");
}

void test_template_hash_and_launch(Runtime& runtime) {
    const auto source_1 = get_template_source(1);
    const auto source_2 = get_template_source(2);
    DJ_HOST_ASSERT(count_different_bytes(source_1, source_2) == 1,
                   "template source fixture must differ by exactly one byte");
    const auto initial_memory_cache_size = runtime.mem_cache.cache.size();
    const auto artifact_1 = runtime.compile_without_load("template_add", source_1);
    check_artifact(artifact_1, source_1);
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == initial_memory_cache_size,
                   "compile_without_load populated the memory cache");
    DJ_HOST_ASSERT(runtime.compile_without_load("template_add", source_1) == artifact_1,
                   "identical template source must reuse the cache entry");

    const auto kernel_1 = runtime.compile("template_add", source_1);
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == initial_memory_cache_size + 1,
                   "compile did not populate the memory cache");
    expect_failure(
        [&] { runtime.launch(std::shared_ptr<deep_jit::maca::Kernel>{}, LaunchOptions {}); },
        "kernel must not be null");
    DJ_HOST_ASSERT(launch_value(runtime, kernel_1, 1) == 2,
                   "template source was not compiled and launched");

    // Runtime arguments must not enter the cache key: a second launch of the
    // same kernel with a different input reuses the compiled artifact.
    const auto cache_size = runtime.mem_cache.cache.size();
    DJ_HOST_ASSERT(launch_value(runtime, kernel_1, 7) == 8, "unexpected second launch result");
    DJ_HOST_ASSERT(launch_value(runtime, kernel_1, 41) == 42, "runtime argument changed compiled behavior");
    DJ_HOST_ASSERT(runtime.mem_cache.cache.size() == cache_size, "runtime arguments must not change the cache key");
    DJ_HOST_ASSERT(runtime.compile("template_add", source_1) == kernel_1, "memory cache did not reuse the kernel");

    constexpr int cache_hit_iterations = 10000;
    const auto cache_hit_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < cache_hit_iterations; ++i)
        DJ_HOST_ASSERT(runtime.compile("template_add", source_1) == kernel_1);
    const auto cache_hit_elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - cache_hit_begin).count();
    std::printf("JIT memory-cache hit overhead: %.3f us\n", cache_hit_elapsed / cache_hit_iterations);

    // ... but a changed compile-time constant is a different kernel entirely.
    const auto artifact_2 = runtime.compile_without_load("template_add", source_2);
    check_artifact(artifact_2, source_2);
    DJ_HOST_ASSERT(artifact_1 != artifact_2, "changing a template argument must change the cache key");
    DJ_HOST_ASSERT(launch_value(runtime, runtime.compile("template_add", source_2), 7) == 9,
                   "changed template argument was not compiled");
}

void test_cooperative_launch(Runtime& runtime) {
    constexpr unsigned int num_blocks = 2;
    // A cooperative launch requires every block to be resident at once, so a
    // grid of 1<<20 blocks can never co-reside on any current device.  That
    // makes the pair below device-independent, and it is the whole point of
    // the case: the *same* grid is accepted plainly and refused
    // cooperatively.  Without it a backend that silently dropped the
    // attribute would still pass every other assertion here.
    constexpr unsigned int oversized_blocks = 1u << 20;
    const auto kernel = runtime.compile("cooperative_probe", get_source("cooperative_probe.cu"));

    int* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(int)));
    const auto launch_and_read = [&](const bool cooperative, const unsigned int blocks) {
        runtime.launch(
            kernel, {
                .grid_dim = dim3(blocks, 1, 1),
                .block_dim = dim3(32, 1, 1),
                .cluster_dim = dim3(1, 1, 1),
                .cooperative = cooperative,
                .enable_pdl = false,
                .nonportable_cluster_size_allowed = false,
            },
            output);
        DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());
        int result = 0;
        DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(int), mcMemcpyDeviceToHost));
        return result;
    };

    try {
        // The plain path first: this is what must keep working afterwards.
        DJ_HOST_ASSERT(launch_and_read(false, num_blocks) == static_cast<int>(num_blocks),
                       "non-cooperative launch produced the wrong grid");

        // The cooperative attribute is the one launch attribute the MACA
        // runtime honours; everything else is refused up front (see
        // `test_launch_option_validation`).  A cooperative grid that fits is
        // accepted and runs with the grid shape it asked for.
        DJ_HOST_ASSERT(launch_and_read(true, num_blocks) == static_cast<int>(num_blocks),
                       "cooperative launch produced the wrong grid");

        // The discriminating pair: the oversized grid is accepted on the
        // plain path ...
        DJ_HOST_ASSERT(launch_and_read(false, oversized_blocks) == static_cast<int>(oversized_blocks),
                       "the oversized grid did not run on the plain path");

        // ... and refused on the cooperative path, by the driver, with its
        // cooperative-specific error (`mcErrorCooperativeLaunchTooLarge`).
        // The message is the driver's, so only the keyword is asserted.
        expect_failure(
            [&] { launch_and_read(true, oversized_blocks); },
            "cooperative");

        DJ_MACA_RUNTIME_CHECK(mcFree(output));
        output = nullptr;
    } catch (...) {
        if (output != nullptr)
            mcFree(output);
        throw;
    }
}

void test_large_kernel_arguments(Runtime& runtime) {
    struct OpaqueArgument {
        unsigned char bytes[128];
    };
    struct LargeArgumentPayload {
        unsigned long long values[128];
    };
    struct IndirectArgumentPayload {
        int values[8];
    };
    static_assert(sizeof(LargeArgumentPayload) == 1024);
    static_assert(sizeof(OpaqueArgument) == 128);

    const auto make_payload = [] {
        LargeArgumentPayload payload{};
        for (int i = 0; i < 128; ++i)
            payload.values[i] = static_cast<unsigned long long>(i * 17 + 3);
        return payload;
    };
    std::array<OpaqueArgument, 18> opaques{};
    for (std::size_t argument = 0; argument < opaques.size(); ++argument) {
        for (std::size_t byte = 0; byte < sizeof(OpaqueArgument); ++byte)
            opaques[argument].bytes[byte] = static_cast<unsigned char>(argument * 19 + byte * 7);
    }
    IndirectArgumentPayload indirect_payload {{2, 3, 5, 7, 11, 13, 17, 19}};

    unsigned long long expected = 1469598103934665603ull;
    const auto payload_for_checksum = make_payload();
    for (const auto value : payload_for_checksum.values)
        expected = (expected ^ value) * 1099511628211ull;
    for (const auto& opaque : opaques) {
        for (std::size_t i = 0; i < sizeof(OpaqueArgument); ++i)
            expected = (expected ^ opaque.bytes[i]) * 1099511628211ull;
    }
    for (const auto value : indirect_payload.values)
        expected = (expected ^ static_cast<unsigned int>(value)) * 1099511628211ull;

    const auto kernel = runtime.compile("large_arguments", get_source("large_arguments.cu"));
    unsigned long long* output = nullptr;
    DJ_MACA_RUNTIME_CHECK(mcMalloc(reinterpret_cast<void**>(&output), sizeof(unsigned long long)));
    try {
        const deep_jit::NoRefPtr indirect_storage {&indirect_payload};
        void* optional_pointer = nullptr;
        runtime.launch(
            kernel, {
                .grid_dim = dim3(1, 1, 1),
                .block_dim = dim3(1024, 1, 1),
            },
            output,
            make_payload(),
            opaques[0], opaques[1], opaques[2], opaques[3], opaques[4], opaques[5],
            opaques[6], opaques[7], opaques[8], opaques[9], opaques[10], opaques[11],
            opaques[12], opaques[13], opaques[14], opaques[15], opaques[16], opaques[17],
            indirect_storage,
            optional_pointer);
        DJ_MACA_RUNTIME_CHECK(mcDeviceSynchronize());

        unsigned long long result = 0;
        DJ_MACA_RUNTIME_CHECK(mcMemcpy(&result, output, sizeof(result), mcMemcpyDeviceToHost));
        DJ_HOST_ASSERT(result == expected, "large kernel arguments produced an incorrect checksum");
        DJ_MACA_RUNTIME_CHECK(mcFree(output));
        output = nullptr;
    } catch (...) {
        if (output != nullptr)
            mcFree(output);
        throw;
    }
}

// A stand-in mxcc that reports a version but never runs the real compiler's
// job: each scenario below fails at a different point in the compile, and each
// one must leave the cache untouched.
void write_fake_mxcc(const fs::path& path, const std::string& body) {
    write_executable(path, "#!/bin/sh\n" + body);
}

void make_fake_toolkit_dir(const fs::path& tool_dir, const fs::path& real_llvm_nm) {
    deep_jit::make_dirs(tool_dir);
    // `llvm-nm` is resolved next to the selected mxcc, and the load path needs
    // it even when the compiler itself is a stand-in.
    const auto llvm_nm = tool_dir / "llvm-nm";
    if (not fs::exists(llvm_nm))
        std::filesystem::create_symlink(real_llvm_nm, llvm_nm);
}

void test_backend_output_validation(Runtime& runtime, const fs::path& cache_root) {
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto source = get_template_source(94);
    const auto tool_dir = cache_root / "fake_compiler_tools";
    make_fake_toolkit_dir(tool_dir, runtime.backend.toolkit.llvm_nm);

    const auto real_mxcc = runtime.backend.toolkit.mxcc;
    const auto recovery_body = std::format("exec {} \"$@\"\n", real_mxcc.string());
    const std::string version_body = "if [ \"$1\" = \"--version\" ]; then\n"
                                     "  printf '%s\\n' 'mxcc version 1.0.0 (d9102a1572)'\n"
                                     "  exit 0\n"
                                     "fi\n";
    // The device-binary pass names its output after `-o`; the assembly pass is
    // the one that carries `-aop`.
    const std::string output_argument = "previous=''\n"
                                        "for argument in \"$@\"; do\n"
                                        "  if [ \"$previous\" = \"-o\" ]; then\n";

    // 1. The compiler fails after leaving a partial device binary behind.
    const auto failing_mxcc = tool_dir / "mxcc_failing_after_binary";
    write_fake_mxcc(failing_mxcc, version_body + output_argument +
        "    printf '%s' 'partial device binary' > \"$argument\"\n"
        "    break\n"
        "  fi\n"
        "  previous=\"$argument\"\n"
        "done\n"
        "exit 37\n");
    const auto failing_cache = cache_root / "failing_after_binary";
    set_env("FAILING_AFTER_BINARY_JIT_CACHE_DIR", failing_cache.string());
    set_env("FAILING_AFTER_BINARY_JIT_MXCC_COMPILER", failing_mxcc.string());
    const auto failing_runtime = make_runtime_with_prefix("FAILING_AFTER_BINARY", include_dir);
    expect_failure(
        [&] { failing_runtime->compile_without_load("failed_after_binary", source); },
        "command failed with exit code 37");
    DJ_HOST_ASSERT(not fs::exists(failing_cache / "cache"),
                   "failed compiler output published a cache artifact");
    check_tmp_is_empty(failing_cache);
    write_fake_mxcc(failing_mxcc, recovery_body);
    DJ_HOST_ASSERT(launch_value(*failing_runtime, failing_runtime->compile("failed_after_binary", source), 1) == 95,
                   "runtime did not recover after the compiler failed with partial output");
    unset_env("FAILING_AFTER_BINARY_JIT_CACHE_DIR");
    unset_env("FAILING_AFTER_BINARY_JIT_MXCC_COMPILER");

    // 2. The compiler succeeds without producing a device binary at all.
    const auto no_binary_mxcc = tool_dir / "mxcc_without_binary";
    write_fake_mxcc(no_binary_mxcc, version_body + "exit 0\n");
    const auto no_binary_cache = cache_root / "no_binary";
    set_env("NO_BINARY_JIT_CACHE_DIR", no_binary_cache.string());
    set_env("NO_BINARY_JIT_MXCC_COMPILER", no_binary_mxcc.string());
    const auto no_binary_runtime = make_runtime_with_prefix("NO_BINARY", include_dir);
    expect_failure(
        [&] { no_binary_runtime->compile_without_load("missing_binary_output", source); },
        "mxcc did not produce a valid device binary");
    DJ_HOST_ASSERT(not fs::exists(no_binary_cache / "cache"),
                   "missing device binary published a cache artifact");
    check_tmp_is_empty(no_binary_cache);

    // 3. ... and the empty-file variant of the same failure.
    write_fake_mxcc(no_binary_mxcc, version_body + output_argument +
        "    : > \"$argument\"\n"
        "    exit 0\n"
        "  fi\n"
        "  previous=\"$argument\"\n"
        "done\n"
        "exit 0\n");
    expect_failure(
        [&] { no_binary_runtime->compile_without_load("missing_binary_output", source); },
        "mxcc did not produce a valid device binary");
    DJ_HOST_ASSERT(not fs::exists(no_binary_cache / "cache"),
                   "empty device binary published a cache artifact");
    check_tmp_is_empty(no_binary_cache);
    write_fake_mxcc(no_binary_mxcc, recovery_body);
    DJ_HOST_ASSERT(launch_value(*no_binary_runtime, no_binary_runtime->compile("missing_binary_output", source), 1) == 95,
                   "runtime did not recover after missing device-binary output");
    unset_env("NO_BINARY_JIT_CACHE_DIR");
    unset_env("NO_BINARY_JIT_MXCC_COMPILER");

    // 4. The assembly pass produces nothing under `-aop` while the device
    //    binary pass succeeds.
    const auto no_asm_mxcc = tool_dir / "mxcc_without_assembly";
    write_fake_mxcc(no_asm_mxcc, version_body +
        "is_assembly=0\n"
        "for argument in \"$@\"; do\n"
        "  if [ \"$argument\" = \"-aop\" ]; then\n"
        "    is_assembly=1\n"
        "  fi\n"
        "done\n"
        "if [ \"$is_assembly\" = \"1\" ]; then\n"
        "  exit 0\n"
        "fi\n" + recovery_body);
    const auto no_asm_cache = cache_root / "no_assembly";
    set_env("NO_ASM_JIT_CACHE_DIR", no_asm_cache.string());
    set_env("NO_ASM_JIT_MXCC_COMPILER", no_asm_mxcc.string());
    const auto no_asm_runtime = make_runtime_with_prefix("NO_ASM", include_dir);
    expect_failure(
        [&] {
            no_asm_runtime->compile_without_load("missing_assembly_output", source,
                                                 CompilerOptions {.dump_asm = true});
        },
        "mxcc did not produce a valid device assembly");
    DJ_HOST_ASSERT(not fs::exists(no_asm_cache / "cache"),
                   "missing assembly output published a cache artifact");
    check_tmp_is_empty(no_asm_cache);
    write_fake_mxcc(no_asm_mxcc, recovery_body);
    const auto recovered_asm_artifact = no_asm_runtime->compile_without_load(
        "missing_assembly_output", source, CompilerOptions {.dump_asm = true});
    DJ_HOST_ASSERT(fs::file_size(recovered_asm_artifact / "kernel.s") > 0,
                   "runtime did not recover after missing assembly output");
    unset_env("NO_ASM_JIT_CACHE_DIR");
    unset_env("NO_ASM_JIT_MXCC_COMPILER");

    // 5. The artifact directory is written in order, and a failure while
    //    writing metadata must not publish the entry.  The hook runs before
    //    the metadata is written, so it can occupy the metadata path.
    const auto metadata_hook_root = tool_dir / "metadata_hook_root";
    const auto metadata_hook_path = metadata_hook_root / "hooks/metadata_conflict.py";
    const auto metadata_cache = cache_root / "metadata_failure";
    deep_jit::make_dirs(metadata_hook_path.parent_path());
    deep_jit::write_file_sync(
        metadata_hook_path,
        "from pathlib import Path\n"
        "Path('meta.json').mkdir()\n");
    set_env("METADATA_FAILURE_JIT_CACHE_DIR", metadata_cache.string());
    const auto metadata_runtime = std::make_shared<Runtime>(deep_jit::Config(
        fs::absolute(metadata_hook_root),
        "METADATA_FAILURE",
        "maca-test-signature",
        std::vector<fs::path>{include_dir},
        std::vector<std::string>{"test_maca/"}));
    const CompilerOptions metadata_options {.post_hook = "hooks/metadata_conflict.py"};
    const auto effective_metadata_options =
        metadata_runtime->default_compiler_options.override_with(metadata_options);
    const auto metadata_artifact = metadata_cache / "cache" /
        std::format("metadata_failure.{}", metadata_runtime->cache_key(source, effective_metadata_options));
    expect_failure(
        [&] { metadata_runtime->compile_without_load("metadata_failure", source, metadata_options); },
        "failed to open for writing");
    DJ_HOST_ASSERT(not fs::exists(metadata_artifact), "metadata write failure published a cache artifact");
    check_tmp_is_empty(metadata_cache);
    deep_jit::write_file_sync(metadata_hook_path, "# successful retry\n");
    DJ_HOST_ASSERT(metadata_runtime->compile_without_load("metadata_failure", source, metadata_options) == metadata_artifact,
                   "metadata failure retry changed the cached post-hook key");
    DJ_HOST_ASSERT(launch_value(
        *metadata_runtime,
        metadata_runtime->compile("metadata_failure", source, metadata_options), 1) == 95,
        "runtime did not recover after a metadata write failure");
    unset_env("METADATA_FAILURE_JIT_CACHE_DIR");
}

// Two devices, two runtimes.  The point is not that both launch -- it is that
// a runtime keeps the device it was built on: its properties must not follow
// the process current device, while the disk cache still follows the
// architecture, so two devices of the same family share an entry and two of
// different families do not.  Returns false (a skip) on a single-device host.
bool test_multiple_devices(const fs::path& cache_root) {
    int num_devices = 0;
    DJ_MACA_RUNTIME_CHECK(mcGetDeviceCount(&num_devices));
    std::printf("           visible devices: %d\n", num_devices);
    if (num_devices < 2)
        return false;

    int original_device = 0;
    DJ_MACA_RUNTIME_CHECK(mcGetDevice(&original_device));
    const auto include_dir = get_test_maca_project_dir() / "kernels";
    const auto shared_cache = cache_root / "multiple_devices";
    set_env("MULTI_DEVICE_JIT_CACHE_DIR", shared_cache.string());

    try {
        DJ_MACA_RUNTIME_CHECK(mcSetDevice(0));
        int device_0_clock_rate = 0;
        DJ_MACA_RUNTIME_CHECK(mcDeviceGetAttribute(&device_0_clock_rate, mcDeviceAttributeClockRate, 0));
        const auto runtime_0 = make_runtime_with_prefix("MULTI_DEVICE", include_dir);
        const auto runtime_0_arch = *runtime_0->default_compiler_options.arch;
        const auto runtime_0_clock_rate = runtime_0->device.get_clock_rate();
        const auto source = get_template_source(21);
        const auto artifact_0 = runtime_0->compile_without_load("multiple_devices", source);
        const auto kernel_0 = runtime_0->compile("multiple_devices", source);
        DJ_HOST_ASSERT(launch_value(*runtime_0, kernel_0, 1) == 22, "device 0 launch produced the wrong value");

        DJ_MACA_RUNTIME_CHECK(mcSetDevice(1));
        const auto runtime_1 = make_runtime_with_prefix("MULTI_DEVICE", include_dir);
        const auto runtime_1_arch = *runtime_1->default_compiler_options.arch;
        const auto artifact_1 = runtime_1->compile_without_load("multiple_devices", source);
        if (runtime_0_arch == runtime_1_arch) {
            DJ_HOST_ASSERT(artifact_0 == artifact_1,
                           "matching architectures did not share the disk cache across devices");
        } else {
            DJ_HOST_ASSERT(artifact_0 != artifact_1,
                           "different architectures unexpectedly shared the disk cache");
        }

        // The process current device is now 1; runtime 0 must be unaffected.
        DJ_HOST_ASSERT(runtime_0->device.get_arch() == runtime_0_arch,
                       "runtime device properties changed with the process current device");
        DJ_HOST_ASSERT(runtime_0_clock_rate == static_cast<int64_t>(device_0_clock_rate) * 1000,
                       "device clock rate does not match the reported attribute");
        DJ_HOST_ASSERT(runtime_0->device.get_clock_rate() == runtime_0_clock_rate,
                       "runtime clock rate was not cached");

        const auto kernel_1 = runtime_1->compile("multiple_devices", source);
        DJ_HOST_ASSERT(launch_value(*runtime_1, kernel_1, 2) == 23, "device 1 launch produced the wrong value");
        DJ_MACA_RUNTIME_CHECK(mcSetDevice(0));
        DJ_HOST_ASSERT(launch_value(*runtime_0, kernel_0, 3) == 24, "device 0 relaunch produced the wrong value");
        DJ_MACA_RUNTIME_CHECK(mcSetDevice(1));
        DJ_HOST_ASSERT(launch_value(*runtime_1, kernel_1, 4) == 25, "device 1 relaunch produced the wrong value");
    } catch (...) {
        mcSetDevice(original_device);
        unset_env("MULTI_DEVICE_JIT_CACHE_DIR");
        throw;
    }

    DJ_MACA_RUNTIME_CHECK(mcSetDevice(original_device));
    unset_env("MULTI_DEVICE_JIT_CACHE_DIR");
    check_tmp_is_empty(shared_cache);
    return true;
}

// Compile/load/launch one kernel and report the artifact and the result; used
// by the Python driver to race several processes onto the same cache entry.
int run_compile_once(const std::string& tag, const int bias,
                     const fs::path& ready_path, const fs::path& start_path) {
    try {
        set_env("TEST_JIT_CACHE_DIR", get_cache_root().string());
        const auto runtime = make_runtime(get_test_maca_project_dir() / "kernels");

        deep_jit::write_file_sync(ready_path, "ready");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (not fs::exists(start_path)) {
            DJ_HOST_ASSERT(std::chrono::steady_clock::now() < deadline, "compile worker did not leave the barrier");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // The bias is baked into the source text, so a fixed input of 1 is
        // enough to observe which entry was compiled.
        const auto source = get_template_source(bias);
        const auto artifact = runtime->compile_without_load(tag, source);
        const auto value = launch_value(*runtime, runtime->compile(tag, source), 1);
        std::printf("ARTIFACT=%s\nRESULT=%d\n", artifact.string().c_str(), value);
        std::fflush(stdout);
        return value == bias + 1 ? 0 : 1;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "compile worker failed: %s\n", exception.what());
        return 1;
    }
}

// One compile/load/launch with diagnostics enabled from the environment; the
// Python driver asserts on the printed compiler and load reports.  The cache
// root is left as it is, so running this twice over one root exercises the
// cache-hit path.
int run_diagnostics() {
    const auto cache_root = get_cache_root();
    deep_jit::make_dirs(cache_root);
    set_env("TEST_JIT_CACHE_DIR", cache_root.string());
    const auto runtime = make_runtime(get_test_maca_project_dir() / "kernels");
    const auto kernel = runtime->compile("scalar_increment", get_source("scalar_increment.cu"));
    std::printf("RESULT=%d\n", launch_value(*runtime, kernel, 41));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 and std::string_view(argv[1]) == "--compile-once")
        return run_compile_once(argv[2], std::stoi(argv[3]), argv[4], argv[5]);
    if (argc == 2 and std::string_view(argv[1]) == "--diagnostics")
        return run_diagnostics();
    DJ_HOST_ASSERT(argc == 1, "unexpected arguments");

    const auto cache_root = get_cache_root();
    deep_jit::safe_remove_all(cache_root);
    deep_jit::make_dirs(cache_root);
    set_env("TEST_JIT_CACHE_DIR", (cache_root / "cache").string());

    const auto runtime = make_runtime(get_test_maca_project_dir() / "kernels");

    // Case names follow the CUDA and Ascend harnesses ("<subject> <aspect>",
    // spaces) so the same case can be located across the three.
    run_test("environment precedence", [&] { test_environment(cache_root); });
    run_test("configuration", [&] { test_config(); });
    run_test("filesystem utilities", [&] { test_filesystem(cache_root); });
    run_test("external command and UUID", [&] { test_command_and_uuid(); });
    run_test("disk cache lifecycle", [&] { test_disk_cache(cache_root); });
    run_test("memory cache", [&] { test_memory_cache(); });
    run_test("JSON serialization", [&] { test_json(); });
    run_test("hash boundaries", [&] { test_hash_boundaries(); });
    run_test("hash robustness", [&] { test_hash_robustness(); });
    run_test("lazy initialization", [&] { test_lazy_init(); });
    run_test("lazy runtime", [&] { test_lazy_runtime(cache_root); });
    run_test("include parser", [&] { test_parser(cache_root); });
    run_test("include dirs and signature hashes", [&] { test_include_dirs(cache_root); });
    run_test("untracked dependency include flags", [&] { test_untracked_dependency_include_flags(*runtime, cache_root); });
    run_test("relocated wheel include directory", [&] { test_relocated_wheel_include_dir(cache_root); });
    run_test("MACA toolkit discovery", [&] { test_toolkit_discovery(cache_root); });
    run_test("MACA device", [&] { test_device_properties(*runtime); });
    run_test("compiler options", [&] { test_options(*runtime); });
    run_test("consumer default compiler flags", [&] { test_consumer_default_compiler_flags(cache_root); });
    run_test("debug environment", [&] { test_debug_environment(cache_root); });
    run_test("DG and EP environment compatibility", [&] { test_library_environment_compatibility(*runtime, cache_root); });
    run_test("launch options override", [&] { test_launch_options_override(*runtime); });
    run_test("launch option validation", [&] { test_launch_option_validation(*runtime); });
    run_test("multidimensional launch", [&] { test_multidimensional_launch(*runtime); });
    run_test("cooperative launch", [&] { test_cooperative_launch(*runtime); });
    run_test("JIT round trip", [&] { test_jit_round_trip(*runtime); });
    run_test("JIT vector add", [&] { test_jit_vector_add(*runtime); });
    run_test("template hash and runtime arguments", [&] { test_template_hash_and_launch(*runtime); });
    run_test("kernel count", [&] { test_multiple_kernels_rejected(*runtime); });
    run_test("kernel lifecycle", [&] { test_kernel_lifecycle(cache_root); });
    run_test("compiler options affect result", [&] { test_compiler_options_affect_result(*runtime); });
    run_test("mixed kernel arguments", [&] { test_mixed_arguments(*runtime); });
    run_test("large kernel arguments", [&] { test_large_kernel_arguments(*runtime); });
    run_test("macro-selected kernel ABI", [&] { test_macro_selected_abi(*runtime); });
    run_test("resource usage checks", [&] { test_resource_usage_checks(*runtime); });
    run_test("cache key tracks flags", [&] { test_cache_key_tracks_flags(*runtime); });
    run_test("cache key option matrix", [&] { test_cache_key_option_matrix(*runtime); });
    run_test("cache artifacts", [&] { test_cache_artifacts(*runtime, cache_root); });
    run_test("backend output validation", [&] { test_backend_output_validation(*runtime, cache_root); });
    run_test("assembly dump", [&] { test_dump_asm(*runtime); });
    run_test("dump options on cache hit", [&] { test_dump_options_on_cache_hit(cache_root); });
    run_test("post hook", [&] { test_post_hook(*runtime, cache_root); });
    run_test("default post hook", [&] { test_default_post_hook(cache_root); });
    run_test("generated include graph", [&] { test_generated_include_graph(cache_root); });
    run_test("compiler failure cleanup", [&] { test_compiler_failure_cleanup(*runtime); });
    run_test("invalid cache tag", [&] { test_invalid_cache_tag(*runtime); });
    run_test("secondary disk cache", [&] { test_secondary_disk_cache(cache_root); });
    run_test("architecture override", [&] { test_architecture_override(*runtime); });
    run_test("cross-family compilation", [&] { test_cross_family_compilation(*runtime, cache_root); });
    run_test("multiple runtimes", [&] { test_multiple_runtimes(cache_root); });
    run_test("large dynamic shared memory", [&] { test_large_dynamic_shared_memory(*runtime); });
    run_test("multiple devices", [&] { return test_multiple_devices(cache_root); });

    // Nothing may be left half-published behind the suite.  The JIT cache root
    // is `<cache_root>/cache` (see the `TEST_JIT_CACHE_DIR` above) and
    // `DiskCache` stages under `<jit_cache_root>/tmp`, so the directory to
    // inspect is the one below, not `cache_root` itself.
    check_tmp_is_empty(cache_root / "cache");

    std::printf("\nAll MACA DeepJIT tests passed.\n");
    return 0;
}
