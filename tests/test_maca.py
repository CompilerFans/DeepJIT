"""DeepJIT MACA backend end-to-end test.

Builds the test extension with the C++20 host toolchain, then drives the
registered JIT runtime: mxcc compiles the kernel source, `mcModuleLoad`
loads the resulting device binary and the kernel runs on the C500.

Run with the MACA environment sourced (MACA_PATH, LD_LIBRARY_PATH including
`$MACA_PATH/lib` and `$MACA_PATH/mxgpu_llvm/lib`).
"""
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import torch


if not __debug__:
    raise RuntimeError('DeepJIT tests require Python assertions to be enabled')


ROOT = Path(__file__).resolve().parent.parent
TEST_MACA_PROJECT = ROOT / 'tests' / 'test_maca_proj'
COMPILER_OPTION_SUFFIXES = (
    'JIT_CHECK_NO_LOCAL_MEMORY',
    'JIT_CHECK_NO_SPILLS',
    'JIT_CPP_STANDARD',
    'JIT_DEBUG',
    'JIT_DUMP_ASM',
    'JIT_DUMP_PTX',
    'JIT_LLVM_NM',
    'JIT_MXCC_COMPILER',
    'JIT_PRINT_COMPILER_COMMAND',
    'JIT_PRINT_LOAD_TIME',
    'JIT_PTXAS_VERBOSE',
    'JIT_WITH_LINEINFO',
)


def maca_path():
    return Path(os.environ.get('MACA_PATH') or os.environ.get('MACA_HOME') or '/opt/maca')


def host_toolchain():
    """Root of the conda host toolchain that builds the harness.

    `MACA_TOOLCHAIN` names it directly; otherwise it is derived from the
    compiler named by `MACA_HOST_CXX`/`CXX` (both live at `<root>/bin/clang++-22`,
    the spelling the CUDA driver uses for its `CXX`).  The in-image default is
    only a default -- every path here is overridable from the environment -- and
    a toolchain that cannot be found is a hard error, never a skipped check.
    """
    root = os.environ.get('MACA_TOOLCHAIN')
    if root is None:
        for name in ('MACA_HOST_CXX', 'CXX'):
            compiler = os.environ.get(name)
            if compiler:
                resolved = Path(shutil.which(compiler) or compiler).resolve()
                root = resolved.parent.parent
                break
    if root is None:
        root = '/home/compiler_gfx/gpu_model/tools/hipcc'
    root = Path(root)
    assert (root / 'bin/clang++-22').is_file(), \
        f'no usable MACA host toolchain at {root}: set MACA_TOOLCHAIN'
    return root


def gcc_limits_include(toolchain):
    """GCC's own limits.h, which the sysroot's limits.h delegates to.

    It ships in the `llvm` tree beside the toolchain rather than inside it
    (the toolchain has the C++ headers but no limits.h of its own), so it is
    located by glob rather than by a pinned version.
    """
    explicit = os.environ.get('GCC_LIMITS_INCLUDE')
    if explicit:
        return Path(explicit)
    base = toolchain.parent / 'llvm' / 'lib' / 'gcc' / 'x86_64-conda-linux-gnu'
    candidates = sorted(base.glob('*/include/limits.h'))
    assert candidates, f'no GCC limits.h under {base}: set GCC_LIMITS_INCLUDE'
    return candidates[-1].parent


def clear_external_jit_environment():
    """Drop ambient JIT settings so the harness runs on its own defaults."""
    suffixes = COMPILER_OPTION_SUFFIXES + ('JIT_CACHE_DIR',)
    for name in tuple(os.environ):
        if name in suffixes or any(name.endswith('_' + suffix) for suffix in suffixes):
            os.environ.pop(name)


def build_harness(temporary_dir):
    """Build the standalone C++ harness (the MACA analogue of test_cuda_proj)."""
    output = temporary_dir / 'test_maca'
    subprocess.run(
        ['bash', str(TEST_MACA_PROJECT / 'build.sh'), str(output)],
        check=True, cwd=str(ROOT))
    return output


def run_harness(binary, temporary_dir):
    environment = harness_environment(temporary_dir)
    result = subprocess.run([str(binary)], capture_output=True, text=True,
                            env=environment, timeout=1800)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    assert result.returncode == 0, f'harness failed with exit code {result.returncode}'
    return result.stdout


def harness_environment(temporary_dir):
    environment = dict(os.environ)
    environment['DEEP_JIT_MACA_TEST_SOURCE_DIR'] = str(TEST_MACA_PROJECT)
    environment['DEEP_JIT_MACA_TEST_CACHE_DIR'] = str(temporary_dir / 'cache')
    environment.setdefault('MACA_PATH', str(maca_path()))
    library_path = environment.get('LD_LIBRARY_PATH', '')
    for extra in (str(maca_path() / 'lib'), str(maca_path() / 'mxgpu_llvm/bin')):
        if extra not in library_path.split(':'):
            library_path = f'{extra}:{library_path}'
    environment['LD_LIBRARY_PATH'] = library_path
    return environment


def validate_artifacts(cache_root):
    """The cached entry must carry the source, the device binary and metadata."""
    entries = [p for p in cache_root.rglob('*') if p.is_dir() and (p / 'meta.json').is_file()]
    assert entries, f'no committed cache entries under {cache_root}'
    for entry in entries:
        tag, digest = entry.name.rsplit('.', 1)
        assert tag.replace('_', '').isalnum(), entry
        assert len(digest) == 32 and all(character in '0123456789abcdef' for character in digest), entry
        assert (entry / '.committed').is_file(), f'missing commit marker in {entry}'
        assert (entry / 'kernel.cu').is_file(), f'missing source in {entry}'
        assert (entry / 'kernel.devbin').is_file(), f'missing device binary in {entry}'
        assert (entry / 'kernel.devbin').stat().st_size > 0, f'empty device binary in {entry}'
        assert {path.name for path in entry.iterdir()} <= {
            '.committed', 'kernel.cu', 'kernel.devbin', 'kernel.s', 'meta.json', 'post_hook.marker',
        }, entry

        metadata = json.loads((entry / 'meta.json').read_text())
        assert set(metadata) == {'command', 'config', 'compiler_info', 'compiler_options'}, metadata
        assert set(metadata['config']) == {
            'python_library_root', 'extra_signature', 'include_dirs', 'include_prefixes',
        }, metadata
        assert set(metadata['compiler_info']) == {'path', 'version'}, metadata
        assert set(metadata['compiler_options']) == {
            'optimize_level', 'fast_math', 'compiler_verbose', 'check_no_spills',
            'check_no_local_memory', 'with_line_info', 'dump_asm', 'arch', 'mxcc_flags',
            'extra_mxcc_flags', 'post_hook',
        }, metadata
        assert metadata['compiler_info']['path'] and metadata['compiler_info']['version'], metadata
        assert metadata['compiler_options']['arch'] and isinstance(metadata['compiler_options']['arch'], str), metadata
        assert isinstance(metadata['compiler_options']['mxcc_flags'], list), metadata
        assert 'mxcc' in metadata['command'], f'metadata does not record mxcc: {entry}'
        assert '--offload-arch=xcore' in metadata['command'], f'missing offload target: {entry}'
        assert '-device-bin' in metadata['command'], f'metadata does not record the device artifact: {entry}'
    return entries


def validate_header_self_containment(temporary_dir):
    """Each backend header must compile on its own.

    A missing toolchain is a failure, not a skip: silently dropping this check
    is how a header stops being self-contained without anyone noticing.
    """
    compiler = host_toolchain() / 'bin' / 'clang++-22'
    assert compiler.is_file(), f'no host compiler at {compiler}: set MACA_TOOLCHAIN'
    headers = sorted((ROOT / 'include' / 'deep_jit' / 'backend' / 'maca').glob('*.hpp'))
    assert len(headers) == 5, f'expected five MACA backend headers, found {[h.name for h in headers]}'
    for header in headers:
        source = temporary_dir / f'self_containment_{header.stem}.cpp'
        source.write_text(f'#include <deep_jit/backend/maca/{header.name}>\n')
        result = subprocess.run(
            header_self_containment_command(source),
            capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, f'{header.name} is not self-contained:\n{result.stderr}'
    print(f'validated {len(headers)} self-contained MACA backend headers', flush=True)


def fmt_root():
    """The root of the fmt checkout DeepJIT's headers format through.

    DeepJIT goes through fmt instead of `<format>` so that it stays embeddable
    on host toolchains whose C++20 library predates `<format>` (MACA's mxcc
    drives GCC 11, which has none).  Consuming repositories vendor fmt as a
    sibling checkout; `FMT_ROOT` overrides it.
    """
    return Path(os.environ.get('FMT_ROOT') or ROOT.parent / 'fmt')


def header_self_containment_command(source):
    """The device headers need the MACA, cu-bridge, torch and fmt include roots
    (plus the host C++20 standard library); the roots mirror `build.sh`."""
    toolchain = host_toolchain()
    sysroot = toolchain / 'x86_64-conda-linux-gnu' / 'sysroot'
    gcc15 = toolchain / 'lib' / 'gcc' / 'x86_64-conda-linux-gnu' / '15.2.0'
    gcc_limits = gcc_limits_include(toolchain)
    torch_root = Path(os.environ.get('TORCH_ROOT') or Path(torch.__file__).resolve().parent)
    maca = maca_path()

    command = [
        str(toolchain / 'bin/clang++-22'), '-fsyntax-only', '-std=c++20',
        '-Werror', '-Wno-attributes', '-Wno-deprecated-declarations',
        '-Wno-missing-field-initializers', '-Wno-psabi', '-Wno-unused-function',
        '-DUSE_MACA', '-D_GNU_SOURCE',
        f'--sysroot={sysroot}', f'--gcc-toolchain={toolchain}',
    ]
    for path in (
        ROOT / 'include',
        fmt_root() / 'include',
        maca / 'include', maca / 'include/mcr', maca / 'include/mcc',
        maca / 'include/mcsparse', maca / 'include/mcblas', maca / 'include/mcsolver',
        maca / 'include/misc', maca / 'include/common', maca / 'include/mctx',
        maca / 'tools/cu-bridge/include', toolchain / 'include',
        torch_root / 'include', torch_root / 'include/torch/csrc/api/include',
        Path(sysconfig.get_paths()['include']),
    ):
        command.append('-I' + str(path))
    for path in (
        gcc15 / 'include', gcc15 / 'include/c++',
        gcc15 / 'include/c++/x86_64-conda-linux-gnu', gcc15 / 'include/c++/backward',
        sysroot / 'usr/include', gcc_limits,
    ):
        command.extend(['-isystem', str(path)])
    command.append(str(source))
    return command


def validate_diagnostic_output(binary, temporary_dir):
    """Compiler-command and load-time reports follow the library prefix, the
    reserved `DJ_` default, and nothing else."""
    def run_case(cache_name, values):
        environment = harness_environment(temporary_dir)
        environment['DEEP_JIT_MACA_TEST_CACHE_DIR'] = str(temporary_dir / cache_name)
        for prefix in ('TEST_', 'DJ_', ''):
            for suffix in ('JIT_PRINT_COMPILER_COMMAND', 'JIT_PRINT_LOAD_TIME'):
                environment.pop(prefix + suffix, None)
        environment.update(values)
        return subprocess.run([str(binary), '--diagnostics'], env=environment,
                              capture_output=True, text=True, timeout=600, check=True)

    diagnostics = {
        'TEST_JIT_PRINT_COMPILER_COMMAND': '1',
        'TEST_JIT_PRINT_LOAD_TIME': '1',
    }
    first = run_case('diagnostic_cache_library', diagnostics)
    assert 'Running command:' in first.stdout, first.stdout
    assert 'Load time (' in first.stdout, first.stdout
    assert 'RESULT=42' in first.stdout, first.stdout

    # A cache hit must not compile again, but still reports the load.
    second = run_case('diagnostic_cache_library', diagnostics)
    assert 'Running command:' not in second.stdout, second.stdout
    assert 'Load time (' in second.stdout, second.stdout
    assert 'RESULT=42' in second.stdout, second.stdout

    global_fallback = run_case('diagnostic_cache_global', {
        'DJ_JIT_PRINT_COMPILER_COMMAND': '1',
        'DJ_JIT_PRINT_LOAD_TIME': '1',
    })
    assert 'Running command:' in global_fallback.stdout, global_fallback.stdout
    assert 'Load time (' in global_fallback.stdout, global_fallback.stdout

    library_false = run_case('diagnostic_cache_false', {
        'DJ_JIT_PRINT_COMPILER_COMMAND': '1',
        'DJ_JIT_PRINT_LOAD_TIME': '1',
        'TEST_JIT_PRINT_COMPILER_COMMAND': '0',
        'TEST_JIT_PRINT_LOAD_TIME': '0',
    })
    assert 'Running command:' not in library_false.stdout, library_false.stdout
    assert 'Load time (' not in library_false.stdout, library_false.stdout

    unprefixed = run_case('diagnostic_cache_unprefixed', {
        'JIT_PRINT_COMPILER_COMMAND': '1',
        'JIT_PRINT_LOAD_TIME': '1',
    })
    assert 'Running command:' not in unprefixed.stdout, unprefixed.stdout
    assert 'Load time (' not in unprefixed.stdout, unprefixed.stdout
    print('validated compiler and load diagnostic output', flush=True)


def validate_multiprocess_compile(binary, temporary_dir):
    """Several processes racing onto one entry must agree on the published
    artifact, and a later process must reuse it instead of compiling."""
    cache_root = temporary_dir / 'process_cache'
    coordination = temporary_dir / 'process_coordination'
    coordination.mkdir()
    start_path = coordination / 'start'
    environment = harness_environment(temporary_dir)
    environment['DEEP_JIT_MACA_TEST_CACHE_DIR'] = str(cache_root)

    def compile_once(ready_path, start, tag='process_cache', bias=89):
        return subprocess.run(
            [str(binary), '--compile-once', tag, str(bias), str(ready_path), str(start)],
            env=environment, capture_output=True, text=True, timeout=900)

    processes = []
    for index in range(4):
        ready_path = coordination / f'ready_{index}'
        processes.append(subprocess.Popen(
            [str(binary), '--compile-once', 'process_cache', '89', str(ready_path), str(start_path)],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))

    deadline = time.monotonic() + 300
    while len(list(coordination.glob('ready_*'))) != len(processes):
        for process in processes:
            if process.poll() is not None:
                raise AssertionError(process.communicate(timeout=5)[0])
        assert time.monotonic() < deadline, 'compile workers did not reach the barrier'
        time.sleep(0.01)
    start_path.touch()

    artifacts = []
    for process in processes:
        output, _ = process.communicate(timeout=900)
        assert process.returncode == 0, output
        artifact_lines = [line for line in output.splitlines() if line.startswith('ARTIFACT=')]
        assert len(artifact_lines) == 1, output
        assert 'RESULT=90' in output, output
        artifacts.append(Path(artifact_lines[0].removeprefix('ARTIFACT=')))

    assert len(set(artifacts)) == 1, artifacts
    entry = artifacts[0]
    assert entry.parent == cache_root / 'cache', entry
    tag, digest = entry.name.rsplit('.', 1)
    assert tag == 'process_cache', entry
    assert len(digest) == 32 and all(character in '0123456789abcdef' for character in digest), entry
    assert (entry / '.committed').is_file(), entry
    assert (entry / 'kernel.cu').is_file(), entry
    assert (entry / 'kernel.devbin').stat().st_size > 0, entry
    assert len(list((cache_root / 'cache').iterdir())) == 1, list((cache_root / 'cache').iterdir())
    stale = list((cache_root / 'tmp').iterdir()) if (cache_root / 'tmp').is_dir() else []
    assert not stale, f'uncommitted temporary entries were left behind: {stale}'

    # The start file already exists, so the barrier is a no-op for this run.
    reuse = compile_once(start_path, start_path)
    assert reuse.returncode == 0, reuse.stdout + reuse.stderr
    assert f'ARTIFACT={entry}' in reuse.stdout, reuse.stdout
    assert 'RESULT=90' in reuse.stdout, reuse.stdout
    assert len(list((cache_root / 'cache').iterdir())) == 1, list((cache_root / 'cache').iterdir())
    print('validated 4-process same-key compilation and cache reuse', flush=True)


def main():
    clear_external_jit_environment()
    temporary_dir = Path(tempfile.mkdtemp(prefix='deep_jit_maca_test_'))
    try:
        validate_header_self_containment(temporary_dir)
        binary = build_harness(temporary_dir)
        output = run_harness(binary, temporary_dir)
        assert 'All MACA DeepJIT tests passed.' in output, 'harness did not report success'
        entries = validate_artifacts(temporary_dir / 'cache')
        validate_multiprocess_compile(binary, temporary_dir)
        validate_diagnostic_output(binary, temporary_dir)
        print(f'\nMACA DeepJIT test passed; {len(entries)} cache entries validated.')
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
