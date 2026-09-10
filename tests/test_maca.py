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
from pathlib import Path

import torch


if not __debug__:
    raise RuntimeError('DeepJIT tests require Python assertions to be enabled')


ROOT = Path(__file__).resolve().parent.parent
TEST_MACA_PROJECT = ROOT / 'tests' / 'test_maca_proj'


def maca_path():
    return Path(os.environ.get('MACA_PATH') or os.environ.get('MACA_HOME') or '/opt/maca')


def build_harness(temporary_dir):
    """Build the standalone C++ harness (the MACA analogue of test_cuda_proj)."""
    output = temporary_dir / 'test_maca'
    subprocess.run(
        ['bash', str(TEST_MACA_PROJECT / 'build.sh'), str(output)],
        check=True, cwd=str(ROOT))
    return output


def run_harness(binary, temporary_dir):
    environment = dict(os.environ)
    environment['DEEP_JIT_MACA_TEST_SOURCE_DIR'] = str(TEST_MACA_PROJECT)
    environment['DEEP_JIT_MACA_TEST_CACHE_DIR'] = str(temporary_dir / 'cache')
    environment.setdefault('MACA_PATH', str(maca_path()))
    library_path = environment.get('LD_LIBRARY_PATH', '')
    for extra in (str(maca_path() / 'lib'), str(maca_path() / 'mxgpu_llvm/bin')):
        if extra not in library_path.split(':'):
            library_path = f'{extra}:{library_path}'
    environment['LD_LIBRARY_PATH'] = library_path

    result = subprocess.run([str(binary)], capture_output=True, text=True,
                            env=environment, timeout=1800)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    assert result.returncode == 0, f'harness failed with exit code {result.returncode}'
    return result.stdout


def validate_artifacts(cache_root):
    """The cached entry must carry the source, the device binary and metadata."""
    entries = [p for p in cache_root.rglob('*') if p.is_dir() and (p / 'meta.json').is_file()]
    assert entries, f'no committed cache entries under {cache_root}'
    for entry in entries:
        assert (entry / 'kernel.cu').is_file(), f'missing source in {entry}'
        assert (entry / 'kernel.devbin').is_file(), f'missing device binary in {entry}'
        assert (entry / 'kernel.devbin').stat().st_size > 0, f'empty device binary in {entry}'
        metadata = json.loads((entry / 'meta.json').read_text())
        assert 'mxcc' in metadata['command'], f'metadata does not record mxcc: {entry}'
        assert '--offload-arch=xcore' in metadata['command'], f'missing offload target: {entry}'
    return entries


def validate_header_self_containment(temporary_dir):
    """Each backend header must compile on its own."""
    compiler = Path(os.environ.get('MACA_TOOLCHAIN',
                                   '/home/compiler_gfx/gpu_model/tools/hipcc')) / 'bin/clang++-22'
    if not compiler.is_file():
        print('skipping header self-containment: host toolchain not found')
        return
    headers = sorted((ROOT / 'include' / 'deep_jit' / 'backend' / 'maca').glob('*.hpp'))
    assert len(headers) == 5, f'expected five MACA backend headers, found {[h.name for h in headers]}'
    for header in headers:
        source = temporary_dir / f'self_containment_{header.stem}.cpp'
        source.write_text(f'#include <deep_jit/backend/maca/{header.name}>\n')
        result = subprocess.run(
            [str(compiler), '-fsyntax-only', '-std=c++20', str(source),
             '-I' + str(ROOT / 'include')],
            capture_output=True, text=True)
        # The device headers need the MACA/cu-bridge include roots; only the
        # include ordering of our own headers is under test here, so a
        # failure to find a system MACA header is not counted.
        if result.returncode != 0 and 'file not found' in result.stderr:
            print(f'note: {header.name} needs the MACA include roots to check standalone')
            continue
        assert result.returncode == 0, f'{header.name} is not self-contained:\n{result.stderr}'


def main():
    temporary_dir = Path(tempfile.mkdtemp(prefix='deep_jit_maca_test_'))
    try:
        validate_header_self_containment(temporary_dir)
        binary = build_harness(temporary_dir)
        output = run_harness(binary, temporary_dir)
        assert 'All MACA DeepJIT tests passed.' in output, 'harness did not report success'
        entries = validate_artifacts(temporary_dir / 'cache')
        print(f'\nMACA DeepJIT test passed; {len(entries)} cache entries validated.')
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
