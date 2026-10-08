import os
import sys
import tempfile
import traceback
from pathlib import Path

from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[1]


def extra_include_paths():
    """Include roots the embedded headers need beyond this repository's own.

    `deep_jit/utils/exception.hpp` includes `<elfutils/libdwfl.h>`, from the
    platform's libdw, and the headers format through the vendored fmt checkout
    (`FMT_ROOT`, a sibling by default) rather than `<format>`.  libdw's root is
    not on every host's default search path -- the MACA host toolchain carries
    it, the system default is expected to -- so it is located rather than
    assumed, and added only when found.
    """
    toolchain = Path(os.environ.get('MACA_TOOLCHAIN') or '/home/compiler_gfx/gpu_model/tools/hipcc')
    maca = Path(os.environ.get('MACA_PATH') or os.environ.get('MACA_HOME') or '/opt/maca')
    fmt = Path(os.environ.get('FMT_ROOT') or ROOT.parent / 'fmt') / 'include'
    assert (fmt / 'fmt/format.h').is_file(), f'no fmt checkout at {fmt.parent}: set FMT_ROOT'

    roots = [ROOT / 'include', fmt]
    for candidate in (toolchain / 'include', maca / 'include'):
        if (candidate / 'elfutils/libdwfl.h').is_file():
            roots.append(candidate)
            break
    return [str(root) for root in roots]


CPP_SOURCE = r'''
#line 1 "embedded_exception_test.cpp"
#include <pybind11/pybind11.h>

#include <deep_jit/utils/exception.hpp>

namespace deep_jit::exception_test {

__attribute__((noinline, visibility("default"))) void panic_leaf() {
    DJ_PANIC("test failure from pybind11");
}

__attribute__((noinline, visibility("default"))) void panic_middle() {
    panic_leaf();
}

__attribute__((noinline, visibility("default"))) void panic_entry() {
    panic_middle();
}

}  // namespace deep_jit::exception_test

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("raise_exception", &deep_jit::exception_test::panic_entry);
}
'''


def python_entry(module):
    module.raise_exception()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='deep-jit-exception-') as directory:
        module = load_inline(
            name='deep_jit_exception_test',
            cpp_sources=CPP_SOURCE,
            extra_cflags=[
                '-std=c++20',
                '-O3',
                '-g1',
                '-fno-omit-frame-pointer',
                '-fno-optimize-sibling-calls',
            ],
            extra_include_paths=extra_include_paths(),
            build_directory=directory,
            with_cuda=False,
            verbose=True,
        )

        try:
            python_entry(module)
        except RuntimeError:
            python_traceback = traceback.format_exc()
        else:
            raise AssertionError('the pybind11 call did not raise RuntimeError')

        print(python_traceback, file=sys.stderr, end='')
        assert 'Traceback (most recent call last):' in python_traceback
        assert 'in python_entry' in python_traceback
        assert (
            'RuntimeError: Panic error (embedded_exception_test.cpp:8): '
            'test failure from pybind11\n'
            'C++ trace (most recent call first):\n'
        ) in python_traceback
        assert 'deep_jit::exception_test::panic_leaf()' in python_traceback
        assert 'deep_jit::exception_test::panic_middle()' in python_traceback
        assert 'deep_jit::exception_test::panic_entry()' in python_traceback
        assert 'embedded_exception_test.cpp:' in python_traceback


if __name__ == '__main__':
    main()
