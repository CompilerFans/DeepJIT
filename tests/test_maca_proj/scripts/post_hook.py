import sys
from pathlib import Path


binary_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert binary_path.is_absolute()
assert Path.cwd() == binary_path.parent
assert binary_path.is_file()

# The hook must edit the artifact in place -- that edit has to be what gets
# published, which is the property `test_post_hook` checks by looking for this
# marker inside the cached `kernel.devbin`.
#
# It APPENDS rather than overwriting a header field: measured on this platform,
# a `kernel.devbin` with trailing bytes still satisfies `mcModuleLoad`,
# `mcModuleGetFunction` and `mcModuleLaunchKernelEx`, so the same hook can also
# prove that a hooked artifact stays loadable.  (CUDA's hook overwrites the
# cubin's first four bytes instead, which no load path survives.)
with binary_path.open('ab') as artifact:
    artifact.write(b'DJ_POST_HOOK_MARKER')

# The sidecar records that the hook ran at all, for the assertions about the
# hook's environment rather than its effect on the artifact.
(Path.cwd() / 'post_hook.marker').write_text('first\n')
