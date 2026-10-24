import sys
from pathlib import Path


binary_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert binary_path.is_absolute()
assert Path.cwd() == binary_path.parent
assert binary_path.is_file()

# Edit the artifact in place and append rather than overwrite: trailing bytes
# keep the device binary loadable, and `test_post_hook` looks for this marker
# in the published entry.
with binary_path.open('ab') as artifact:
    artifact.write(b'DJ_POST_HOOK_MARKER')

# The sidecar records that the hook ran at all, for the assertions about the
# hook's environment rather than its effect on the artifact.
(Path.cwd() / 'post_hook.marker').write_text('first\n')
