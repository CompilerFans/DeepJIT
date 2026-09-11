import sys
from pathlib import Path


binary_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert binary_path.is_absolute()
assert Path.cwd() == binary_path.parent
assert binary_path.is_file()

# The hook runs before the artifact is published and the device binary is
# loaded afterwards, so it stays valid: record the fact that it ran in a
# sidecar file inside the artifact directory instead of rewriting it.
(Path.cwd() / 'post_hook.marker').write_text('first\n')
