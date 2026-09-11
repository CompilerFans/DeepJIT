import sys
from pathlib import Path


binary_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert binary_path.is_absolute()
assert Path.cwd() == binary_path.parent

# Edit the artifact and leave a stray file behind, then fail: a hook that dies
# must take the whole compilation with it, with nothing published and nothing
# left staged for publication.
with binary_path.open('ab') as artifact:
    artifact.write(b'DJ_POST_HOOK_FAILED')
(binary_path.parent / 'partial_hook_output').write_text('partial', encoding='utf-8')
raise SystemExit(7)
