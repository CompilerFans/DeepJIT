# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import sys
from pathlib import Path


binary_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert binary_path.is_absolute()
assert Path.cwd() == binary_path.parent
assert binary_path.is_file()

# A marker distinct from `post_hook.py`'s, proving which hook ran.
with binary_path.open('ab') as artifact:
    artifact.write(b'DJ_POST_HOOK_ALT_MARKER')

(Path.cwd() / 'post_hook.marker').write_text('second\n')
