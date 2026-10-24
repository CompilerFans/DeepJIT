# Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import sys
from pathlib import Path


binary_path = Path(sys.argv[1])
assert len(sys.argv) == 2
assert binary_path.is_absolute()
assert Path.cwd() == binary_path.parent

# Edit the artifact and leave a stray file, then fail: the whole compilation
# must fail with nothing published or left staged.
with binary_path.open('ab') as artifact:
    artifact.write(b'DJ_POST_HOOK_FAILED')
(binary_path.parent / 'partial_hook_output').write_text('partial', encoding='utf-8')
raise SystemExit(7)
