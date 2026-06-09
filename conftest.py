"""Rend les packages du repo importables par pytest sans installation."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
