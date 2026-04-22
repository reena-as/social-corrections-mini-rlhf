import sys
from pathlib import Path

# Ensure `src/` is on sys.path when running pytest from the repo root without
# first doing `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
