from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from gauge_reader.evaluate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
