from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABEL_LAB_SRC = ROOT / "gauge_label_lab" / "src"
GAUGE_READER_SRC = ROOT / "src"
sys.path.insert(0, str(LABEL_LAB_SRC))
sys.path.insert(0, str(GAUGE_READER_SRC))

from gauge_label_lab.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
