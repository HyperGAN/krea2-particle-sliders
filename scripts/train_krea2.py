#!/usr/bin/env python3
"""Train a Krea2 turbo-bbox concept slider from this repo.

    python scripts/train_krea2.py --help
    python scripts/train_krea2.py --dummy

Does not look for a particle-sliders checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krea2.train import main  # noqa: E402


if __name__ == "__main__":
    main()
