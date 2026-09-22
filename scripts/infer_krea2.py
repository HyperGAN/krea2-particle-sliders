#!/usr/bin/env python3
"""Sample a Krea2 turbo-bbox slider from this repo.

    python scripts/infer_krea2.py --help
    python scripts/infer_krea2.py --dummy --load_te_lora models/smile-krea2-bbox_lora

Same code as training. ``--load_te_lora`` skips the train loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krea2.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
