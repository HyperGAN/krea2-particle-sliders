#!/usr/bin/env python3
"""Train wrapper for Krea-2 turbo-bbox concept sliders.

Forwards to particle-sliders ``conceptmod/textsliders/train_lora_krea2.py``
(PR #131). A missing local checkout prints a clone / PYTHONPATH hint
and exits 2. Does not vendor the backend, download Hub weights, or
call ``train_lora_krea.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from krea2_backend import run  # noqa: E402


def main() -> None:
    sys.exit(run("train"))


if __name__ == "__main__":
    main()
