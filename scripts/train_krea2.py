#!/usr/bin/env python3
"""Train wrapper for Krea-2 turbo-bbox concept sliders.

Forwards to particle-sliders ``train_lora_krea2.py`` when that file is
on disk. Until then, exits 2 with the intended CLI. Does not download
Hub weights and does not call ``train_lora_krea.py``.
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
