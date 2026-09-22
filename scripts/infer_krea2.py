#!/usr/bin/env python3
"""Infer wrapper for Krea-2 turbo-bbox concept sliders.

Forwards to particle-sliders ``infer_lora_krea2.py`` when that file is
on disk. Until then, exits 2. There is no trained adapter in this repo
to run, and this script does not download one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from krea2_backend import run  # noqa: E402


def main() -> None:
    sys.exit(run("infer"))


if __name__ == "__main__":
    main()
