#!/usr/bin/env python3
"""Infer wrapper for Krea-2 turbo-bbox concept sliders.

Forwards to the same file as training:
``conceptmod/textsliders/train_lora_krea2.py`` (particle-sliders PR #131).
Requires ``--load_te_lora PATH``, which skips the train loop and writes
the sample grid. There is no separate infer script and no adapter in
this repo to download.
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
