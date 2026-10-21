#!/usr/bin/env python3
"""Train a Krea2 turbo-bbox concept slider from this repo.

    python scripts/train_krea2.py --help
    python scripts/train_krea2.py --dummy

The live game executes particle_sliders.pg09.ConditionalPG09 with released
ParticleGAN 0.9.1. Hub id, Comfy, turbo samples and prompt cards stay here.
Dummy output is a diagnostic and never trains product weights.
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
