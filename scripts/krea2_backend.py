#!/usr/bin/env python3
"""Find the particle-sliders Krea-2 turbo-bbox entrypoint, or fail closed.

Stock ``train_lora_krea.py`` loads a full ``Krea2Pipeline`` from
``--model_id`` (default ``krea/Krea-2-Raw``). It does not swap in
``jimmycarter/krea2-turbo-bbox`` at ``epoch-14-step-73184/transformer``.
This wrapper never calls that trainer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krea2_defaults import (  # noqa: E402
    CONTROL_PROMPT,
    HOLD_WEIGHT,
    INTENDED_INFER,
    INTENDED_TRAINER,
    MODEL_ID,
    PARTICLE_SLIDERS_REPO,
    PIPELINE_ID,
    STOCK_DOC,
    STOCK_LIVE,
    STOCK_TRAINER,
    TRANSFORMER_SUBFOLDER,
    TURBO_GUIDANCE,
    TURBO_MU,
    TURBO_STEPS,
)

EXIT_MISSING = 2


def candidate_roots() -> list[Path]:
    """Roots that may contain particle-sliders.

    ``PARTICLE_SLIDERS_ROOT`` always wins, including when it is empty
    (empty means "do not search"). Otherwise a sibling checkout named
    ``particle-sliders`` is the only implicit candidate.
    """
    if "PARTICLE_SLIDERS_ROOT" in os.environ:
        raw = os.environ.get("PARTICLE_SLIDERS_ROOT", "").strip()
        return [Path(raw)] if raw else []
    sibling = REPO_ROOT.parent / "particle-sliders"
    return [sibling] if sibling.is_dir() else []


def entrypoint(kind: str) -> str:
    if kind == "train":
        return INTENDED_TRAINER
    if kind == "infer":
        return INTENDED_INFER
    raise ValueError(f"unknown backend kind: {kind}")


def resolve_entrypoint(kind: str) -> Path | None:
    relative = entrypoint(kind)
    for root in candidate_roots():
        path = root / relative
        if path.is_file():
            return path
    return None


def default_args(kind: str) -> list[str]:
    args = [
        "--model_id",
        MODEL_ID,
        "--transformer_subfolder",
        TRANSFORMER_SUBFOLDER,
        "--pipeline_id",
        PIPELINE_ID,
        "--sample_steps",
        str(TURBO_STEPS),
        "--sample_guidance",
        str(TURBO_GUIDANCE),
        "--mu",
        str(TURBO_MU),
    ]
    if kind == "train":
        args.extend(
            [
                "--hold_weight",
                str(HOLD_WEIGHT),
                "--prompts_file",
                "configs/krea2/prompts-expression.yaml",
                "--config_file",
                "configs/krea2/config-expression.yaml",
                "--control_prompt",
                CONTROL_PROMPT,
            ]
        )
    return args


def merge_args(kind: str, argv: list[str]) -> list[str]:
    """Prepend product defaults for flags the caller did not set.

    ``--prompts_file`` and ``--config_file`` are a pair. If the caller
    sets either one, the expression-card default for both is left out
    so a lighting prompt file is not trained against the expression config.
    """
    present = set(argv)
    skip_pair = "--prompts_file" in present or "--config_file" in present
    merged: list[str] = []
    defaults = default_args(kind)
    index = 0
    while index < len(defaults):
        flag = defaults[index]
        value = defaults[index + 1]
        index += 2
        if flag in present:
            continue
        if skip_pair and flag in ("--prompts_file", "--config_file"):
            continue
        merged.extend([flag, value])
    return merged + list(argv)


def intended_command(kind: str) -> str:
    script = entrypoint(kind)
    parts = ["python", script, *default_args(kind)]
    if kind == "train":
        parts.append("--dummy")
    return " \\\n  ".join(parts)


def missing_message(kind: str) -> str:
    looked = candidate_roots()
    if not looked:
        where = "PARTICLE_SLIDERS_ROOT is unset and no sibling particle-sliders checkout was found"
    else:
        lines = "\n".join(f"  {root / entrypoint(kind)}" for root in looked)
        where = "not found:\n" + lines
    return f"""krea2-concept-sliders: the particle-sliders {kind} backend is not available ({where}).

This repo is the product surface. It does not vendor weights and it will
not fall back to {STOCK_TRAINER}. That trainer's live loader calls
Krea2Pipeline.from_pretrained(--model_id) and defaults to krea/Krea-2-Raw.
A transformer-only upload at {MODEL_ID} ({TRANSFORMER_SUBFOLDER}) needs
its own entrypoint, which loads

  Krea2Transformer2DModel.from_pretrained(
      {MODEL_ID!r}, subfolder={TRANSFORMER_SUBFOLDER!r})
  Krea2Pipeline.from_pretrained({PIPELINE_ID!r}, transformer=tf)

and samples at {TURBO_STEPS} steps, guidance_scale={TURBO_GUIDANCE}, mu={TURBO_MU}.

Intended CLI once {entrypoint(kind)} exists:

{intended_command(kind)}

Point PARTICLE_SLIDERS_ROOT at a particle-sliders checkout that contains
that file. Sibling work on the stock Raw / official-Turbo path:

  {PARTICLE_SLIDERS_REPO}/blob/main/{STOCK_TRAINER}
  {PARTICLE_SLIDERS_REPO}/blob/main/{STOCK_LIVE}
  {PARTICLE_SLIDERS_REPO}/blob/main/{STOCK_DOC}

Nothing was downloaded.
"""


def apply_hub_policy(argv: list[str]) -> None:
    """Keep a smoke from hitting the Hub unless the caller opts in."""
    if "--allow_hub" in argv:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def run(kind: str, argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_hub_policy(argv)
    path = resolve_entrypoint(kind)
    if path is None:
        print(missing_message(kind), file=sys.stderr)
        return EXIT_MISSING
    cmd = [sys.executable, str(path), *merge_args(kind, argv)]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)
