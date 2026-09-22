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
    BACKEND_DOC,
    BACKEND_PR,
    CONTROL_PROMPT,
    HOLD_WEIGHT,
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
    if kind in ("train", "infer"):
        return INTENDED_TRAINER
    raise ValueError(f"unknown backend kind: {kind}")


def resolve_entrypoint(kind: str) -> Path | None:
    relative = entrypoint(kind)
    for root in candidate_roots():
        path = root / relative
        if path.is_file():
            return path
    return None


def default_args(kind: str) -> list[str]:
    """Flags from particle-sliders PR #131 (``train_lora_krea2.py``).

    ``--skeleton_model`` is that trainer's name for the Raw pipeline
    that supplies the VAE, text encoder, and scheduler. Product yamls
    record the same id as ``pretrained_model.pipeline_id``.
    """
    prompts = REPO_ROOT / "configs" / "krea2" / "prompts-expression.yaml"
    config = REPO_ROOT / "configs" / "krea2" / "config-expression.yaml"
    args = [
        "--model_id",
        MODEL_ID,
        "--transformer_subfolder",
        TRANSFORMER_SUBFOLDER,
        "--skeleton_model",
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
                str(prompts),
                "--config_file",
                str(config),
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


def _shell_token(token: str) -> str:
    if token == "" or any(ch.isspace() for ch in token):
        return "'" + token.replace("'", "'\"'\"'") + "'"
    return token


def intended_command(kind: str) -> str:
    tokens = ["python", entrypoint(kind), *default_args(kind)]
    if kind == "train":
        tokens.append("--dummy")
    else:
        tokens.extend(["--load_te_lora", "models/expression-krea2_lora"])
    lines = [f"python {entrypoint(kind)}"]
    rest = tokens[2:]
    index = 0
    while index < len(rest):
        flag = rest[index]
        if (
            flag.startswith("--")
            and index + 1 < len(rest)
            and not rest[index + 1].startswith("--")
        ):
            lines.append(f"{_shell_token(flag)} {_shell_token(rest[index + 1])}")
            index += 2
            continue
        lines.append(_shell_token(flag))
        index += 1
    return " \\\n  ".join(lines)


def missing_message(kind: str) -> str:
    looked = candidate_roots()
    if not looked:
        where = "PARTICLE_SLIDERS_ROOT is unset and no sibling particle-sliders checkout was found"
    else:
        lines = "\n".join(f"  {root / INTENDED_TRAINER}" for root in looked)
        where = "not found:\n" + lines
    return f"""krea2-concept-sliders: {INTENDED_TRAINER} is not in the local checkout ({where}).

Install it from particle-sliders PR #131 (draft). Do not vendor that
backend into this repo.

  {BACKEND_PR}
  {INTENDED_TRAINER}
  {BACKEND_DOC}

  git clone {PARTICLE_SLIDERS_REPO}.git
  cd particle-sliders
  git fetch origin pull/131/head:pr-131
  git checkout pr-131
  export PARTICLE_SLIDERS_ROOT="$PWD"
  export PYTHONPATH="$PARTICLE_SLIDERS_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
  PYTHONPATH=. python {INTENDED_TRAINER} --print_card

Then, from this product repo:

{intended_command(kind)}

Both wrappers call {INTENDED_TRAINER}. Infer adds --load_te_lora so the
train loop is skipped. This product does not call {STOCK_TRAINER}
(stock Raw / official Turbo). Sample card stays {TURBO_STEPS} steps,
guidance_scale={TURBO_GUIDANCE}, mu={TURBO_MU}, skeleton {PIPELINE_ID},
transformer {MODEL_ID} @ {TRANSFORMER_SUBFOLDER}.

Stock Raw notes, not this trainer:

  {PARTICLE_SLIDERS_REPO}/blob/main/{STOCK_TRAINER}
  {PARTICLE_SLIDERS_REPO}/blob/main/{STOCK_LIVE}
  {PARTICLE_SLIDERS_REPO}/blob/main/{STOCK_DOC}

Nothing was downloaded.
"""


def infer_usage() -> str:
    return f"""krea2-concept-sliders: infer is {INTENDED_TRAINER} --load_te_lora PATH.

That flag skips the train loop and writes the sample grid. There is no
separate infer_lora_krea2.py. Pass the adapter directory from a previous
train. This repo does not ship one.

  python scripts/infer_krea2.py --dummy --load_te_lora models/expression-krea2_lora

Guide: {BACKEND_PR} ({BACKEND_DOC}).
"""


def apply_hub_policy(argv: list[str]) -> None:
    """Keep a smoke from hitting the Hub unless the caller opts in."""
    if "--allow_hub" in argv:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def child_env(backend_path: Path) -> dict[str, str]:
    """PYTHONPATH includes the particle-sliders checkout that owns the trainer."""
    env = os.environ.copy()
    root = str(backend_path.resolve().parents[2])
    prior = env.get("PYTHONPATH", "")
    parts = [root] if not prior else [root, prior]
    # Avoid duplicating the root when the caller already exported it.
    if prior.split(os.pathsep)[0] == root:
        parts = [prior]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run(kind: str, argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    apply_hub_policy(argv)
    path = resolve_entrypoint(kind)
    if path is None:
        print(missing_message(kind), file=sys.stderr)
        return EXIT_MISSING
    if kind == "infer" and "--load_te_lora" not in argv:
        print(infer_usage(), file=sys.stderr)
        return EXIT_MISSING
    cmd = [sys.executable, str(path), *merge_args(kind, argv)]
    completed = subprocess.run(cmd, check=False, env=child_env(path))
    return int(completed.returncode)
