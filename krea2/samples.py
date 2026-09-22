"""Smile-first sample grid. Dummy images are tiny PNGs, not Hub samples."""

from __future__ import annotations

import json
from pathlib import Path

from krea2.defaults import CONTROL_PROMPT, SAMPLE_SCALES
from krea2.dummy import DummyBackend
from krea2.prompts import SliderPrompt


def infer_sample_prompts(
    prompts: list[SliderPrompt],
    control_prompt: str = CONTROL_PROMPT,
) -> list[str]:
    seen: list[str] = []
    for prompt in prompts:
        caption = (prompt.neutral or prompt.target or "").strip()
        if caption and caption not in seen:
            seen.append(caption)
    control = str(control_prompt or "").strip()
    if control and control not in seen:
        seen.append(control)
    return seen


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(part for part in cleaned.split("-") if part)[:48] or "prompt"


def emit_grid(
    backend: DummyBackend,
    prompts: list[SliderPrompt],
    save_dir: Path,
    *,
    dummy: bool,
    control_prompt: str,
    card: dict[str, float | int | str],
    sample_seed: int = 42,
) -> list[dict[str, object]]:
    sample_prompts = infer_sample_prompts(prompts, control_prompt)
    scales = list(SAMPLE_SCALES)
    out_dir = Path(save_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = 2 if dummy else int(card["sample_steps"])
    guidance = float(card["sample_guidance"])
    height = 8 if dummy else 64
    records: list[dict[str, object]] = []
    for prompt in sample_prompts:
        for scale in scales:
            image = backend.generate(
                prompt,
                seed=int(sample_seed),
                num_steps=steps,
                guidance=guidance,
                scale=float(scale),
                height=height,
                width=height,
            )
            name = f"final_{_slug(prompt)}_scale{float(scale):g}.png"
            path = out_dir / name
            image.save(path)
            records.append(
                {
                    "prompt": prompt,
                    "scale": float(scale),
                    "path": name,
                    "seed": int(sample_seed),
                    "sample_steps": steps,
                    "cfg": guidance,
                    "height": image.height,
                    "width": image.width,
                    "control": prompt == control_prompt,
                }
            )
    payload = {
        "dummy": bool(dummy),
        "seed": int(sample_seed),
        "scales": scales,
        "prompts": sample_prompts,
        "gate": "smile-first",
        "crop_purity": False,
        "samples": records,
    }
    (out_dir / "final_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError("Krea2 sample grid is empty")
    return records
