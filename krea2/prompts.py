"""Bare UNI prompt rows. Attributes are bookkeeping, not a caption prefix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from krea2.defaults import CONTROL_PROMPT, RAW_GUIDANCE


@dataclass
class SliderPrompt:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = 0.0
    batch_size: int = 1
    unconditional: str = ""


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [0.0, 2.0])
    concept_words: str = ""
    control_prompt: str = CONTROL_PROMPT
    bare_captions: bool = False


def word_tokens(text: str) -> list[str]:
    cleaned = str(text).replace(",", " ").replace(".", " ")
    return [part.lower() for part in cleaned.split() if part.strip()]


def unused_words_for(prompt: SliderPrompt) -> list[str]:
    words: list[str] = []
    for attr in prompt.attributes:
        words.extend(word_tokens(attr))
    return words


def expand_attributes(row: dict, *, prefix: bool = True) -> list[dict]:
    attributes = [str(a).strip() for a in (row.get("attributes") or []) if str(a).strip()]
    if not attributes:
        return [dict(row)]
    if not prefix:
        item = dict(row)
        item["attributes"] = attributes
        return [item]
    rows = []
    for attribute in attributes:
        item = dict(row)
        item["attributes"] = attributes
        for key in ("target", "positive", "negative", "neutral"):
            value = row.get(key)
            if value:
                item[key] = f"{attribute} {value}"
        rows.append(item)
    return rows


def load_prompts(path: Path) -> tuple[list[SliderPrompt], PromptsMeta]:
    with Path(path).open(encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    meta = PromptsMeta()
    prefix_attributes = True
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        meta.concept_words = str(raw.get("concept_words") or "")
        meta.control_prompt = str(raw.get("control_prompt") or CONTROL_PROMPT)
        bare = bool(raw.get("bare_captions"))
        if "prefix_attributes" in raw:
            prefix_attributes = bool(raw.get("prefix_attributes"))
        elif bare:
            prefix_attributes = False
        meta.bare_captions = not prefix_attributes
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[SliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "positive" not in item:
            raise ValueError(f"each prompt must be a mapping with positive: {item!r}")
        for row in expand_attributes(item, prefix=prefix_attributes):
            target = str(row.get("target") or row.get("neutral") or row["positive"])
            prompts.append(
                SliderPrompt(
                    target=target,
                    positive=str(row["positive"]),
                    neutral=str(row.get("neutral") or target),
                    negative=str(row.get("negative") or ""),
                    attributes=[str(a) for a in (row.get("attributes") or [])],
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", RAW_GUIDANCE)),
                    batch_size=int(row.get("batch_size", 1)),
                    unconditional=str(row.get("unconditional") or ""),
                )
            )
    return prompts, meta
