"""Load plan and train card for jimmycarter/krea2-turbo-bbox.

No Hub download. Live weights go through ``krea2.live``.
"""

from __future__ import annotations

from pathlib import Path

from krea2.defaults import (
    AGE_HOLD_WEIGHT,
    BBOX_ID_MARKERS,
    COMFY_FILENAME,
    CONTROL_PROMPT,
    EPOCH,
    HOLD_WEIGHT,
    LORA_TARGETS_DEFAULT,
    MODEL_ID,
    RANK,
    RAW_GUIDANCE,
    RAW_STEPS,
    RESOLUTION,
    SKELETON_MODEL,
    TRANSFORMER_SUBFOLDER,
    TURBO_GUIDANCE,
    TURBO_MU,
    TURBO_STEPS,
)

_FOREIGN_BACKENDS = (
    "anima",
    "sana",
    "supra",
    "z-image",
    "zimage",
    "zit",
    "minimax",
    "h3",
)
_STOCK_PIPELINE_MARKERS = ("krea-2-raw", "krea-2-turbo")

DEFAULT_PROMPTS = "configs/krea2/prompts-smile.yaml"


def is_krea2_turbo_bbox_id(model_id: str) -> bool:
    text = str(model_id).lower().replace(" ", "")
    return any(marker in text for marker in BBOX_ID_MARKERS)


def _refuse(text: str, *, role: str) -> None:
    lowered = text.lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                "this trainer is Krea2-turbo-bbox-only; "
                f"refused foreign backend {name!r} in {role} {text!r}"
            )
    if is_krea2_turbo_bbox_id(lowered):
        return
    for marker in _STOCK_PIPELINE_MARKERS:
        if marker in lowered:
            raise ValueError(
                "this trainer is Krea2-turbo-bbox-only; refused stock Krea "
                f"pipeline id {text!r} ({role}). Raw CFG {RAW_GUIDANCE:g} / "
                f"{RAW_STEPS} steps stay on the stock Krea trainer. This "
                f"product trains and samples at CFG {TURBO_GUIDANCE:g}, "
                f"{TURBO_STEPS} steps, mu={TURBO_MU:g}."
            )


def assert_krea2_only(model_id: str, transformer: str | None = None) -> None:
    _refuse(str(model_id), role="model_id")
    if transformer:
        _refuse(str(transformer), role="transformer")


def assert_krea2_skeleton(skeleton: str) -> None:
    text = str(skeleton)
    lowered = text.lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                "Krea2-turbo-bbox skeleton is krea/Krea-2-Raw "
                f"(VAE + Qwen3-VL); refused foreign backend {name!r} "
                f"in skeleton {text!r}"
            )
    if Path(text).exists():
        return
    if "krea-2-raw" in lowered:
        return
    raise ValueError(
        "Krea2-turbo-bbox skeleton must be krea/Krea-2-Raw "
        "(VAE, text encoder, tokenizer, scheduler). "
        f"Refused {text!r}."
    )


def resolve_card(
    sample_steps: int | None,
    sample_guidance: float | None,
    mu: float | None = None,
) -> dict[str, float | int | str | bool]:
    return {
        "variant": "turbo_bbox",
        "sample_steps": int(TURBO_STEPS if sample_steps is None else sample_steps),
        "sample_guidance": float(
            TURBO_GUIDANCE if sample_guidance is None else sample_guidance
        ),
        "mu": float(TURBO_MU if mu is None else mu),
        "is_distilled": True,
        "skeleton": SKELETON_MODEL,
    }


def _local_safetensors(path: Path) -> Path | None:
    if path.suffix == ".safetensors" and path.is_file():
        return path.resolve()
    return None


def _local_transformer_subfolder(path: Path, subfolder: str) -> str | None:
    if (path / "config.json").is_file() and not (path / "model_index.json").is_file():
        return ""
    nested = path / subfolder
    if subfolder and nested.is_dir() and (nested / "config.json").is_file():
        return subfolder
    if (path / "transformer" / "config.json").is_file():
        return "transformer"
    return None


def _plan(**fields: object) -> dict[str, object]:
    fields.update(
        {
            "mu": float(TURBO_MU),
            "sample_steps": int(TURBO_STEPS),
            "sample_guidance": float(TURBO_GUIDANCE),
            "is_distilled": True,
            "comfy_filename": COMFY_FILENAME,
            "epoch": EPOCH,
        }
    )
    return fields


def resolve_source(
    model_id: str,
    *,
    subfolder: str | None = None,
    transformer: str | None = None,
    skeleton: str | None = None,
) -> dict[str, object]:
    """Hub subfolder, local diffusers dir, or Comfy safetensors. No download."""
    sub = str(subfolder or TRANSFORMER_SUBFOLDER).strip()
    skel = str(skeleton or SKELETON_MODEL)
    chosen = str(transformer).strip() if transformer else str(model_id)
    path = Path(chosen)
    weights = _local_safetensors(path)
    if weights is not None:
        return _plan(
            source="comfy_safetensors",
            model_id=str(model_id),
            transformer_path=str(weights),
            subfolder=None,
            repo_id=None,
            skeleton=skel,
        )
    if path.is_dir():
        local_sub = _local_transformer_subfolder(path, sub)
        if local_sub is None:
            raise ValueError(
                f"local path {path} has no Krea2 transformer config.json "
                f"(looked at the directory, {sub!r}, and transformer/)"
            )
        return _plan(
            source="local_transformer",
            model_id=str(model_id),
            transformer_path=str(path.resolve()),
            subfolder=local_sub,
            repo_id=None,
            skeleton=skel,
        )
    if transformer:
        raise ValueError(
            f"--transformer {transformer!r} is not a local .safetensors file "
            "or diffusers directory"
        )
    if not sub:
        raise ValueError("hub krea2-turbo-bbox load needs a transformer subfolder")
    return _plan(
        source="hub_subfolder",
        model_id=str(model_id),
        transformer_path=None,
        subfolder=sub,
        repo_id=str(model_id),
        skeleton=skel,
    )


def live_train_card(
    *,
    name: str = "smile-krea2-bbox",
    prompts_file: str = DEFAULT_PROMPTS,
    model_id: str = MODEL_ID,
    subfolder: str = TRANSFORMER_SUBFOLDER,
    skeleton: str = SKELETON_MODEL,
    rank: int = RANK,
    resolution: int = RESOLUTION,
    sample_steps: int = TURBO_STEPS,
    sample_guidance: float = TURBO_GUIDANCE,
    mu: float = TURBO_MU,
    hold_weight: float = HOLD_WEIGHT,
    lora_targets: str = LORA_TARGETS_DEFAULT,
) -> dict[str, object]:
    return {
        "backend": "krea2_turbo_bbox",
        "name": name,
        "prompts_file": prompts_file,
        "model_id": model_id,
        "transformer_subfolder": subfolder,
        "skeleton": skeleton,
        "comfy_filename": COMFY_FILENAME,
        "epoch": EPOCH,
        "load": (
            "Krea2Transformer2DModel.from_pretrained(model_id, subfolder=...) "
            "into Krea2Pipeline.from_pretrained(skeleton, transformer=tf)"
        ),
        "rank": int(rank),
        "resolution": int(resolution),
        "lora_targets": lora_targets,
        "sample_steps": int(sample_steps),
        "sample_guidance": float(sample_guidance),
        "mu": float(mu),
        "is_distilled": True,
        "hold_weight": float(hold_weight),
        "control_prompt": CONTROL_PROMPT,
        "uni": {
            "plus": "v(pos) at CFG 0 (not Raw CFG 4.5)",
            "zero": "v(neu)",
            "minus": "canary only",
            "hold": "unused tokens to encode(neu); concept words are not held",
            "captions": "bare (attributes are not prefixed)",
            "control_prompt": "verify only, never a teacher",
        },
        "not_raw_card": {
            "raw_cfg": RAW_GUIDANCE,
            "raw_steps": RAW_STEPS,
            "raw_model": SKELETON_MODEL,
        },
        "comfy_cfg_note": (
            "Comfy's Krea-2 Turbo template says CFG 1.0, which is guidance "
            "off in that UI. This trainer's diffusers convention is "
            "guidance_scale 0 (v = v(cond)). Do not pass 4.5."
        ),
        "age_hold_default": AGE_HOLD_WEIGHT,
        "smile_hold": HOLD_WEIGHT,
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
        "non_goals": (
            "vendored weights",
            "Music 3 default changes",
            "GPU train in CI",
            "particle-sliders checkout",
        ),
    }


def live_train_command(
    *,
    name: str = "smile-krea2-bbox",
    prompts_file: str = DEFAULT_PROMPTS,
    model_id: str = MODEL_ID,
    subfolder: str = TRANSFORMER_SUBFOLDER,
    skeleton: str = SKELETON_MODEL,
    rank: int = RANK,
    resolution: int = RESOLUTION,
    sample_steps: int = TURBO_STEPS,
    sample_guidance: float = TURBO_GUIDANCE,
    mu: float = TURBO_MU,
    hold_weight: float = HOLD_WEIGHT,
    lora_targets: str = LORA_TARGETS_DEFAULT,
    save_dir: str | None = None,
) -> str:
    dest = save_dir or f"models/{name}"
    guidance = f"{float(sample_guidance):g}"
    return (
        "CUDA_VISIBLE_DEVICES=0 python scripts/train_krea2.py \\\n"
        f"  --name {name} \\\n"
        f"  --prompts_file {prompts_file} \\\n"
        f"  --model_id {model_id} --allow_hub \\\n"
        f"  --transformer_subfolder {subfolder} \\\n"
        f"  --skeleton_model {skeleton} \\\n"
        f"  --lora_targets {lora_targets} --rank {int(rank)} "
        f"--resolution {int(resolution)} \\\n"
        f"  --sample_steps {int(sample_steps)} --sample_guidance {guidance} "
        f"--mu {float(mu):g} \\\n"
        f"  --hold_weight {float(hold_weight):g} --steps 800 --lr 1e-4 "
        "--seed 7 --device 0 \\\n"
        f"  --save_dir {dest}"
    )
