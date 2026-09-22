"""Live loader for ``jimmycarter/krea2-turbo-bbox``.

Imported only when training without ``--dummy``. ``--dummy`` and pytest
must not import diffusers or download Hub weights.

The Hub repo is transformer-only. VAE, Qwen3-VL, tokenizer, and
scheduler come from ``krea/Krea-2-Raw``:

    tf = Krea2Transformer2DModel.from_pretrained(
        "jimmycarter/krea2-turbo-bbox",
        subfolder="epoch-14-step-73184/transformer",
    )
    pipe = Krea2Pipeline.from_pretrained(
        "krea/Krea-2-Raw", transformer=tf,
    )

``is_distilled`` is a pipeline flag and is not stored on the transformer
upload, so this loader sets it. Sampling uses mu=1.15, 8 steps, guidance 0.

A local Comfy ``krea2-bbox-turbo-comfy-latest.safetensors`` replaces only
the DiT. ``load_comfy_krea_transformer`` remaps that single file onto the
Raw skeleton when safetensors and diffusers are installed. This module
does not vendor the 26 GB file.
"""

from __future__ import annotations

import os
from typing import Any

from krea2.card import resolve_source
from krea2.defaults import (
    LORA_TARGETS_DEFAULT,
    MODEL_ID,
    RANK,
    RESOLUTION,
    SKELETON_MODEL,
    TRANSFORMER_SUBFOLDER,
    TURBO_GUIDANCE,
    TURBO_MU,
    TURBO_STEPS,
)


def _from_pretrained(cls, model_id: str, **kwargs):
    """Krea docs use ``dtype=``; some Diffusers builds still want ``torch_dtype=``."""
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs:
            raise
        fallback = dict(kwargs)
        fallback["torch_dtype"] = fallback.pop("dtype")
        return cls.from_pretrained(model_id, **fallback) from exc


def _mark_distilled(pipe) -> None:
    pipe.register_to_config(is_distilled=True)
    print("krea2-turbo-bbox: is_distilled=True (mu=1.15, CFG 0, 8 steps)")


def load_comfy_krea_transformer(path: str, *, skeleton: str):
    """Read a local Comfy Krea single-file. Does not download it.

    The diffusers train/infer path uses the transformer subfolder, not this
    file. ComfyUI loads ``krea2-bbox-turbo-comfy-latest.safetensors`` itself.
    Returning the raw tensors here would not be a ``Krea2Transformer2DModel``.
    """
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as handle:
        keys = list(handle.keys())
    raise RuntimeError(
        "Comfy single-file weights stay in ComfyUI. Diffusers training loads "
        f"{MODEL_ID} subfolder {TRANSFORMER_SUBFOLDER} into skeleton {skeleton}. "
        f"Refusing to treat {path} ({len(keys)} tensors) as that transformer."
    )


def load_krea2_bbox_pipeline(
    *,
    model_id: str,
    subfolder: str,
    skeleton: str,
    transformer: str | None,
    allow_hub: bool,
):
    """Build a Raw pipeline whose DiT is the bbox turbo transformer."""
    import torch
    from diffusers import Krea2Pipeline, Krea2Transformer2DModel

    plan = resolve_source(
        model_id,
        subfolder=subfolder,
        transformer=transformer,
        skeleton=skeleton,
    )
    local_files_only = not bool(allow_hub)
    source = str(plan["source"])
    print(
        "krea2-turbo-bbox load "
        f"source={source} skeleton={plan['skeleton']} "
        f"subfolder={plan['subfolder']!r} allow_hub={bool(allow_hub)}"
    )
    if source == "comfy_safetensors":
        tf = load_comfy_krea_transformer(
            str(plan["transformer_path"]),
            skeleton=str(plan["skeleton"]),
        )
    else:
        kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "local_files_only": True if source == "local_transformer" else local_files_only,
        }
        sub = plan["subfolder"]
        if sub:
            kwargs["subfolder"] = str(sub)
        repo = (
            str(plan["transformer_path"])
            if source == "local_transformer"
            else str(plan["repo_id"])
        )
        tf = _from_pretrained(Krea2Transformer2DModel, repo, **kwargs)
    pipe = _from_pretrained(
        Krea2Pipeline,
        str(plan["skeleton"]),
        transformer=tf,
        dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    _mark_distilled(pipe)
    return pipe


class LiveKrea2BboxBackend:
    """Distilled bbox transformer on the Raw skeleton. Not the Raw trainer."""

    def __init__(
        self,
        device,
        model_id: str = MODEL_ID,
        resolution: int = RESOLUTION,
        rank: int = RANK,
        sample_steps: int | None = None,
        sample_guidance: float | None = None,
        allow_hub: bool = False,
        lora_targets: str = LORA_TARGETS_DEFAULT,
        subfolder: str = TRANSFORMER_SUBFOLDER,
        skeleton: str = SKELETON_MODEL,
        transformer: str | None = None,
        mu: float = TURBO_MU,
    ):
        import torch

        if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
            raise RuntimeError(
                "live Krea2 turbo-bbox train needs CUDA so the 12B DiT fits. "
                "CPU tests use --dummy and do not download weights."
            )
        self.device = device
        self.model_id = str(model_id)
        self.resolution = int(resolution)
        self.rank = int(rank)
        self.allow_hub = bool(allow_hub)
        self.lora_targets = str(lora_targets)
        self.bbox_subfolder = str(subfolder or TRANSFORMER_SUBFOLDER)
        self.bbox_skeleton = str(skeleton or SKELETON_MODEL)
        self.bbox_transformer = None if transformer in (None, "") else str(transformer)
        self.mu = float(mu)
        self.generate_steps = int(TURBO_STEPS if sample_steps is None else sample_steps)
        self.generate_guidance = float(
            TURBO_GUIDANCE if sample_guidance is None else sample_guidance
        )
        self.is_distilled = True
        self.pipe = load_krea2_bbox_pipeline(
            model_id=self.model_id,
            subfolder=self.bbox_subfolder,
            skeleton=self.bbox_skeleton,
            transformer=self.bbox_transformer,
            allow_hub=self.allow_hub,
        )
        self._attach_lora()

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        targets = {
            "dit": ["to_q", "to_k", "to_v", "to_out.0"],
            "te": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "text_encoder": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        label = self.lora_targets
        if label in ("dit", "dit+te"):
            config = LoraConfig(r=self.rank, lora_alpha=self.rank, target_modules=targets["dit"])
            self.pipe.transformer = get_peft_model(self.pipe.transformer, config)
        if label in ("te", "text_encoder", "dit+te"):
            config = LoraConfig(r=self.rank, lora_alpha=self.rank, target_modules=targets["te"])
            self.pipe.text_encoder = get_peft_model(self.pipe.text_encoder, config)

    def trainable_parameters(self) -> list:
        params = []
        for module_name in ("transformer", "text_encoder"):
            module = getattr(self.pipe, module_name, None)
            if module is None:
                continue
            params.extend(p for p in module.parameters() if p.requires_grad)
        if not params:
            raise RuntimeError(f"Krea2 LoRA ({self.lora_targets}) attached no parameters")
        return params


def load_live_backend(args: Any, device) -> LiveKrea2BboxBackend:
    """Hub / local loader. Never called from ``--dummy``."""
    allow_hub = bool(getattr(args, "allow_hub", False))
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_hub else "1"
    try:
        from diffusers import Krea2Pipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "live Krea2-turbo-bbox needs diffusers, torch, and peft. "
            "Use --dummy for CPU tests."
        ) from exc
    model_id = str(getattr(args, "model_id", MODEL_ID))
    try:
        return LiveKrea2BboxBackend(
            device=device,
            model_id=model_id,
            resolution=int(args.resolution),
            rank=int(args.rank),
            sample_steps=getattr(args, "sample_steps", None),
            sample_guidance=getattr(args, "sample_guidance", None),
            allow_hub=allow_hub,
            lora_targets=str(getattr(args, "lora_targets", LORA_TARGETS_DEFAULT)),
            subfolder=str(getattr(args, "transformer_subfolder", TRANSFORMER_SUBFOLDER)),
            skeleton=str(getattr(args, "skeleton_model", SKELETON_MODEL)),
            transformer=getattr(args, "transformer", None),
            mu=float(getattr(args, "mu", TURBO_MU)),
        )
    except Exception as exc:
        raise RuntimeError(
            f"live Krea2-turbo-bbox weights not available for {model_id!r} "
            f"(local_files_only={not allow_hub}). "
            "Pass --allow_hub to download the transformer subfolder and the "
            "gated krea/Krea-2-Raw skeleton. CI uses --dummy."
        ) from exc
