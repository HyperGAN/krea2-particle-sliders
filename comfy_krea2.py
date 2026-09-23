"""ComfyUI node for a Krea2 turbo-bbox LoRA.

Drop this repo in ``ComfyUI/custom_nodes``. The diffusion model is
``krea2-bbox-turbo-comfy-latest.safetensors`` (see COMFYUI.md). Adapter
files go in ``ComfyUI/models/loras/``. This module
does not download either.

Strength 0 returns the cloned model unchanged. A non-zero strength
loads a local safetensors LoRA and refuses files that do not look like
a diffusion LoRA. Particle-cloud adapters are not registered; this node
is the LoRA surface.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def validate_strength(strength: float) -> float:
    value = float(strength)
    if not math.isfinite(value) or not 0.0 <= value <= 5.0:
        raise ValueError("Strength must be between 0 and 5")
    return value


def _looks_like_lora(keys: list[str]) -> bool:
    if not keys:
        return False
    markers = ("lora_a", "lora_b", "lora_up", "lora_down")
    lowered = [key.lower() for key in keys]
    return any(any(marker in key for marker in markers) for key in lowered)


class Krea2BboxLora:
    """Apply a DiT LoRA on the Krea2 turbo-bbox diffusion model."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "NTC/Krea2"

    def load(self, model, lora_name, strength):
        import folder_paths
        import comfy.lora
        import comfy.model_base
        import torch

        value = validate_strength(strength)
        clone = model.clone()
        if value == 0.0:
            return (clone,)
        if not isinstance(clone.model, comfy.model_base.Krea2):
            raise ValueError(
                "Use the Krea-2 turbo-bbox diffusion model "
                "(krea2-bbox-turbo-comfy-latest.safetensors), "
                f"not {clone.model.__class__.__name__}"
            )
        path = Path(folder_paths.get_full_path_or_raise("loras", lora_name))
        if path.suffix != ".safetensors":
            raise ValueError(f"Krea2 LoRA must be a .safetensors file, got {path.name}")
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata() or {}
        if not _looks_like_lora(keys):
            raise ValueError(
                f"{path.name} has no LoRA tensors. This node loads a DiT LoRA "
                "trained by scripts/train_krea2.py. It does not load Anima "
                "particle files."
            )
        state = load_file(str(path))
        # Diffusers stores alpha in its safetensors metadata; Comfy uses tensors.
        config = json.loads(metadata.get('lora_adapter_metadata', '{}'))
        if config.get('transformer.alpha_pattern'):
            raise ValueError('Use the ComfyUI export for a per-projection alpha adapter')
        alpha = config.get('transformer.lora_alpha')
        if alpha is not None:
            if not math.isfinite(float(alpha)) or float(alpha) <= 0:
                raise ValueError('Invalid embedded LoRA alpha')
            for key in keys:
                if key.endswith('.lora_A.weight'):
                    state.setdefault(key.removesuffix('.lora_A.weight') + '.alpha', torch.tensor(float(alpha)))
        mapping = comfy.lora.model_lora_keys_unet(clone.model, {})
        patches = comfy.lora.load_lora(state, mapping)
        expected = sum(key.endswith(('.lora_A.weight', '.lora_down.weight')) for key in state)
        if not expected or len(patches) != expected:
            raise ValueError(f'Only {len(patches)}/{expected} Krea LoRA projections matched')
        applied = clone.add_patches(patches, value)
        if len(applied) != expected:
            raise ValueError(f'Only {len(applied)}/{expected} Krea LoRA patches applied')
        return (clone,)


NODE_CLASS_MAPPINGS = {"NTCKrea2BboxLora": Krea2BboxLora}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NTCKrea2BboxLora": "Krea2 Turbo-BBox LoRA (ntc-ai)",
}
