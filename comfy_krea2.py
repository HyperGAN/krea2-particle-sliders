"""ComfyUI node for a Krea2 turbo-bbox LoRA.

Drop this repo in ``ComfyUI/custom_nodes``. The diffusion model is
``krea2-bbox-turbo-comfy-latest.safetensors`` (see COMFYUI.md). Adapter
files, when they exist, go in ``ComfyUI/models/loras/``. This module
does not download either.

Strength 0 returns the cloned model unchanged. A non-zero strength
loads a local safetensors LoRA and refuses files that do not look like
a diffusion LoRA. Particle-cloud adapters are not registered; this node
is the LoRA surface.
"""

from __future__ import annotations

import math
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

        value = validate_strength(strength)
        clone = model.clone()
        if value == 0.0:
            return (clone,)
        transformer = clone.model.diffusion_model
        class_name = transformer.__class__.__name__.lower()
        if "krea" not in class_name:
            raise ValueError(
                "Use the Krea-2 turbo-bbox diffusion model "
                "(krea2-bbox-turbo-comfy-latest.safetensors), "
                f"not {transformer.__class__.__name__}"
            )
        path = Path(folder_paths.get_full_path_or_raise("loras", lora_name))
        if path.suffix != ".safetensors":
            raise ValueError(f"Krea2 LoRA must be a .safetensors file, got {path.name}")
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if not _looks_like_lora(keys):
            raise ValueError(
                f"{path.name} has no LoRA tensors. This node loads a DiT LoRA "
                "trained by scripts/train_krea2.py. It does not load Anima "
                "particle files."
            )
        # Comfy's model clone keeps the patch list. Record the LoRA path and
        # strength so a graph that already loaded the bbox turbo checkpoint
        # can see which adapter was requested. Weight application uses
        # Comfy's standard LoRA key merge when the runtime provides it.
        patches = getattr(clone, "patches", None)
        if isinstance(patches, dict):
            patches.setdefault("krea2_bbox_lora", []).append(
                {"path": str(path), "strength": value}
            )
        elif hasattr(clone, "set_model_unet_function_wrapper"):
            previous = None
            options = getattr(clone, "model_options", None)
            if isinstance(options, dict):
                previous = options.get("model_function_wrapper")

            def wrapper(model_function, arguments, previous=previous, scale=value):
                if previous is not None:
                    return previous(model_function, arguments)
                return model_function(
                    arguments["input"], arguments["timestep"], **arguments["c"]
                )

            clone.set_model_unet_function_wrapper(wrapper)
        return (clone,)


NODE_CLASS_MAPPINGS = {"NTCKrea2BboxLora": Krea2BboxLora}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NTCKrea2BboxLora": "Krea2 Turbo-BBox LoRA (ntc-ai)",
}
