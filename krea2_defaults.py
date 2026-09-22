"""Pinned product defaults for Krea-2 turbo-bbox sliders.

Importing this module does not download Hub weights. The transformer
checkpoint stays on Hugging Face; this repo only names it.
"""

from __future__ import annotations

MODEL_ID = "jimmycarter/krea2-turbo-bbox"
TRANSFORMER_SUBFOLDER = "epoch-14-step-73184/transformer"
PIPELINE_ID = "krea/Krea-2-Raw"
COMFY_FILENAME = "krea2-bbox-turbo-comfy-latest.safetensors"
COMFY_TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
COMFY_VAE = "qwen_image_vae.safetensors"
COMFY_ORG_REPO = "Comfy-Org/Krea-2"

TURBO_STEPS = 8
TURBO_GUIDANCE = 0.0
TURBO_MU = 1.15
COMFY_CFG = 1.0

RANK = 16
ALPHA = 16.0
RESOLUTION = 512
CONTROL_PROMPT = "a bowl of fruit on a table"
HOLD_WEIGHT = 0.1

PARTICLE_SLIDERS_REPO = "https://github.com/HyperGAN/particle-sliders"
BACKEND_PR = "https://github.com/HyperGAN/particle-sliders/pull/131"
BACKEND_DOC = "docs/krea2-turbo-bbox-slider.md"
STOCK_TRAINER = "conceptmod/textsliders/train_lora_krea.py"
STOCK_LIVE = "conceptmod/textsliders/krea_live.py"
STOCK_DOC = "docs/krea-slider.md"
# Train and infer are the same entrypoint. Infer passes --load_te_lora,
# which skips the train loop and writes the sample grid.
INTENDED_TRAINER = "conceptmod/textsliders/train_lora_krea2.py"
INTENDED_INFER = INTENDED_TRAINER

HUB_CARD = "https://huggingface.co/jimmycarter/krea2-turbo-bbox"
HUB_PROMPTING = "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md"
HUB_LICENSE = "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/LICENSE.md"
CONCEPTMOD_REPO = "https://github.com/HyperGAN/conceptmod"
