"""Pinned Krea2 turbo-bbox defaults. Importing this module hits no network."""

from __future__ import annotations

MODEL_ID = "jimmycarter/krea2-turbo-bbox"
TRANSFORMER_SUBFOLDER = "epoch-14-step-73184/transformer"
EPOCH = "epoch-14-step-73184"
PIPELINE_ID = "krea/Krea-2-Raw"
SKELETON_MODEL = PIPELINE_ID
COMFY_FILENAME = "krea2-bbox-turbo-comfy-latest.safetensors"
COMFY_TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
COMFY_VAE = "qwen_image_vae.safetensors"
COMFY_ORG_REPO = "Comfy-Org/Krea-2"

TURBO_STEPS = 8
TURBO_GUIDANCE = 0.0
TURBO_MU = 1.15
COMFY_CFG = 1.0
RAW_STEPS = 28
RAW_GUIDANCE = 4.5

RANK = 16
ALPHA = 16.0
RESOLUTION = 512
CONTROL_PROMPT = "a bowl of fruit on a table"
HOLD_WEIGHT = 0.1
AGE_HOLD_WEIGHT = 1.0
SAMPLE_SCALES = (0.0, 0.25, 0.5, 1.0)

LM_TARGET_DEFAULT = "v"
LORA_TARGETS_DEFAULT = "dit"
RECIPE_DEFAULT = "uni"

PRODUCT_NAME = "krea2-particle-sliders"
HUB_CARD = "https://huggingface.co/jimmycarter/krea2-turbo-bbox"
HUB_PROMPTING = (
    "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md"
)
HUB_LICENSE = (
    "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/LICENSE.md"
)
CONCEPTMOD_REPO = "https://github.com/HyperGAN/conceptmod"
ANIMA_PRODUCT = "https://github.com/HyperGAN/anima-particle-sliders"

# ID fragments that mean this finetune, not stock Raw or Krea-2-Turbo.
BBOX_ID_MARKERS = (
    "krea2-turbo-bbox",
    "krea2_turbo_bbox",
    "krea2-bbox",
    "krea2_bbox",
    "turbo-bbox",
    "turbo_bbox",
)
