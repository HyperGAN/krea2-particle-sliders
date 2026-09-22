# ComfyUI: Krea2 turbo-bbox

Use the bbox turbo transformer with the **stock** Krea-2 text encoder and VAE. This repo does not vendor the diffusion weights (~26 GB), those encoders, or a trained LoRA.

Install this repository as a custom node and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HyperGAN/krea2-particle-sliders.git
python -m pip install -r krea2-particle-sliders/requirements.txt
```

The GitHub repo is `krea2-particle-sliders` (renamed from `krea2-concept-sliders`). Use the Python that belongs to ComfyUI. The node module is `comfy_krea2.py`, registered from `__init__.py`.

## Models

1. Download [`krea2-bbox-turbo-comfy-latest.safetensors`](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/krea2-bbox-turbo-comfy-latest.safetensors) from `jimmycarter/krea2-turbo-bbox` into `ComfyUI/models/diffusion_models/`.

   The Hub sidecar `krea2-bbox-turbo-comfy-latest.json` records the epoch packed into that file. At the time of this scaffold it is **epoch-14-step-73184 (raw)**, `source_subfolder` `epoch-14-step-73184/transformer`. The safetensors file is replaced in place when a newer epoch is published, so re-check the sidecar if the image behavior changes under the same filename.

2. Text encoder and VAE are unchanged from stock Krea-2. Take them from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2):

   - `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
   - `vae/qwen_image_vae.safetensors`

## Graph

Start from the built-in **Text to Image (Krea-2 Turbo)** template and swap this diffusion file in where the template loads `krea2_turbo_*.safetensors`.

```text
Load Diffusion Model (krea2-bbox-turbo-comfy-latest.safetensors)
  → Krea2 Turbo-BBox LoRA (ntc-ai)   # optional, once a LoRA exists
  → Krea-2 Turbo sampler
```

The node category is **NTC/Krea2**. Strength 0 returns the cloned model with no adapter. Strength must be between 0 and 5. The node checks that the diffusion class name contains `krea` and that the chosen file is a safetensors LoRA (it looks for `lora_A` / `lora_B` style keys). It does not load Anima particle files. It does not merge those tensors into the Krea projections yet: there is no trained file in this repo to test that merge against. The node is the place that merge will plug in.

These are distilled weights. Keep the turbo settings from the Hub card:

- **8 steps**
- **CFG 1.0**

CFG 1.0 in Comfy is the neutral scale. The diffusers equivalent is `guidance_scale=0.0` plus `mu=1.15` when the pipeline does not already treat the checkpoint as distilled. Do not run the Raw card (28 steps, CFG 4.5) on this file.

Grounded prompts and plain prose both go in the same text box. There is no extra node for the grounding DSL. See [PROMPTING.md](PROMPTING.md). A comic page is happier near `1296×1824` (about 1536² of area, both sides a multiple of 16) than as a square.

## Where adapters come from

Nothing in this repo is a trained LoRA yet. `python scripts/train_krea2.py --dummy` writes a CPU sidecar, not a Comfy file. When a DiT LoRA exists, put it in `ComfyUI/models/loras/` and select it on **Krea2 Turbo-BBox LoRA**. CLIP strength does not apply: this node touches the diffusion model only.

There is no calibrated strength 1.0, because no slider has been trained on a GPU and scored. Strength 0 leaves the bbox turbo checkpoint unchanged.
