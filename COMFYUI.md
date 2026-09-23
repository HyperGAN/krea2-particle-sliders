# ComfyUI: Krea2 turbo-bbox sliders

Download Final Boss or Eldritch from
[ntc-ai/krea2-particle-sliders](https://huggingface.co/ntc-ai/krea2-particle-sliders).
Both the rank-16 originals and rank-8 distills have calibrated alpha: start at
**MODEL strength 1**, **CLIP strength 0**. This repository does not vendor the
base model, encoders, or adapter weights.

## Models

1. Download [`krea2-bbox-turbo-comfy-latest.safetensors`](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/ec7aa643a4da7e56a08c1778e03e837a2e57a94b/krea2-bbox-turbo-comfy-latest.safetensors)
   into `ComfyUI/models/diffusion_models/`. This release uses the pinned
   `epoch-14-step-73184/transformer` checkpoint. The file named `latest` on the
   upstream main branch may change; use the pinned link for reproduction.
2. Use the stock text encoder and VAE from
   [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2):
   `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` and
   `vae/qwen_image_vae.safetensors`.
3. Put a **ComfyUI export** from the release's `weights/comfyui/` or
   `distilled/comfyui/` folder in `ComfyUI/models/loras/`. The two formats use
   the same filename, so keep originals and distills in separate subfolders.

## Graph and settings

Start from the built-in **Text to Image (Krea-2 Turbo)** template. Swap in the
bbox diffusion model and insert standard **Load LoRA** on its MODEL connection:

```text
Load Diffusion Model → Load LoRA → Krea-2 Turbo sampler
```

Set MODEL strength to **1** and CLIP strength to **0**. Use **8 steps**, **CFG 1.0**
and the template's turbo timestep shift. In Diffusers these correspond to
`guidance_scale=0.0`, `mu=1.15`, and `pipe.register_to_config(is_distilled=True)`.
The adapter distillation reduces rank; it keeps the same eight denoising steps.
Strength 0 disables the effect. The alpha is already inside each export, so
Eldritch needs no additional 1.5 multiplier.

Grounded prompts and ordinary prose use the same text box. See
[PROMPTING.md](PROMPTING.md) for the bbox DSL. Release comparisons use 768×768,
seed 42, and identical prompts for Original / Distill / Off.

## Optional custom node

The standard node works with the ComfyUI exports. To also load native Diffusers
exports with their embedded alpha metadata, install this repository:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HyperGAN/krea2-particle-sliders.git
```

Restart ComfyUI. `comfy_krea2.py`, registered by `__init__.py`, adds
**Krea2 Turbo-BBox LoRA (ntc-ai)** under **NTC/Krea2**. Connect its MODEL output
to the sampler and select the adapter. It accepts strengths from 0 to 5,
clones the incoming model, checks the Krea model type, and refuses partial
projection matches. At strength 0 it returns the clone without new patches.
It modifies the diffusion model only.

The release audit checks all 128 attention projections against the real ComfyUI
Krea topology, including text fusion. Native and ComfyUI factors match exactly,
and patched weight deltas include alpha/rank. This is a CPU integration check;
the published images were generated through Diffusers, not a full ComfyUI GPU
render. See the release's `validation/comfyui.json` for the tested revision.
