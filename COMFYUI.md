# ComfyUI: Krea-2 turbo-bbox

Use the bbox turbo transformer with the **stock** Krea-2 text encoder and VAE. This repo does not vendor the diffusion weights (~26 GB) or those encoders.

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
  → Load text encoder (qwen3vl_4b_fp8_scaled.safetensors)
  → Load VAE (qwen_image_vae.safetensors)
  → Krea-2 Turbo sampler
```

These are distilled weights. Keep the turbo settings from the Hub card:

- **8 steps**
- **CFG 1.0**

CFG 1.0 in Comfy is the neutral scale. The diffusers equivalent is `guidance_scale=0.0` plus `mu=1.15` when the pipeline does not already treat the checkpoint as distilled. Do not run the Raw card (28 steps, CFG 4.5) on this file.

Grounded prompts and plain prose both go in the same text box. There is no extra node for the grounding DSL. See [PROMPTING.md](PROMPTING.md). A comic page is happier near `1296×1824` (about 1536² of area, both sides a multiple of 16) than as a square.

## Where adapters plug in

Nothing in this repo is a trained LoRA or particle adapter. When `train_lora_krea2` writes one:

- Put the file in `ComfyUI/models/loras/`.
- Insert **Load LoRA** (MODEL strength, CLIP strength **0**) or **Load LoRA (Model Only)** between the diffusion-model load and the sampler. The starter recipe trains the diffusion transformer, not the Qwen text encoder. CLIP strength 0 keeps a DiT LoRA from being applied to the text encoder by accident.
- Strength 0 is off. There is no calibrated strength 1.0 yet, because no slider has been trained or scored.

If a later backend emits a nonlinear particle adapter (the Anima plugin pattern) rather than a plain LoRA, it will need its own Comfy node. This scaffold does not register one. Do not load an Anima particle file on this graph.

A text-encoder LoRA is not the default here. Stock smile-krea on Raw eventually trained Qwen3-VL only, after a measured velocity gap. That result is not a measurement on `epoch-14-step-73184/transformer`. If a future card is TE-only, say so on the sidecar and load it on the text encoder instead of the diffusion model.
