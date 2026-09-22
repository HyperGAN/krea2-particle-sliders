# Krea-2 Concept Sliders

Product repo for particle / concept sliders on **[jimmycarter/krea2-turbo-bbox](https://huggingface.co/jimmycarter/krea2-turbo-bbox)**, the layout-control finetune of Krea-2 Turbo. Same split as [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders) and [supra-concept-sliders](https://github.com/HyperGAN/supra-concept-sliders): this repo owns the product surface (docs, Comfy notes, starter recipes). Train and infer live in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders).

This is a **scaffold**. It does not ship slider weights, a Comfy node, or a finished GPU slider.

## Base model

| | |
|---|---|
| Hub | [`jimmycarter/krea2-turbo-bbox`](https://huggingface.co/jimmycarter/krea2-turbo-bbox) |
| What is in that repo | Transformer only |
| Default subfolder | `epoch-14-step-73184/transformer` (card label: epoch-14-step-73184 raw). Confirmed on the Hub: that epoch directory contains `transformer` and no newer epoch is published. |
| VAE, text encoder, tokenizer, scheduler | `krea/Krea-2-Raw` |
| Comfy single file | `krea2-bbox-turbo-comfy-latest.safetensors` (same epoch, replaced in place when a newer one is uploaded) |
| Lock | [`configs/krea2/model.lock.json`](configs/krea2/model.lock.json) |

The finetune is `turbo_epoch = epoch_checkpoint + (krea/Krea-2-Turbo - krea/Krea-2-Raw)`. It was trained for **layout control**: panels, speech bubbles inside those panels, character identity across panels, and text placed where it was asked for. About 90% of that training used the grounding DSL in the Hub [PROMPTING.md](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md). Plain prose still works. A short local pointer is [PROMPTING.md](PROMPTING.md).

### How this differs from stock Krea Raw sliders

[particle-sliders](https://github.com/HyperGAN/particle-sliders) already has an opt-in Krea UNI trainer aimed at **stock** `krea/Krea-2-Raw`, with Turbo as the sample target:

- `conceptmod/textsliders/train_lora_krea.py`
- `conceptmod/textsliders/krea_live.py`
- [docs/krea-slider.md](https://github.com/HyperGAN/particle-sliders/blob/main/docs/krea-slider.md)

That live loader calls `Krea2Pipeline.from_pretrained(model_id)` (default `krea/Krea-2-Raw`). The bbox repo is a **transformer-only** upload, so that call does not load `epoch-14-step-73184/transformer`. `krea_looks_turbo()` would see the substring `turbo` in the repo id and pick an 8-step / CFG-0 sample card, but the weights underneath would still be whatever full pipeline `from_pretrained` returned. Do not point `train_lora_krea.py` at this product and treat it as the bbox finetune.

`train_lora_krea2` is **not** on particle-sliders `main` (no `conceptmod/textsliders/train_lora_krea2.py`). The Supra split is the pattern to follow: backend PR ([particle-sliders #130](https://github.com/HyperGAN/particle-sliders/pull/130), `train_lora_supra.py`) plus a thin product repo. Until that Krea-2 entrypoint exists, [`scripts/train_krea2.py`](scripts/train_krea2.py) and [`scripts/infer_krea2.py`](scripts/infer_krea2.py) exit 2 with the intended CLI. They do not fall back to the Raw trainer.

Music 3 and Anima defaults stay in their own trainers. Nothing here changes them.

## Turbo sample recipe

These weights are distilled. Sample in 8 steps. `is_distilled` is a **pipeline** flag (`model_index.json`), not a field on the transformer, so a transformer-only upload cannot bake it in.

Diffusers, matching the Hub card:

```python
import torch
from diffusers import Krea2Pipeline, Krea2Transformer2DModel

tf = Krea2Transformer2DModel.from_pretrained(
    "jimmycarter/krea2-turbo-bbox",
    subfolder="epoch-14-step-73184/transformer",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Raw",
    transformer=tf,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

image = pipe(
    "a photo of a cat",
    num_inference_steps=8,
    guidance_scale=0.0,
    mu=1.15,
).images[0]
```

| Runtime | Steps | Guidance |
|---|---|---|
| Diffusers | 8 | `guidance_scale=0.0`, and `mu=1.15` when the pipeline does not infer `is_distilled` |
| ComfyUI | 8 | CFG **1.0** (the card's turbo template; CFG 1.0 is the neutral scale) |

Page size on the Hub card is about 1536×1536 of area, both sides a multiple of 16. Comics in that training set are about 1.4 tall per 1 wide, so `1296×1824` is the card's comic example. Starter train yamls stay at **512**, the particle-sliders Krea UNI resolution, until `train_lora_krea2` measures something else. 512 is a recipe default, not a claim that this finetune was tuned at 512.

## Starter concepts

UNI cards in the same shape as particle-sliders `prompts-krea-happy.yaml`: bare captions, `attributes` for unused-token bookkeeping (not prefixed onto the caption), a canary `negative` that is not a teacher, and the fruit-bowl control prompt.

| Card | Plus | Files |
|---|---|---|
| Expression | closed mouth → readable smile | [`configs/krea2/prompts-expression.yaml`](configs/krea2/prompts-expression.yaml) |
| Lighting | flat light → directional cel / rim | [`configs/krea2/prompts-lighting.yaml`](configs/krea2/prompts-lighting.yaml) |
| Panel clarity | crowded gutters → clean borders and bubbles inside panels | [`configs/krea2/prompts-panel-clarity.yaml`](configs/krea2/prompts-panel-clarity.yaml) |

Inference prompts may use the grounding DSL. The trainer rows do not. That keeps the UNI pair format the backend already parses.

Intended train defaults, once the backend exists: `--recipe uni --lm_target v --lora_targets dit --hold_weight 0.1`, rank 16. Smile-krea on **Raw** later moved to a text-encoder embed target because DiT velocities there were almost identical. That measurement is not this transformer. Do not copy the smile-krea-v5 TE-only flags until someone measures the neu/plus gap on `epoch-14-step-73184/transformer`. Panel clarity is a spatial edit and is the one most likely to need the DiT.

No schedule here is a published result. Iterations (500), rank, and learning rate match the stock happy card so the yaml stays familiar.

## ComfyUI

Load `krea2-bbox-turbo-comfy-latest.safetensors` with the stock Krea-2 text encoder and VAE from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2). Swap that file into the built-in **Text to Image (Krea-2 Turbo)** template. LoRA and particle adapters plug in later, in front of the sampler. Details: [COMFYUI.md](COMFYUI.md).

## Reproduce

CPU checks in this repo do not download the Hub checkpoint. A GPU slider is not reproducible from this tree until weights and `train_lora_krea2` exist. Outline: [REPRODUCE.md](REPRODUCE.md).

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/train_krea2.py --dummy
```

The last command is expected to exit 2 today.

## Formulation

Model integration and these recipes belong here. Toy gates and formulation search belong in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). ParticleGAN is the core primitive, not a place to park this product's experiments. Short note: [FORMULATION.md](FORMULATION.md).

## License

Code in this repository is [MIT](LICENSE). The Krea-2 weights and the bbox finetune are not included and are not MIT. See [NOTICE](NOTICE).
