# Krea2 Particle Sliders

Product repo for particle / concept sliders on **[jimmycarter/krea2-turbo-bbox](https://huggingface.co/jimmycarter/krea2-turbo-bbox)**, the layout-control finetune of Krea-2 Turbo. The GitHub repo is **[krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders)** (renamed from `krea2-concept-sliders`), matching [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders).

This repo **is** the product. It owns the model pin, the train and infer entrypoints, the UNI prompt cards, and the Comfy node. It does not wrap a sibling checkout of [particle-sliders](https://github.com/HyperGAN/particle-sliders). An early draft of the trainer lived on particle-sliders #131; that ownership moved here, and #131 can be closed.

This is a **scaffold**. It does not ship slider weights, a finished GPU train, or a calibrated Comfy strength.

## Base model

| | |
|---|---|
| Hub | [`jimmycarter/krea2-turbo-bbox`](https://huggingface.co/jimmycarter/krea2-turbo-bbox) |
| What is in that repo | Transformer only |
| Default subfolder | `epoch-14-step-73184/transformer` (card label: epoch-14-step-73184 raw) |
| VAE, text encoder, tokenizer, scheduler | `krea/Krea-2-Raw` (`--skeleton_model`) |
| Comfy single file | `krea2-bbox-turbo-comfy-latest.safetensors` |
| Lock | [`configs/krea2/model.lock.json`](configs/krea2/model.lock.json) |

The finetune is `turbo_epoch = epoch_checkpoint + (krea/Krea-2-Turbo - krea/Krea-2-Raw)`. It was trained for **layout control**: panels, speech bubbles inside those panels, character identity across panels, and text placed where it was asked for. About 90% of that training used the grounding DSL in the Hub [PROMPTING.md](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md). Plain prose still works. A short local pointer is [PROMPTING.md](PROMPTING.md).

Stock Krea Raw sliders (28 steps, CFG 4.5) are a different card. This product trains and samples the distilled transformer: **8 steps, guidance 0, mu=1.15**.

## Turbo sample recipe

These weights are distilled. Sample in 8 steps. `is_distilled` is a **pipeline** flag (`model_index.json`), not a field on the transformer, so a transformer-only upload cannot bake it in. `krea2/live.py` sets it when a live pipeline is built.

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

Page size on the Hub card is about 1536×1536 of area, both sides a multiple of 16. Comics in that training set are about 1.4 tall per 1 wide, so `1296×1824` is the card's comic example. Starter train yamls stay at **512**. That is the UNI recipe default, not a claim that this finetune was tuned at 512.

## Train and infer

Clone this repo only. A particle-sliders checkout is not required.

```bash
python -m pip install -r requirements.txt
python scripts/train_krea2.py --help
python scripts/infer_krea2.py --help
python scripts/train_krea2.py --dummy
pytest -q
```

`--dummy` runs the in-repo CPU UNI loop (2 steps, tiny PNGs, no Hub download). A run without `--dummy` is refused before any download. The live loader is [`krea2/live.py`](krea2/live.py): `Krea2Transformer2DModel` from the epoch subfolder, dropped into `Krea2Pipeline` from `krea/Krea-2-Raw`. Guide: [docs/krea2-turbo-bbox-slider.md](docs/krea2-turbo-bbox-slider.md). Reproduction outline: [REPRODUCE.md](REPRODUCE.md).

Infer is the same code with `--load_te_lora`, which skips the train loop and writes the smile-first grid:

```bash
python scripts/infer_krea2.py --dummy --load_te_lora models/smile-krea2-bbox_lora
```

## Starter concepts

UNI cards: bare captions, `attributes` for unused-token bookkeeping (not prefixed onto the caption), a canary `negative` that is not a teacher, and the fruit-bowl control prompt.

| Card | Plus | Files |
|---|---|---|
| Smile (trainer default) | closed mouth → readable smile, plus one grounded row | [`configs/krea2/prompts-smile.yaml`](configs/krea2/prompts-smile.yaml) |
| Expression | closed mouth → readable smile, comic panel wording | [`configs/krea2/prompts-expression.yaml`](configs/krea2/prompts-expression.yaml) |
| Lighting | flat light → directional cel / rim | [`configs/krea2/prompts-lighting.yaml`](configs/krea2/prompts-lighting.yaml) |
| Panel clarity | crowded gutters → clean borders and bubbles inside panels | [`configs/krea2/prompts-panel-clarity.yaml`](configs/krea2/prompts-panel-clarity.yaml) |

Inference prompts may use the grounding DSL. The trainer rows stay bare UNI captions, including the one grounded smile row, so attributes are not prefixed into the boxes.

Defaults: `--lm_target v`, `--lora_targets dit`, `--hold_weight 0.1`, rank 16. `--lm_target embed` forces `--lora_targets te` and still samples at guidance 0. No schedule here is a published result.

## ComfyUI

Load `krea2-bbox-turbo-comfy-latest.safetensors` with the stock Krea-2 text encoder and VAE from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2). The custom node is `comfy_krea2.py` (`NTC/Krea2`, **Krea2 Turbo-BBox LoRA**). Details: [COMFYUI.md](COMFYUI.md).

## Formulation

Model integration and these recipes belong in this product. Toy gates and formulation search belong in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). ParticleGAN is the core primitive, not a place to park product experiments. Short note: [FORMULATION.md](FORMULATION.md).

## License

Code in this repository is [MIT](LICENSE). The Krea-2 weights and the bbox finetune are not included and are not MIT. See [NOTICE](NOTICE).
