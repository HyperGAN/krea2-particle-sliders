# Krea2 Turbo-BBox Particle Sliders

**Final Boss and Eldritch**, with original rank-16 LoRAs and compressed rank-8 distills.
Use **strength 1** for the calibrated effect. Same prompt, seed and sampler across each comparison.

Built on [jimmycarter/krea2-turbo-bbox](https://huggingface.co/jimmycarter/krea2-turbo-bbox),
`epoch-14-step-73184/transformer`. These are ordinary attention LoRAs. The originals
were trained directly as LoRAs; the distills compress those linear adapters.

## Samples

Each comparison shows **Off / On (Original)**, followed by **Off / Distill**.
On and Distill both use **strength 1**.
Click any image to open its full-resolution PNG.
All images below are AI-generated, 768 × 768, 8 steps, guidance 0, mu=1.15.
Each comparison states its seed and keeps it fixed across Original, Distill and Off.
They were rendered from the released files, with no external alpha multiplier.

### Final Boss

#### Robot in a rainy yard

A plain blue service robot gains angular shoulder and chest armor, heavier mechanical limbs and weathered panels. Same rainy factory yard, prompt and seed.

Seed **31415** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/final-boss-photo-rainy-yard-robot-seed-31415.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Final Boss — Robot in a rainy yard: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/off.png) | [![Final Boss — Robot in a rainy yard: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Final Boss — Robot in a rainy yard: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/off.png) | [![Final Boss — Robot in a rainy yard: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-rainy-yard-robot-seed-31415/distill.png) |

<details><summary>Exact prompt</summary>

```text
A full-body documentary photograph of an industrial robot standing outside after rain.
@35mm documentary photography, realistic wet metal, natural overcast daylight; Photograph
~A quiet factory yard with puddles, a brick wall and distant steel pipes.
o[230,80,770,950] A life-size humanoid service robot with a compact flat head, a single dark camera visor, plain faded blue metal panels and black mechanical joints, two arms and two legs, standing still with both hands lowered, feet firmly on the wet concrete.
```

</details>

<details><summary>Additional comparisons and unrelated fruit control</summary>

#### Heldout Bridge

Seed **42** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/final-boss-heldout-bridge.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Final Boss — Heldout Bridge: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/off.png) | [![Final Boss — Heldout Bridge: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Final Boss — Heldout Bridge: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/off.png) | [![Final Boss — Heldout Bridge: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/heldout-bridge/distill.png) |

<details><summary>Exact prompt</summary>

```text
A lone warrior guarding a volcanic bridge.
@dramatic lighting, detailed fantasy game art; Digital illustration
~A stone bridge above a glowing lava river.
pe:1[220,80,780,950] An adult warrior in steel armor, holding a sword, standing guard.
```

</details>

#### Knight

Seed **42** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/final-boss-knight.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Final Boss — Knight: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/off.png) | [![Final Boss — Knight: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Final Boss — Knight: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/off.png) | [![Final Boss — Knight: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/knight/distill.png) |

<details><summary>Exact prompt</summary>

```text
A full-body armored knight in a ruined cathedral.
@cinematic lighting, detailed game concept art; Digital illustration
~Ruined gothic arches and a cold stone floor.
pe:1[230,100,770,940] An adult knight in practical steel armor, plain helmet, holding a longsword, calm stance.
```

</details>

#### Street Photo

Seed **4242** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/final-boss-street-photo.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Final Boss — Street Photo: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/off.png) | [![Final Boss — Street Photo: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Final Boss — Street Photo: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/off.png) | [![Final Boss — Street Photo: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/street-photo/distill.png) |

<details><summary>Exact prompt</summary>

```text
A candid full-body street photograph of a commuter on a rainy night in Tokyo.
@35mm street photography, realistic skin texture, natural proportions, cinematic neon reflections; Photograph
~A narrow city street with small restaurants, wet asphalt, red and blue neon reflections, soft background bokeh and gentle rain.
pe:1[240,100,760,950] An adult man with short dark hair in a simple dark wool overcoat, gray sweater, jeans and ordinary leather shoes, holding a closed black umbrella at his side, standing casually and looking toward the camera.
```

</details>

#### Fruit Control

Seed **42** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/final-boss-fruit-control.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Final Boss — Fruit Control: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/off.png) | [![Final Boss — Fruit Control: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Final Boss — Fruit Control: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/off.png) | [![Final Boss — Fruit Control: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/fruit-control/distill.png) |

<details><summary>Exact prompt</summary>

```text
a bowl of fruit on a table
```

</details>

#### Workshop robot

Seed **2026** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/final-boss-photo-robot-seed-2026.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Final Boss — Workshop robot: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/off.png) | [![Final Boss — Workshop robot: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Final Boss — Workshop robot: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/off.png) | [![Final Boss — Workshop robot: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/final-boss/photo-robot-seed-2026/distill.png) |

<details><summary>Exact prompt</summary>

```text
A full-body photograph of a humanoid robot in an engineering workshop.
@industrial editorial photography, realistic metal and plastic, soft window light; Photograph
~A real workshop with concrete floors, workbenches and neatly arranged tools.
o[200,80,800,960] A life-size humanoid research robot with a simple rounded head, plain silver panels, exposed black joints and two ordinary arms, standing upright facing the camera.
```

</details>

</details>

### Eldritch

#### Heldout Bridge

Seed **42** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/eldritch-heldout-bridge.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Eldritch — Heldout Bridge: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/off.png) | [![Eldritch — Heldout Bridge: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Eldritch — Heldout Bridge: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/off.png) | [![Eldritch — Heldout Bridge: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/heldout-bridge/distill.png) |

<details><summary>Exact prompt</summary>

```text
A lone warrior guarding a volcanic bridge.
@dramatic lighting, detailed fantasy game art; Digital illustration
~A stone bridge above a glowing lava river.
pe:1[220,80,780,950] An adult warrior in steel armor, holding a sword, standing guard.
```

</details>

<details><summary>Additional comparisons and unrelated fruit control</summary>

#### Knight

Seed **42** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/eldritch-knight.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Eldritch — Knight: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/off.png) | [![Eldritch — Knight: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Eldritch — Knight: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/off.png) | [![Eldritch — Knight: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/knight/distill.png) |

<details><summary>Exact prompt</summary>

```text
A full-body armored knight in a ruined cathedral.
@cinematic lighting, detailed game concept art; Digital illustration
~Ruined gothic arches and a cold stone floor.
pe:1[230,100,770,940] An adult knight in practical steel armor, plain helmet, holding a longsword, calm stance.
```

</details>

#### Fruit Control

Seed **42** · [Side-by-side overview](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/assets/eldritch-fruit-control.jpg)

| Off | On (Original) · strength 1 |
| :---: | :---: |
| [![Eldritch — Fruit Control: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/off.png) | [![Eldritch — Fruit Control: On (Original)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/original.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/original.png) |

| Off | Distill · strength 1 |
| :---: | :---: |
| [![Eldritch — Fruit Control: Off](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/off.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/off.png) | [![Eldritch — Fruit Control: Distill](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/distill.png)](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/samples/eldritch/fruit-control/distill.png) |

<details><summary>Exact prompt</summary>

```text
a bowl of fruit on a table
```

</details>

</details>

The bridge prompt was excluded from the original six-pair training set, then used for
development comparisons. These examples are not a final-test benchmark. The featured
Final Boss photograph was visually selected from eight robot photo prompts at strength 1,
following an earlier eight-subject photo search. The Tokyo street photo and first workshop
robot remain available in the additional examples.
The [robot selection notes and all eight comparisons](https://huggingface.co/ntc-ai/krea2-particle-sliders/tree/main/evidence/preview-selection-robots-v3)
and [earlier photo search](https://huggingface.co/ntc-ai/krea2-particle-sliders/tree/main/evidence/preview-selection-v2)
record the curation.
The original
Eldritch effect emphasizes organic armor and curling appendages; extra eyes and facial
tentacles remain weak. The fruit control shows some rendering-style drift.

## Downloads

| Slider | Original · ComfyUI | Distill · ComfyUI | Original · Diffusers | Distill · Diffusers |
|---|---|---|---|---|
| Final Boss | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/weights/comfyui/krea2-final-boss-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/distilled/comfyui/krea2-final-boss-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/weights/native/krea2-final-boss-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/distilled/native/krea2-final-boss-unit-alpha.safetensors?download=true) |
| Eldritch | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/weights/comfyui/krea2-eldritch-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/distilled/comfyui/krea2-eldritch-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/weights/native/krea2-eldritch-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/distilled/native/krea2-eldritch-unit-alpha.safetensors?download=true) |

Original files are about 77 MB; distills are about 38 MB. Both contain 128 projection adapters.
[PEFT originals](https://huggingface.co/ntc-ai/krea2-particle-sliders/tree/main/weights/peft) · [PEFT distills](https://huggingface.co/ntc-ai/krea2-particle-sliders/tree/main/distilled/peft) ·
[Catalog](https://huggingface.co/ntc-ai/krea2-particle-sliders/blob/main/catalog.json) · [File hashes](https://huggingface.co/ntc-ai/krea2-particle-sliders/blob/main/release-manifest.json).

## ComfyUI

Use standard **Load LoRA** with a **ComfyUI export**, MODEL strength **1**, CLIP strength **0**.
No custom node is required. The optional `comfy_krea2.py` node, **Krea2 Turbo-BBox LoRA**
under **NTC/Krea2**, also handles these files and the native exports' alpha metadata.
[Plugin ZIP](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/comfyui/krea2-particle-sliders.zip?download=true) · [Setup](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/COMFYUI.md).

Use `krea2-bbox-turbo-comfy-latest.safetensors` with the stock text encoder and VAE from
[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2). Start from the Krea-2 Turbo
template: **8 steps**, **CFG 1.0**. The Diffusers equivalent is `guidance_scale=0.0`.
The bbox checkpoint uses the distilled timestep shift **mu=1.15**.

## Diffusers

Use a Diffusers build with `Krea2Pipeline` and Krea LoRA metadata support. The release
validation records the tested versions. After loading the bbox transformer with the
`krea/Krea-2-Raw` pipeline components:

```python
pipe.register_to_config(is_distilled=True)  # selects mu=1.15
pipe.load_lora_weights(
    "ntc-ai/krea2-particle-sliders",
    weight_name="weights/native/krea2-eldritch-unit-alpha.safetensors",
    adapter_name="eldritch",
)
pipe.set_adapters("eldritch", adapter_weights=1.0)
image = pipe(prompt, height=768, width=768, num_inference_steps=8,
             guidance_scale=0.0).images[0]
```

The alpha is inside the file. For direct `transformer.load_lora_adapter` calls, use
`use_safetensors=True` so the loader reads that metadata. PEFT users can load the
corresponding folder and its `adapter_config.json`.

Grounded captions use x-first `[x0,y0,x1,y1]` boxes on a 0–1000 grid, one element per
line, numeric character IDs, and panel-contained text. [Prompting guide](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/PROMPTING.md).
The actual model tokenizer checked the 507-token content budget before training.

## Distillation and alpha

| Slider | Original rank / alpha | Distill rank / alpha | Projection relative MSE | Full-edit relative MSE | Edit cosine |
|---|---:|---:|---:|---:|---:|
| Final Boss | 16 / 16 | 8 / 8 | 0.000006 | 0.002129 | 0.998935 |
| Eldritch | 16 / 24 | 8 / 12 | 0.000006 | 0.001236 | 0.999382 |

Distills fit output principal components on 480 training activations per projection.
Alpha selection uses 16 separate development states and complete denoiser edits. These
errors measure approximation, not image quality. Compare the images before choosing
a format. [Method and all candidate measurements](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/DISTILLATION.md).

The originals retain every learned matrix. Final Boss embeds alpha 16 at rank 16;
Eldritch embeds alpha 24 at rank 16, making released strength 1 equivalent to its
previous strength 1.5. Student alphas are selected separately. Both formats keep
the base's 8-step schedule; this distillation reduces adapter rank, not denoising steps.

## Training and source

Both originals used physical GPU 0, rank 16, 400 updates, learning rate 5e-5, 512px,
six paired captions, two cached trajectory seeds per pair, and preservation weight 0.1
every fifth update. The base and text encoder stayed frozen. Source, configurations,
validation and reproduction belong to [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders), following
the release layout of [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders).
The Krea originals are linear LoRAs; this is not the Anima nonlinear particle training recipe.

The GitHub repository does not ship slider weights or logs. Download weights from this Hub release.
[Reproduction](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/REPRODUCE.md) · [Training formulation](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/docs/final-boss.md) ·
[Release source archive](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/source.zip) · [Source provenance](https://huggingface.co/ntc-ai/krea2-particle-sliders/blob/main/source-provenance.json).

```bash
python -m pip install -r requirements.txt
python scripts/train_krea2.py --dummy
python scripts/infer_krea2.py --help
bash scripts/train_final_boss_gpu0.sh
bash scripts/train_eldritch_gpu0.sh
```

The release tests real ComfyUI Krea modules, all 128 patch mappings, embedded alpha
and clone isolation on CPU. Images are rendered through Diffusers. Full ComfyUI GPU
generation is not part of this audit.

## License

These adapters modify Krea 2 and are distributed under the [Krea 2 Community License
Agreement](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/KREA2-LICENSE.pdf). By accessing or using these weights, recipients
must agree to and be bound by that agreement. See [NOTICE](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/NOTICE).
This is an independent ntc-ai release, not an official or endorsed Krea product.
Independently authored source code is [MIT](https://huggingface.co/ntc-ai/krea2-particle-sliders/resolve/main/LICENSE); that does not relicense the weights.
