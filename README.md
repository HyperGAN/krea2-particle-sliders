# Krea2 Turbo-BBox Sliders

**Final Boss and Eldritch**, with original rank-16 LoRAs and compressed rank-8 distills.
Use **strength 1** for the calibrated effect. Same prompt, seed and sampler across each comparison.

Built on [jimmycarter/krea2-turbo-bbox](https://huggingface.co/jimmycarter/krea2-turbo-bbox),
`epoch-14-step-73184/transformer`. These are ordinary attention LoRAs. The originals
were trained directly as LoRAs; the distills compress those linear adapters.

## Samples

Left to right: **Original · strength 1 → Distill · strength 1 → Off**.
All images below are AI-generated, 768 × 768, seed 42, 8 steps, guidance 0, mu=1.15.
They were rendered from the released files, with no external alpha multiplier.

### Final Boss

![Final Boss: Original, Distill, Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/assets/final-boss-heldout-bridge.jpg)

Full resolution: [Original](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/heldout-bridge/original.png) · [Distill](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/heldout-bridge/distill.png) · [Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/heldout-bridge/off.png)

<details><summary>Exact prompt</summary>

```text
A lone warrior guarding a volcanic bridge.
@dramatic lighting, detailed fantasy game art; Digital illustration
~A stone bridge above a glowing lava river.
pe:1[220,80,780,950] An adult warrior in steel armor, holding a sword, standing guard.
```

</details>

<details><summary>Cathedral knight and unrelated fruit control</summary>

![Final Boss: Original, Distill, Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/assets/final-boss-knight.jpg)

Full resolution: [Original](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/knight/original.png) · [Distill](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/knight/distill.png) · [Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/knight/off.png)

<details><summary>Exact prompt</summary>

```text
A full-body armored knight in a ruined cathedral.
@cinematic lighting, detailed game concept art; Digital illustration
~Ruined gothic arches and a cold stone floor.
pe:1[230,100,770,940] An adult knight in practical steel armor, plain helmet, holding a longsword, calm stance.
```

</details>

![Final Boss: Original, Distill, Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/assets/final-boss-fruit-control.jpg)

Full resolution: [Original](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/fruit-control/original.png) · [Distill](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/fruit-control/distill.png) · [Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/final-boss/fruit-control/off.png)

<details><summary>Exact prompt</summary>

```text
a bowl of fruit on a table
```

</details>

</details>

### Eldritch

![Eldritch: Original, Distill, Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/assets/eldritch-heldout-bridge.jpg)

Full resolution: [Original](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/heldout-bridge/original.png) · [Distill](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/heldout-bridge/distill.png) · [Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/heldout-bridge/off.png)

<details><summary>Exact prompt</summary>

```text
A lone warrior guarding a volcanic bridge.
@dramatic lighting, detailed fantasy game art; Digital illustration
~A stone bridge above a glowing lava river.
pe:1[220,80,780,950] An adult warrior in steel armor, holding a sword, standing guard.
```

</details>

<details><summary>Cathedral knight and unrelated fruit control</summary>

![Eldritch: Original, Distill, Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/assets/eldritch-knight.jpg)

Full resolution: [Original](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/knight/original.png) · [Distill](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/knight/distill.png) · [Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/knight/off.png)

<details><summary>Exact prompt</summary>

```text
A full-body armored knight in a ruined cathedral.
@cinematic lighting, detailed game concept art; Digital illustration
~Ruined gothic arches and a cold stone floor.
pe:1[230,100,770,940] An adult knight in practical steel armor, plain helmet, holding a longsword, calm stance.
```

</details>

![Eldritch: Original, Distill, Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/assets/eldritch-fruit-control.jpg)

Full resolution: [Original](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/fruit-control/original.png) · [Distill](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/fruit-control/distill.png) · [Off](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/samples/eldritch/fruit-control/off.png)

<details><summary>Exact prompt</summary>

```text
a bowl of fruit on a table
```

</details>

</details>

The bridge prompt was excluded from the original six-pair training set, then used for
development comparisons. These examples are not a final-test benchmark. The original
Eldritch effect emphasizes organic armor and curling appendages; extra eyes and facial
tentacles remain weak. The fruit control shows some rendering-style drift.

## Downloads

| Slider | Original · ComfyUI | Distill · ComfyUI | Original · Diffusers | Distill · Diffusers |
|---|---|---|---|---|
| Final Boss | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/weights/comfyui/krea2-final-boss-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/distilled/comfyui/krea2-final-boss-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/weights/native/krea2-final-boss-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/distilled/native/krea2-final-boss-unit-alpha.safetensors?download=true) |
| Eldritch | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/weights/comfyui/krea2-eldritch-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/distilled/comfyui/krea2-eldritch-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/weights/native/krea2-eldritch-unit-alpha.safetensors?download=true) | [Download](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/distilled/native/krea2-eldritch-unit-alpha.safetensors?download=true) |

Original files are about 77 MB; distills are about 38 MB. Both contain 128 projection adapters.
[PEFT originals](https://huggingface.co/ntc-ai/krea2-concept-sliders/tree/main/weights/peft) · [PEFT distills](https://huggingface.co/ntc-ai/krea2-concept-sliders/tree/main/distilled/peft) ·
[Catalog](https://huggingface.co/ntc-ai/krea2-concept-sliders/blob/main/catalog.json) · [File hashes](https://huggingface.co/ntc-ai/krea2-concept-sliders/blob/main/release-manifest.json).

## ComfyUI

Use standard **Load LoRA** with a **ComfyUI export**, MODEL strength **1**, CLIP strength **0**.
No custom node is required. The optional `comfy_krea2.py` node, **Krea2 Turbo-BBox LoRA**
under **NTC/Krea2**, also handles these files and the native exports' alpha metadata.
[Plugin ZIP](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/comfyui/krea2-particle-sliders.zip?download=true) · [Setup](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/COMFYUI.md).

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
    "ntc-ai/krea2-concept-sliders",
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
validation and reproduction belong to [krea2-particle-sliders](https://github.com/HyperGAN/krea2-particle-sliders), renamed from
`krea2-concept-sliders`, following the release layout of [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders).
The Krea originals are linear LoRAs; this is not the Anima nonlinear particle training recipe.

The GitHub repository does not ship slider weights or logs. Download weights from this Hub release.
[Reproduction](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/REPRODUCE.md) · [Training formulation](https://github.com/HyperGAN/krea2-particle-sliders/blob/main/docs/final-boss.md) ·
[Release source archive](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/source.zip) · [Source provenance](https://huggingface.co/ntc-ai/krea2-concept-sliders/blob/main/source-provenance.json).

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
Agreement](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/KREA2-LICENSE.pdf). By accessing or using these weights, recipients
must agree to and be bound by that agreement. See [NOTICE](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/NOTICE).
This is an independent ntc-ai release, not an official or endorsed Krea product.
Independently authored source code is [MIT](https://huggingface.co/ntc-ai/krea2-concept-sliders/resolve/main/LICENSE); that does not relicense the weights.
