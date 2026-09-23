#!/usr/bin/env python3
"""Build the shared samples-first Hub/GitHub cards and curated release files."""
import argparse
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from release_tools.lora import write_json

REPO = 'ntc-ai/krea2-particle-sliders'
WEB = 'https://huggingface.co/' + REPO
RAW = WEB + '/resolve/main/'
GITHUB = 'https://github.com/HyperGAN/krea2-particle-sliders'


def comparison(folder, entry, case):
    samples = case['samples']
    assert [s['format'] for s in samples] == ['original', 'distill', 'off']
    records = [json.loads((folder / s['metadata']).read_text()) for s in samples]
    for key in ('prompt', 'seed', 'width', 'height', 'steps', 'guidance', 'mu'):
        assert len({r[key] for r in records}) == 1, (entry['id'], case['case'], key)
    tile, bar = 384, 48
    canvas = Image.new('RGB', (tile * 3, tile + bar), '#12151b')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans-Bold.ttf', 26)
    for i, (sample, label) in enumerate(zip(samples, ('ORIGINAL', 'DISTILL', 'OFF'))):
        with Image.open(folder / sample['image']) as image:
            assert image.size == (768, 768)
            canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS), (i * tile, bar))
        draw.text((i * tile + 16, 8), label, fill='#f4f6fa', font=font)
    path = f"assets/{entry['id']}-{case['case']}.jpg"
    (folder / path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / path, quality=94, subsampling=0)
    case['asset'] = path
    return records[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, default=ROOT / 'artifacts/release')
    args = parser.parse_args()
    folder = args.folder.resolve()
    catalog = json.loads((folder / 'catalog.json').read_text())
    body = '''# Krea2 Turbo-BBox Particle Sliders

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

'''
    for entry in catalog['sliders']:
        body += f"### {entry['label']}\n\n"
        featured = entry.get('featured_case', 'heldout-bridge')
        ordered = sorted(entry['comparisons'], key=lambda c: (c['case'] != featured,
                         {'heldout-bridge': 0, 'knight': 1, 'street-photo': 2, 'fruit-control': 3}.get(c['case'], 4)))
        for index, case in enumerate(ordered):
            if index == 1:
                body += '<details><summary>Additional comparisons and unrelated fruit control</summary>\n\n'
            record = comparison(folder, entry, case)
            label = case.get('label', case['case'].replace('-', ' ').title())
            body += f"#### {label}\n\n"
            if case['case'] == featured and entry.get('preview_note'):
                body += entry['preview_note'] + '\n\n'
            body += f"Seed **{record['seed']}** · [Side-by-side overview]({RAW}{case['asset']})\n\n"
            samples = {s['format']: s for s in case['samples']}
            for kind, caption in [('original', 'On (Original)'), ('distill', 'Distill')]:
                body += '<table width="100%">\n<tr><th width="50%">Off</th>'
                body += f'<th width="50%">{caption} · strength {samples[kind]["strength"]:g}</th></tr>\n<tr>\n'
                for key, title in [('off', 'Off'), (kind, caption)]:
                    url = RAW + samples[key]['image']
                    alt = html.escape(f"{entry['label']} — {label}: {title}")
                    body += f'<td width="50%"><a href="{url}"><img src="{url}" alt="{alt}" width="768"></a></td>\n'
                body += '</tr>\n</table>\n\n'
            body += '<details><summary>Exact prompt</summary>\n\n```text\n' + record['prompt'] + '\n```\n\n</details>\n\n'
        body += '</details>\n\n'
    body += '''The bridge prompt was excluded from the original six-pair training set, then used for
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
'''
    for entry in catalog['sliders']:
        links = [f"[Download]({RAW}{entry[kind]['files'][fmt]}?download=true)"
                 for kind, fmt in [('original', 'comfyui'), ('distill', 'comfyui'), ('original', 'native'), ('distill', 'native')]]
        body += '| ' + entry['label'] + ' | ' + ' | '.join(links) + ' |\n'
    body += f'''
Original files are about 77 MB; distills are about 38 MB. Both contain 128 projection adapters.
[PEFT originals]({WEB}/tree/main/weights/peft) · [PEFT distills]({WEB}/tree/main/distilled/peft) ·
[Catalog]({WEB}/blob/main/catalog.json) · [File hashes]({WEB}/blob/main/release-manifest.json).

## ComfyUI

Use standard **Load LoRA** with a **ComfyUI export**, MODEL strength **1**, CLIP strength **0**.
No custom node is required. The optional `comfy_krea2.py` node, **Krea2 Turbo-BBox LoRA**
under **NTC/Krea2**, also handles these files and the native exports' alpha metadata.
[Plugin ZIP]({RAW}comfyui/krea2-particle-sliders.zip?download=true) · [Setup]({GITHUB}/blob/main/COMFYUI.md).

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
    "{REPO}",
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
line, numeric character IDs, and panel-contained text. [Prompting guide]({GITHUB}/blob/main/PROMPTING.md).
The actual model tokenizer checked the 507-token content budget before training.

## Distillation and alpha

| Slider | Original rank / alpha | Distill rank / alpha | Projection relative MSE | Full-edit relative MSE | Edit cosine |
|---|---:|---:|---:|---:|---:|
'''
    for entry in catalog['sliders']:
        report = json.loads((folder / f"evidence/{entry['id']}/distillation.json").read_text())
        metric = report['selected']
        body += f"| {entry['label']} | 16 / {entry['original']['alpha']:g} | 8 / {entry['distill']['alpha']:g} | {report['projection_relative_mse']:.6f} | {metric['relative_full_edit_mse']:.6f} | {metric['edit_cosine']:.6f} |\n"
    body += f'''
Distills fit output principal components on 480 training activations per projection.
Alpha selection uses 16 separate development states and complete denoiser edits. These
errors measure approximation, not image quality. Compare the images before choosing
a format. [Method and all candidate measurements]({GITHUB}/blob/main/DISTILLATION.md).

The originals retain every learned matrix. Final Boss embeds alpha 16 at rank 16;
Eldritch embeds alpha 24 at rank 16, making released strength 1 equivalent to its
previous strength 1.5. Student alphas are selected separately. Both formats keep
the base's 8-step schedule; this distillation reduces adapter rank, not denoising steps.

## Training and source

Both originals used physical GPU 0, rank 16, 400 updates, learning rate 5e-5, 512px,
six paired captions, two cached trajectory seeds per pair, and preservation weight 0.1
every fifth update. The base and text encoder stayed frozen. Source, configurations,
validation and reproduction belong to [krea2-particle-sliders]({GITHUB}), following
the release layout of [anima-particle-sliders](https://github.com/HyperGAN/anima-particle-sliders).
The Krea originals are linear LoRAs; this is not the Anima nonlinear particle training recipe.

The GitHub repository does not ship slider weights or logs. Download weights from this Hub release.
[Reproduction]({GITHUB}/blob/main/REPRODUCE.md) · [Training formulation]({GITHUB}/blob/main/docs/final-boss.md) ·
[Release source archive]({RAW}source.zip) · [Source provenance]({WEB}/blob/main/source-provenance.json).

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
Agreement]({RAW}KREA2-LICENSE.pdf). By accessing or using these weights, recipients
must agree to and be bound by that agreement. See [NOTICE]({RAW}NOTICE).
This is an independent ntc-ai release, not an official or endorsed Krea product.
Independently authored source code is [MIT]({RAW}LICENSE); that does not relicense the weights.
'''
    frontmatter = f'''---
license: other
license_name: krea-2-community-license
license_link: {RAW}KREA2-LICENSE.pdf
base_model: jimmycarter/krea2-turbo-bbox
base_model_relation: adapter
library_name: diffusers
pipeline_tag: text-to-image
language:
- en
tags:
- krea2
- particle-sliders
- lora
- comfyui
---
'''
    (folder / 'README.md').write_text(frontmatter + body)
    (ROOT / 'README.md').write_text(body)
    write_json(folder / 'catalog.json', catalog)
    write_json(folder / 'validation/environment.json', dict(
        python=sys.version,
        versions={name: importlib.metadata.version(name) for name in
                  ('torch', 'diffusers', 'transformers', 'peft', 'accelerate', 'safetensors', 'huggingface_hub')},
        sample_runtime='Diffusers Krea2Pipeline with krea2/attention.py',
        precision='bfloat16 base and loaded Diffusers adapters; float32 stored factors; float32 matmul precision high'))
    shutil.copyfile(ROOT / 'outputs/release-work/development.json', folder / 'evidence/development.json')
    for name in ('DISTILLATION.md', 'REPRODUCE.md', 'COMFYUI.md', 'PROMPTING.md', 'NOTICE', 'LICENSE', 'KREA2-LICENSE.pdf'):
        shutil.copyfile(ROOT / name, folder / name)
    (folder / 'docs').mkdir(exist_ok=True)
    for name in ('final-boss.md', 'eldritch.md'):
        shutil.copyfile(ROOT / 'docs' / name, folder / 'docs' / name)
    for entry in catalog['sliders']:
        name = entry['id']
        run = ROOT / f'outputs/{name}-krea2-bbox'
        source = json.loads((run / 'run.json').read_text())
        keys = ['model_id', 'revision', 'transformer_subfolder', 'skeleton_model', 'skeleton_revision', 'rank',
                'steps', 'resolution', 'lr', 'seed', 'hold_weight', 'cache_seeds', 'versions', 'prompts_sha256',
                'source_files_sha256', 'source_commit', 'objective', 'teacher_trajectories', 'preservation']
        write_json(folder / f'evidence/{name}/training.json', {k: source[k] for k in keys})
        for filename in ('prompts.json', 'prompt_token_counts.json', 'verification.json'):
            shutil.copyfile(run / filename, folder / f'evidence/{name}/{filename}')
    plugin = ['__init__.py', 'comfy_krea2.py', 'COMFYUI.md', 'LICENSE', 'NOTICE', 'KREA2-LICENSE.pdf']
    (folder / 'comfyui').mkdir(exist_ok=True)
    with zipfile.ZipFile(folder / 'comfyui/krea2-particle-sliders.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in plugin:
            archive.write(ROOT / name, 'krea2-particle-sliders/' + name)
    print('Built cards, comparison assets, curated evidence, and plugin archive.')


if __name__ == '__main__':
    main()
