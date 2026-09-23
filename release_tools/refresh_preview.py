#!/usr/bin/env python3
"""Render a matched photographic Final Boss preview from the released adapters."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from release_tools.distill import Runtime
from release_tools.lora import digest, write_json

PROMPT = (
    'A candid full-body street photograph of a commuter on a rainy night in Tokyo.\n'
    '@35mm street photography, realistic skin texture, natural proportions, cinematic neon reflections; Photograph\n'
    '~A narrow city street with small restaurants, wet asphalt, red and blue neon reflections, '
    'soft background bokeh and gentle rain.\n'
    'pe:1[240,100,760,950] An adult man with short dark hair in a simple dark wool overcoat, '
    'gray sweater, jeans and ordinary leather shoes, holding a closed black umbrella at his side, '
    'standing casually and looking toward the camera.'
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT / 'artifacts/release')
    options = parser.parse_args()
    folder = options.folder.resolve()
    work = ROOT / 'outputs/photo-preview'
    work.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((folder / 'catalog.json').read_text())
    entry = next(e for e in catalog['sliders'] if e['id'] == 'final-boss')
    run = ROOT / 'outputs/final-boss-krea2-bbox'
    args = argparse.Namespace(**json.loads((run / 'run.json').read_text()))
    args.resume = None
    args.load_te_lora = str(run / 'checkpoint-0400')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    write_json(work / 'status.json', dict(phase='loading'))
    runtime = Runtime(args, [PROMPT], work)
    for kind in ('original', 'distill'):
        runtime.load(kind, folder / entry[kind]['files']['native'])
    samples = []
    for kind, strength in [('original', 1.), ('distill', 1.), ('off', 0.)]:
        adapter = 'distill' if kind == 'distill' else 'original'
        runtime.active = adapter
        export = entry[adapter]
        image = runtime.generate(PROMPT, strength, 4242, 768)
        path = f'samples/final-boss/street-photo/{kind}.png'
        destination = folder / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        record = dict(prompt=PROMPT, seed=4242, width=768, height=768, steps=8, guidance=0., mu=1.15,
                      format=kind, strength=strength, alpha=export['alpha'], rank=export['rank'],
                      adapter=export['files']['native'], adapter_sha256=digest(folder / export['files']['native']),
                      model_id=args.model_id, model_revision=args.revision, split='preview refresh')
        write_json(destination.with_suffix('.json'), record)
        samples.append(dict(image=path, metadata=str(Path(path).with_suffix('.json')),
                            format=kind, strength=strength))
        write_json(work / 'status.json', dict(phase='rendering', last=path))
        print('Rendered', path, flush=True)
    entry['comparisons'] = [c for c in entry['comparisons'] if c['case'] != 'street-photo']
    entry['comparisons'].append(dict(case='street-photo', samples=samples))
    entry['featured_case'] = 'street-photo'
    write_json(folder / 'catalog.json', catalog)
    write_json(work / 'status.json', dict(phase='complete'))


if __name__ == '__main__':
    main()
