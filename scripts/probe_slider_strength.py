#!/usr/bin/env python3
"""Compare stronger slider settings with its unmodified positive teacher."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from krea2.cuda_train import CudaUNI, HELDOUT, make_grid, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 1.5, 2.0])
    options = parser.parse_args()
    out = options.run.resolve()
    args = argparse.Namespace(**json.loads((out / 'run.json').read_text()))
    args.resume = None
    args.load_te_lora = str(out / f'checkpoint-{args.steps:04d}')
    rows = json.loads((out / 'prompts.json').read_text())
    cases = [('knight', rows[0]['neutral']), ('heldout-bridge', HELDOUT)]
    probes = out / 'strength-probes'
    probes.mkdir(exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision('high')
    backend = CudaUNI(args, [text for _, text in cases] + [rows[0]['positive']], probes)
    metadata = []
    variants = []
    for name, text in cases:
        row = []
        for scale in options.scales:
            print(f'Probe: {name}, scale {scale:g}', flush=True)
            image = backend.generate(text, scale, args.sample_seed, args.sample_resolution)
            filename = f'{name}-scale-{scale:g}-seed-{args.sample_seed}.png'
            image.save(probes / filename)
            row.append(image)
            metadata.append(dict(file=filename, prompt=text, scale=scale, seed=args.sample_seed,
                                 resolution=args.sample_resolution, teacher=False))
            write_json(probes / 'metadata.json', dict(samples=metadata, steps=8, guidance=0., mu=1.15))
        variants.append(row)
    make_grid(variants, options.scales, probes / 'grid.png', [name for name, _ in cases])
    print('Probe: frozen positive-caption teacher', flush=True)
    teacher = backend.generate(rows[0]['positive'], 0., args.sample_seed, args.sample_resolution)
    filename = f'knight-positive-teacher-seed-{args.sample_seed}.png'
    teacher.save(probes / filename)
    metadata.append(dict(file=filename, prompt=rows[0]['positive'], scale=0., seed=args.sample_seed,
                         resolution=args.sample_resolution, teacher=True))
    write_json(probes / 'metadata.json', dict(samples=metadata, steps=8, guidance=0., mu=1.15))
    print(f'Complete: {probes}', flush=True)


if __name__ == '__main__':
    main()
