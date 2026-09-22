#!/usr/bin/env python3
"""Finish the 1536px showcase from the saved adapter without retraining."""
from argparse import Namespace
from pathlib import Path
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from krea2.cuda_train import CudaUNI, HELDOUT, write_json


def main():
    out = ROOT / 'outputs/final-boss-krea2-bbox'
    recovery = out / 'showcase_recovery'
    recovery.mkdir(exist_ok=True)
    for source in (ROOT / 'krea2/attention.py', ROOT / 'krea2/cuda_train.py', Path(__file__)):
        shutil.copyfile(source, recovery / source.name)
    args = Namespace(**json.loads((out / 'run.json').read_text()))
    args.resume = None
    args.load_te_lora = str(out / 'checkpoint-0400')
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    status = json.loads((out / 'status.json').read_text())
    write_json(recovery / 'original_status.json', status)
    status.update(phase='sampling', recovered_from=status.pop('error', None))
    write_json(out / 'status.json', status)
    try:
        backend = CudaUNI(args, [HELDOUT], recovery)
        rows = json.loads((out / 'prompts.json').read_text())
        verify = [('knight', rows[0]['neutral']), ('cave-warrior', rows[4]['neutral']),
                  ('heldout-bridge', HELDOUT), ('fruit-control', 'a bowl of fruit on a table')]
        sample_meta = []
        for label, prompt in verify:
            for scale in (0., .5, 1.):
                filename = f'{label}-scale-{scale:g}-seed-{args.sample_seed}.png'
                assert (out / 'samples' / filename).is_file()
                sample_meta.append(dict(file=filename, prompt=prompt, scale=scale,
                                        seed=args.sample_seed, resolution=args.sample_resolution))
        # Round trip: compare the reloaded PEFT adapter to the training-process render.
        from PIL import Image
        import numpy as np
        check = backend.generate(HELDOUT, 1., args.sample_seed, args.sample_resolution)
        check.save(recovery / 'reloaded-adapter-768.png')
        reference = np.asarray(Image.open(out / 'samples/heldout-bridge-scale-1-seed-42.png')).astype(float)
        delta = np.asarray(check).astype(float) - reference
        write_json(recovery / 'roundtrip.json', dict(pixel_identical=bool((delta == 0).all()),
                                                    pixel_mae=float(abs(delta).mean()),
                                                    pixel_max_error=float(abs(delta).max())))
        print('Adapter roundtrip mean pixel error:', abs(delta).mean(), flush=True)
        for scale in (0., 1.):
            print(f'Rendering 1536px showcase, scale={scale}', flush=True)
            image = backend.generate(HELDOUT, scale, 1234, args.final_resolution)
            filename = f'showcase-scale-{scale:g}-seed-1234.png'
            image.save(out / 'samples' / filename)
            sample_meta.append(dict(file=filename, prompt=HELDOUT, scale=scale,
                                    seed=1234, resolution=args.final_resolution))
        write_json(out / 'samples/metadata.json', dict(samples=sample_meta, steps=8, guidance=0., mu=1.15,
                   large_image_attention='PyTorch EFFICIENT_ATTENTION with repeated KV heads'))
        status.update(phase='complete', completed_at_unix=time.time(), showcase_resolution=args.final_resolution)
        write_json(out / 'status.json', status)
        print('Showcase complete.', flush=True)
    except BaseException as exc:
        status.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
