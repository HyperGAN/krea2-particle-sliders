#!/usr/bin/env python3
"""Validate a completed local slider run and record a human visual review."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--visual-review', required=True)
    parser.add_argument('--cpu-tests-passed', required=True, type=int)
    args = parser.parse_args()
    root = args.run.resolve()
    run = json.loads((root / 'run.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'complete', status
    records = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == list(range(1, run['steps'] + 1))
    assert all(math.isfinite(r[key]) for r in records for key in
               ('loss', 'raw_loss', 'baseline_gap', 'adapter_delta', 'grad_norm'))
    checkpoint = root / f"checkpoint-{run['steps']:04d}"
    weights = checkpoint / f"{run['name']}.safetensors"
    exported = load_file(weights)
    peft = load_file(checkpoint / 'adapter_model.safetensors')
    expected = {'transformer.' + key.removeprefix('base_model.model.'): value
                for key, value in peft.items()}
    assert exported.keys() == expected.keys(), 'Diffusers and PEFT keys differ'
    assert all(torch.equal(value, expected[key]) for key, value in exported.items())
    assert all(torch.isfinite(value).all().item() for value in exported.values())
    assert any(value.count_nonzero().item() for key, value in exported.items() if 'lora_B' in key)
    metadata = json.loads((root / 'samples/metadata.json').read_text())
    assert (metadata['steps'], metadata['guidance'], metadata['mu']) == (8, 0, 1.15)
    samples = metadata['samples']
    assert len(samples) == 14 and len({s['file'] for s in samples}) == 14
    for sample in samples:
        with Image.open(root / 'samples' / sample['file']) as image:
            assert image.size == (sample['resolution'],) * 2
            image.verify()
    first = np.asarray(Image.open(root / 'preview-step-0050-scale-0.png'))
    last = np.asarray(Image.open(root / f"samples/knight-scale-0-seed-{run['sample_seed']}.png"))
    assert np.array_equal(first, last), 'Scale-zero baseline drifted during training'
    counts = json.loads((root / 'prompt_token_counts.json').read_text())
    assert max(counts.values()) <= 507
    recent = [r for r in records[-50:] if r['kind'] == 'edit']
    report = dict(
        completed_updates=len(records), finite_losses_and_gradients=True,
        adapter_tensors=len(exported), adapter_parameters=sum(v.numel() for v in exported.values()),
        adapter_finite=True, diffusers_export_matches_peft=True,
        adapter_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        recent_edit_error_over_base=sum(r['raw_loss'] for r in recent) / sum(r['baseline_gap'] for r in recent),
        max_prompt_tokens=max(counts.values()), cpu_tests_passed=args.cpu_tests_passed,
        scale_zero_matches_early_baseline_exactly=True, sample_files_verified=len(samples),
        showcase_resolution=max(s['resolution'] for s in samples),
        visual_review=args.visual_review,
    )
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
