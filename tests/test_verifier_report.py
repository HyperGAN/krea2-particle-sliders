"""The artifact verifier reports its observed evidence without claiming gradient inspection."""
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
from safetensors.torch import save_file
import torch


def test_gan_verifier_reports_logged_losses_without_gradient_claim(tmp_path, monkeypatch):
    run = dict(steps=2, name='fixture', sample_seed=42, save_every=1,
               final_resolution=0, optimized_mse=False, objective='shared conditional GAN')
    (tmp_path / 'run.json').write_text(json.dumps(run))
    (tmp_path / 'status.json').write_text(json.dumps(dict(phase='complete')))
    (tmp_path / 'train.jsonl').write_text(''.join(json.dumps(dict(step=i, g_adv=.3, d_adv=.4, d_pen=.1)) + '\n'
                                                for i in (1, 2)))
    checkpoint = tmp_path / 'checkpoint-0002'
    checkpoint.mkdir()
    module = 'transformer_blocks.0.attn.to_q'
    factors = {module + '.lora_A.weight': torch.ones(2, 4), module + '.lora_B.weight': torch.ones(4, 2)}
    save_file({'base_model.model.' + k: v for k, v in factors.items()}, checkpoint / 'adapter_model.safetensors')
    save_file({'transformer.' + k: v for k, v in factors.items()}, checkpoint / 'fixture.safetensors')
    samples = tmp_path / 'samples'
    samples.mkdir()
    records = []
    image = Image.new('RGB', (32, 32), 'white')
    for i in range(12):
        name = 'knight-scale-0-seed-42.png' if i == 0 else f'fixture-{i}.png'
        image.save(samples / name)
        records.append(dict(file=name, resolution=32))
    image.save(tmp_path / 'preview-step-0001-scale-0.png')
    (samples / 'metadata.json').write_text(json.dumps(dict(steps=8, guidance=0, mu=1.15, samples=records)))
    (tmp_path / 'prompt_token_counts.json').write_text(json.dumps(dict(fixture=3)))
    script = Path(__file__).resolve().parents[1] / 'scripts/verify_slider_run.py'
    spec = importlib.util.spec_from_file_location('krea2_verifier_fixture', script)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    monkeypatch.setattr(sys, 'argv', [str(script), str(tmp_path), '--visual-review', 'fixture statement',
                                     '--cpu-tests-passed', '1'])
    verifier.main()
    report = json.loads((tmp_path / 'verification.json').read_text())
    assert report['finite_logged_losses'] is True
    assert report['gradients_independently_inspected'] is False
    assert report['finite_logged_gradient_norms'] is False
    assert 'finite_losses_and_gradients' not in report
