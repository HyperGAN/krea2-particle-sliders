#!/usr/bin/env python3
"""Compress both rank-16 LoRAs, calibrate alpha, and render the release files."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from peft.tuners.tuners_utils import BaseTunerLayer

from krea2.cuda_train import CudaUNI, HELDOUT
from release_tools.lora import read_factors, compress_factors, projection_error, export_lora, digest, write_json

GAINS = {'final-boss': 1., 'eldritch': 1.5}
DEV = [
    'An armored traveler at a forest gate.\n'
    '@detailed fantasy concept art; Digital illustration\n'
    '~An ancient stone gate surrounded by moss and tall trees.\n'
    'pe:1[200,80,800,960] An adult traveler in simple steel armor, holding a spear, standing upright.',
    'A guard inside a space station.\n'
    '@cinematic science fiction art; Digital illustration\n'
    '~A quiet orbital station corridor with gray panels and white lights.\n'
    'pe:1[240,90,760,950] An adult guard in a plain protective suit and helmet, holding a baton.',
]


class Runtime(CudaUNI):
    def __init__(self, args, texts, out):
        super().__init__(args, texts, out)
        self.model = self.model.unload()
        self.pipe.transformer = self.model
        self.pipe.unload_lora_weights()
        self.active = None

    def load(self, name, path):
        self.pipe.load_lora_weights(str(path), adapter_name=name)
        self.active = name
        self.pipe.set_adapters(name, adapter_weights=1.)
        self.model.eval()

    @contextmanager
    def scale(self, value):
        if value == 0:
            self.pipe.disable_lora()
            try:
                yield
            finally:
                self.pipe.enable_lora()
        else:
            self.pipe.set_adapters(self.active, adapter_weights=value)
            try:
                yield
            finally:
                self.pipe.set_adapters(self.active, adapter_weights=1.)


@torch.no_grad()
def capture(runtime, records, factors, status_path, phase):
    modules = dict(runtime.model.named_modules())
    grams = {name: torch.zeros((down.shape[0], down.shape[0]), device=runtime.device)
             for name, (down, _) in factors.items()}
    counts = {name: 0 for name in factors}
    handles = []
    for name in factors:
        module = modules[name]
        if not isinstance(module, BaseTunerLayer):
            raise TypeError(f'Adapter projection missing: {name}')
        def hook(module, args, name=name):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            indices = torch.linspace(0, len(x) - 1, min(8, len(x)), device=x.device).long()
            h = x[indices] @ module.lora_A[runtime.active].weight.float().T
            grams[name].add_(h.T @ h)
            counts[name] += len(indices)
        handles.append(module.register_forward_pre_hook(hook))
    predictions = []
    try:
        with runtime.scale(1.):
            for index, record in enumerate(records):
                predictions.append(runtime.forward(record['prompt'], record['z'], record['t']).cpu())
                if (index + 1) % 5 == 0 or index + 1 == len(records):
                    print(f'{phase}: {index + 1}/{len(records)}', flush=True)
                    write_json(status_path, dict(phase=phase, completed=index + 1, total=len(records)))
    finally:
        for handle in handles:
            handle.remove()
    return {k: v.cpu() for k, v in grams.items()}, counts, predictions


@torch.no_grad()
def audit(runtime, records, teachers, multiplier):
    error = reference = predicted = dot = 0.
    with runtime.scale(multiplier):
        for record, teacher in zip(records, teachers):
            student = runtime.forward(record['prompt'], record['z'], record['t']).cpu().double()
            original = teacher.double() - record['baseline'].double()
            edit = student - record['baseline'].double()
            error += float((edit - original).square().sum())
            reference += float(original.square().sum())
            predicted += float(edit.square().sum())
            dot += float((original * edit).sum())
    return dict(multiplier=multiplier, relative_full_edit_mse=error / max(reference, 1e-30),
                edit_cosine=dot / max((reference * predicted) ** .5, 1e-30),
                student_to_teacher_rms=(predicted / max(reference, 1e-30)) ** .5)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/release')
    parser.add_argument('--work', type=Path, default=ROOT / 'outputs/release-work')
    args_cli = parser.parse_args()
    out, work = args_cli.output.resolve(), args_cli.work.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    status = work / 'status.json'
    runs = {name: ROOT / f'outputs/{name}-krea2-bbox' for name in GAINS}
    rows = {name: json.loads((run / 'prompts.json').read_text()) for name, run in runs.items()}
    factors, originals = {}, {}
    for name, run in runs.items():
        source = run / f'checkpoint-0400/{name}-krea2-bbox.safetensors'
        factors[name] = read_factors(source)
        originals[name] = export_lora(out, 'weights', name, factors[name], 16 * GAINS[name],
                                     digest(source), 'Original rank-16 LoRA; only embedded alpha calibrated')
    first = runs['final-boss']
    args = argparse.Namespace(**json.loads((first / 'run.json').read_text()))
    args.resume = None
    args.load_te_lora = str(first / 'checkpoint-0400')
    texts = [row['neutral'] for row in rows['final-boss']] + [HELDOUT, 'a bowl of fruit on a table'] + DEV
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision('high')
    write_json(status, dict(phase='loading'))
    runtime = Runtime(args, texts, work)
    runtime.load('final-boss', out / originals['final-boss']['files']['native'])
    dev = []
    for index, text in enumerate(DEV):
        dev.extend(runtime.trajectory(text, text, 77001 + index))
    write_json(work / 'development.json', dict(prompts=DEV, seeds=[77001, 77002],
        positions=list(range(8)), resolution=512, use='alpha selection and approximation diagnostics; not final test'))
    catalog = dict(base_model=args.model_id, revision=args.revision,
                   transformer_subfolder=args.transformer_subfolder, skeleton_model=args.skeleton_model,
                   skeleton_revision=args.skeleton_revision, steps=8, guidance=0., mu=1.15, sliders=[])
    for name, run in runs.items():
        if name != 'final-boss':
            runtime.load(name, out / originals[name]['files']['native'])
        runtime.active = name
        runtime.pipe.set_adapters(name, adapter_weights=1.)
        cache = torch.load(run / 'teacher_cache.pt', map_location='cpu', weights_only=False)
        records = [record for index, record in enumerate(cache['edit']) if index % 8 in (0, 2, 4, 6, 7)]
        train_grams, train_counts, _ = capture(runtime, records, factors[name], status, name + '-capture-train')
        dev_grams, dev_counts, teacher_predictions = capture(runtime, dev, factors[name], status, name + '-capture-dev')
        student, projection_reports = {}, []
        for module, (down, up) in factors[name].items():
            a, b, mix, train_error = compress_factors(down, up, train_grams[module], rank=8)
            student[module] = (a, b)
            error, norm = projection_error(up, b, mix, dev_grams[module])
            projection_reports.append(dict(module=module, train_tokens=train_counts[module], dev_tokens=dev_counts[module],
                train_relative_mse=train_error, dev_squared_error=error, dev_teacher_squared_norm=norm,
                dev_relative_mse=error / max(norm, 1e-30)))
        initial = export_lora(work, 'initial-distills', name, student, 8 * GAINS[name],
            originals[name]['sha256']['native'], 'Rank-8 output PCA weighted by training activations')
        runtime.load(name + '-candidate', work / initial['files']['native'])
        candidates = []
        for multiplier in (.75, 1., 1.25, 1.5, 2.):
            result = audit(runtime, dev, teacher_predictions, multiplier)
            candidates.append(result)
            print(f'{name} alpha candidate: {result}', flush=True)
            write_json(status, dict(phase=name + '-alpha-sweep', candidate=result))
        selected = min(candidates, key=lambda item: item['relative_full_edit_mse'])
        alpha = 8 * GAINS[name] * selected['multiplier']
        distill = export_lora(out, 'distilled', name, student, alpha,
            originals[name]['sha256']['native'], 'Rank-8 activation-weighted compression; alpha selected on development states')
        runtime.load(name + '-distill', out / distill['files']['native'])
        evaluation = dict(method='activation-weighted output PCA of a linear rank-16 teacher',
            training_records=len(records), development_records=len(dev), student_rank=8,
            projection_relative_mse=sum(r['dev_squared_error'] for r in projection_reports) /
                max(sum(r['dev_teacher_squared_norm'] for r in projection_reports), 1e-30),
            projections=projection_reports, alpha_candidates=candidates, selected=selected,
            original_alpha=originals[name]['alpha'], distilled_alpha=alpha,
            limitation='Development approximation measurements, not image quality or a final-test benchmark')
        write_json(out / f'evidence/{name}/distillation.json', evaluation)
        comparisons, replays = [], []
        cases = [('knight', rows[name][0]['neutral']), ('heldout-bridge', HELDOUT),
                 ('fruit-control', 'a bowl of fruit on a table')]
        for case, text in cases:
            samples = []
            for kind, adapter, strength in [('original', name, 1.), ('distill', name + '-distill', 1.), ('off', name, 0.)]:
                runtime.active = adapter
                image = runtime.generate(text, strength, 42, 768)
                path = f'samples/{name}/{case}/{kind}.png'
                destination = out / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination)
                export = distill if kind == 'distill' else originals[name]
                metadata = dict(prompt=text, seed=42, width=768, height=768, steps=8, guidance=0., mu=1.15,
                    format=kind, strength=strength, alpha=export['alpha'], rank=export['rank'],
                    adapter=export['files']['native'], adapter_sha256=export['sha256']['native'],
                    model_id=args.model_id, model_revision=args.revision, split='development')
                write_json(destination.with_suffix('.json'), metadata)
                samples.append(dict(image=path, metadata=str(Path(path).with_suffix('.json')), format=kind, strength=strength))
                # Check Off against the original run and calibrated originals against their selected sweep.
                reference = run / f'samples/{case}-scale-0-seed-42.png' if kind == 'off' else None
                if kind == 'original' and (name == 'final-boss' or case != 'fruit-control'):
                    reference = run / (f'samples/{case}-scale-1-seed-42.png' if name == 'final-boss' else
                                       f'strength-probes/{case}-scale-1.5-seed-42.png')
                if reference is not None:
                    old, new = np.asarray(Image.open(reference)).astype(float), np.asarray(image).astype(float)
                    replays.append(dict(case=case, format=kind, pixel_identical=bool(np.array_equal(old, new)),
                                        pixel_mae=float(abs(old - new).mean())))
                print(f'Sample: {path}', flush=True)
                write_json(status, dict(phase=name + '-samples', last=path))
            comparisons.append(dict(case=case, samples=samples))
        write_json(out / f'evidence/{name}/alpha-replay.json', dict(checks=replays))
        catalog['sliders'].append(dict(id=name, label='Final Boss' if name == 'final-boss' else 'Eldritch',
            original=originals[name], distill=distill, recommended_strength=1., comparisons=comparisons))
        write_json(out / 'catalog.json', catalog)
        del cache, records, train_grams, dev_grams, teacher_predictions
    write_json(status, dict(phase='complete', sliders=list(GAINS)))
    print('Release distillation and samples complete.', flush=True)


if __name__ == '__main__':
    main()
