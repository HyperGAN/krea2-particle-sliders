"""Activation-weighted rank reduction and portable alpha-bearing exports."""
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def read_factors(path):
    state = load_file(str(path))
    state = {k.removeprefix('base_model.model.').removeprefix('transformer.'): v
             for k, v in state.items()}
    factors = {}
    for key in sorted(state):
        if key.endswith('.lora_A.weight'):
            name = key.removesuffix('.lora_A.weight')
            down, up = state[key], state[name + '.lora_B.weight']
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f'Invalid LoRA shapes: {name}')
            factors[name] = (down, up)
    if not factors or len(state) != len(factors) * 2:
        raise ValueError('Expected only paired ordinary LoRA matrices')
    return factors


def compress_factors(down, up, gram, rank=8):
    """Minimize captured output error by projecting BA onto output principal axes.

    gram is sum(h h^T), h=A x, captured at the original adapter's inputs.
    B=QR reduces the covariance eigensolve to the source rank (16 here).
    Returned mix satisfies student_down = mix @ original_down.
    """
    if not 0 < rank <= down.shape[0]:
        raise ValueError('Student rank must be between one and the teacher rank')
    a, b, g = down.double(), up.double(), gram.double()
    if g.shape != (a.shape[0], a.shape[0]) or not torch.isfinite(g).all():
        raise ValueError('Invalid activation Gram matrix')
    q, r = torch.linalg.qr(b, mode='reduced')
    covariance = r @ g @ r.T
    values, axes = torch.linalg.eigh((covariance + covariance.T) * .5)
    basis = q @ axes[:, -rank:]
    mix = basis.T @ b
    student_down = mix @ a
    # Balance factor magnitudes without changing their product.
    balance = (student_down.norm(dim=1) / basis.norm(dim=0).clamp_min(1e-30)).clamp_min(1e-30).sqrt()
    student_up = basis * balance
    student_down = student_down / balance[:, None]
    mix = mix / balance[:, None]
    energy = values.clamp_min(0)
    error = float(energy[:-rank].sum() / energy.sum().clamp_min(1e-30))
    return student_down.float().contiguous(), student_up.float().contiguous(), mix.float(), error


def projection_error(up, student_up, mix, gram):
    b, s, m, g = (v.double() for v in (up, student_up, mix, gram))
    residual = s @ m - b
    return float(torch.trace(residual.T @ residual @ g)), float(torch.trace(b.T @ b @ g))


def comfy_name(name):
    name = name.replace('transformer_blocks.', 'blocks.').replace('text_fusion.', 'txtfusion.')
    for source, target in (('to_out.0', 'wo'), ('to_q', 'wq'), ('to_k', 'wk'), ('to_v', 'wv')):
        name = name.replace('.attn.' + source, '.attn.' + target)
    if not re.fullmatch(r'(?:blocks\.\d+|txtfusion\.(?:layerwise|refiner)_blocks\.\d+)\.attn\.w[qkvo]', name):
        raise ValueError(f'Unsupported Krea projection: {name}')
    return name


def _save_immutable(path, state, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = load_file(str(path))
        with safe_open(path, framework='pt') as handle:
            old_meta = handle.metadata()
        if old.keys() != state.keys() or any(not torch.equal(old[k], v) for k, v in state.items()) or old_meta != metadata:
            raise FileExistsError(f'Refusing to replace an existing export: {path}')
        return
    save_file(state, str(path), metadata=metadata)


def export_lora(root, tier, name, factors, alpha, parent_sha256, method):
    root = Path(root)
    ranks = {a.shape[0] for a, _ in factors.values()}
    if len(ranks) != 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Exports require a uniform rank and positive finite alpha')
    rank = ranks.pop()
    config = dict(r=rank, lora_alpha=float(alpha), target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
                  lora_dropout=0., bias='none', inference_mode=True, use_dora=False, use_rslora=False)
    native, comfy, peft = {}, {}, {}
    for module, (down, up) in sorted(factors.items()):
        for suffix, tensor in (('lora_A.weight', down), ('lora_B.weight', up)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite tensor: {module}')
            native[f'transformer.{module}.{suffix}'] = tensor.contiguous()
            peft[f'base_model.model.{module}.{suffix}'] = tensor.contiguous()
        prefix = 'diffusion_model.' + comfy_name(module)
        comfy[prefix + '.lora_down.weight'] = down.contiguous()
        comfy[prefix + '.lora_up.weight'] = up.contiguous()
        comfy[prefix + '.alpha'] = torch.tensor(alpha, dtype=torch.float64)
    metadata = dict(format='pt', model='jimmycarter/krea2-turbo-bbox', rank=str(rank),
                    alpha=str(float(alpha)), alpha_scaling='alpha/rank', method=method,
                    parent_sha256=parent_sha256, license='Krea 2 Community License')
    filename = f'krea2-{name}-unit-alpha.safetensors'
    paths = dict(native=f'{tier}/native/{filename}', comfyui=f'{tier}/comfyui/{filename}',
                 peft=f'{tier}/peft/{name}/adapter_model.safetensors')
    native_meta = dict(metadata, lora_adapter_metadata=json.dumps(
        {f'transformer.{k}': v for k, v in config.items()}, sort_keys=True))
    _save_immutable(root / paths['native'], native, native_meta)
    _save_immutable(root / paths['comfyui'], comfy, metadata)
    _save_immutable(root / paths['peft'], peft, metadata)
    config_path = (root / paths['peft']).with_name('adapter_config.json')
    peft_config = dict(config, peft_type='LORA')
    if config_path.exists() and json.loads(config_path.read_text()) != peft_config:
        raise FileExistsError(config_path)
    write_json(config_path, peft_config)
    return dict(rank=rank, alpha=alpha, projections=len(factors), recommended_strength=1.,
                files=paths, sha256={key: digest(root / path) for key, path in paths.items()},
                parent_sha256=parent_sha256, method=method)
