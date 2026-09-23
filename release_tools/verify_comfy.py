#!/usr/bin/env python3
"""Validate exported adapters against real ComfyUI Krea modules on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('comfy', type=Path)
parser.add_argument('folder', type=Path)
options = parser.parse_args()
comfy_root, folder = options.comfy.resolve(), options.folder.resolve()
sys.path[:0] = [str(ROOT), str(comfy_root)]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
import comfy.lora
import comfy.model_base
import comfy.supported_models
from comfy.model_patcher import ModelPatcher
import folder_paths
from safetensors.torch import load_file
from comfy_krea2 import Krea2BboxLora
from release_tools.lora import comfy_name, digest, write_json

torch.set_num_threads(4)
torch.set_grad_enabled(False)
config = comfy.supported_models.Krea2(dict(image_model='krea2', layers=28))
config.set_inference_dtype(torch.float32, None)
config.custom_operations = comfy.ops.manual_cast
host = comfy.model_base.Krea2(config, device=torch.device('meta'))
patcher = ModelPatcher(host, torch.device('cpu'), torch.device('cpu'))
mapping = comfy.lora.model_lora_keys_unet(host, {})
reports = []
for path in sorted(folder.glob('*/comfyui/*.safetensors')):
    state = load_file(str(path))
    native_path = path.parent.parent / 'native' / path.name
    native = load_file(str(native_path))
    patches = comfy.lora.load_lora(state, mapping)
    assert len(patches) == 128, (path, len(patches))
    standard = patcher.clone()
    assert len(standard.add_patches(patches, 1.)) == 128
    folder_paths.add_model_folder_path('loras', str(path.parent), is_default=True)
    custom = Krea2BboxLora().load(patcher, path.name, 1.)[0]
    assert len(custom.patches) == 128 and not patcher.patches
    assert not Krea2BboxLora().load(patcher, path.name, 0.)[0].patches
    errors = []
    for key, down in native.items():
        if not key.endswith('.lora_A.weight'):
            continue
        name = key.removeprefix('transformer.').removesuffix('.lora_A.weight')
        prefix = 'diffusion_model.' + comfy_name(name)
        up = native['transformer.' + name + '.lora_B.weight']
        assert torch.equal(down, state[prefix + '.lora_down.weight'])
        assert torch.equal(up, state[prefix + '.lora_up.weight'])
        weight_key = prefix + '.weight'
        shape = (up.shape[0], down.shape[1])
        assert host.state_dict()[weight_key].shape == shape
        actual = comfy.lora.calculate_weight(standard.patches[weight_key], torch.zeros(shape), weight_key)
        alpha = float(state[prefix + '.alpha'])
        reference = (alpha / down.shape[0]) * (up @ down)
        relative = float((actual - reference).norm() / reference.norm().clamp_min(1e-30))
        assert relative < 1e-6, (path, name, relative)
        errors.append(relative)
        del actual, reference
    # The custom node also translates alpha metadata in the native export.
    folder_paths.add_model_folder_path('loras', str(native_path.parent), is_default=True)
    native_custom = Krea2BboxLora().load(patcher, path.name, 1.)[0]
    assert len(native_custom.patches) == 128
    key = next(iter(native_custom.patches))
    shape = host.state_dict()[key].shape
    actual = comfy.lora.calculate_weight(native_custom.patches[key], torch.zeros(shape), key)
    reference = comfy.lora.calculate_weight(standard.patches[key], torch.zeros(shape), key)
    assert torch.equal(actual, reference)
    del actual, reference
    reports.append(dict(file=str(path.relative_to(folder)), patches=128,
        sha256=digest(path), native_sha256=digest(native_path),
        max_relative_matrix_error=max(errors), native_comfy_matrices_equal=True,
        custom_node_applies_weights=True, native_node_alpha_equal=True, clone_isolation=True, zero_bypass=True))
    print('Verified', path.relative_to(folder), flush=True)
    del standard, custom, native_custom

assert reports, 'No exported adapters found'
import subprocess
write_json(folder / 'validation/comfyui.json', dict(passed=True, files=reports,
    comfy_revision=subprocess.check_output(['git', '-C', str(comfy_root), 'rev-parse', 'HEAD'], text=True).strip(),
    limitation='CPU integration covers real Krea topology and Comfy patch math; no full ComfyUI GPU render.'))
del patcher, host
import gc
gc.collect()
