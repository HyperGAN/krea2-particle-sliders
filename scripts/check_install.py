"""Report the installed cohort without loading weights or constructing a model."""
import argparse
import importlib
from importlib.metadata import distribution
import json
import sys


def inspect_install(*, native=False, profile='supported'):
    product = distribution('krea2-particle-sliders')
    registrations = {(entry.group, entry.name): entry for entry in product.entry_points}
    expected = {
        ('console_scripts', 'krea2-train'): 'krea2.train:main',
        ('console_scripts', 'krea2-infer'): 'krea2.infer:main',
        ('particle_sliders.image_backends', 'krea2'): 'krea2.lifecycle:BACKEND',
        ('particle_sliders.inference_backends', 'krea2'): 'krea2.native_inference:BACKEND',
    }
    if profile == 'hypergan-hndl':
        expected = {
            ('particle_sliders.image_backends', 'krea2-hndl'): 'krea2_hndl.lifecycle:BACKEND',
            ('particle_sliders.inference_backends', 'krea2-hndl'): 'krea2_hndl.native_inference:BACKEND',
        }
    for key, target in expected.items():
        if key not in registrations or registrations[key].value != target:
            raise RuntimeError('missing or incorrect installed registration: ' + str(key))
    if profile == 'hypergan-hndl':
        from krea2_hndl.runtime import validate_runtime as validate_shared_runtime
    else:
        from krea2.shared import validate_shared_runtime
    report = dict(product=product.version, profile=profile, shared=validate_shared_runtime(),
                  registrations={group + ':' + name: target for (group, name), target in expected.items()},
                  qualification='installation only; pretrained/GPU behavior unqualified', origins={})
    modules = ['krea2', 'particle_sliders', 'particlegan']
    if native:
        from diffusers import Krea2Pipeline, Krea2Transformer2DModel
        from peft import LoraConfig
        # Resolve real classes, including lazy optional vision/encoder imports.
        assert all(isinstance(value, type) for value in (Krea2Pipeline, Krea2Transformer2DModel, LoraConfig))
        modules += ['diffusers', 'peft', 'transformers', 'torchvision', 'safetensors']
        report['native_versions'] = {name: distribution(name).version for name in modules[3:]}
    for name in modules:
        report['origins'][name] = importlib.import_module(name).__file__
    import torch
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise RuntimeError('installation inspection unexpectedly initialized CUDA')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true', help='also resolve optional native classes; no weights')
    parser.add_argument('--profile', choices=('supported', 'hypergan-hndl'), default='supported')
    args = parser.parse_args()
    try:
        report = inspect_install(native=args.native, profile=args.profile)
    except Exception as exc:
        print(json.dumps(dict(status='incompatible', error=str(exc), exception=type(exc).__name__)), file=sys.stderr)
        return 1
    print(json.dumps(dict(status='passed', **report), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
