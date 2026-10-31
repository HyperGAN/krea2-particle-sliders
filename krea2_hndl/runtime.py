"""Exact local candidate admission, including the shared HyperGAN/HNDL closure."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def validate_runtime():
    import particle_sliders
    lock = json.loads(Path(__file__).with_name('core.lock.json').read_text())
    if (version('particle-sliders-core'), version('particlegan')) != (lock['version'], lock['particlegan']):
        raise RuntimeError('Krea2 HNDL requires its separate exact core0.12/PG0.9.4 candidate environment')
    root = Path(particle_sliders.__file__).resolve().parent.parent
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('particle_sliders', 'concept_slider_core')
              for p in sorted((root / package).rglob('*')) if p.suffix in ('.py', '.json', '.hndl')}
    if actual != lock['sources']:
        raise RuntimeError('shared source differs from the exact Krea2 HNDL candidate lock')
    from particle_sliders.fibo_hndl import verified_builder
    _, identity = verified_builder()
    return dict(core_commit=lock['commit'], core_version=lock['version'],
                particlegan=lock['particlegan'], network=identity)


def native_factory(*args, **kwargs):
    from krea2.cuda_train import CudaUNI
    return CudaUNI(*args, runtime_profile='hypergan-hndl-v1', **kwargs)
