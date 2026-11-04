"""Exact local candidate admission, including the shared HyperGAN/HNDL closure."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def validate_runtime():
    import particle_sliders
    lock = json.loads(Path(__file__).with_name('core.lock.json').read_text())
    if lock['schema'] != 'krea2-hypergan-hndl-candidate-v2':
        raise RuntimeError('Krea2 HNDL requires its recursive source/data closure lock')
    if (version('particle-sliders-core'), version('particlegan')) != (lock['version'], lock['particlegan']):
        raise RuntimeError('Krea2 HNDL requires its separate exact core 0.12 / PG 0.9.4 candidate environment')
    root = Path(particle_sliders.__file__).resolve().parent.parent
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('particle_sliders', 'concept_slider_core')
              for p in sorted((root / package).rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    if actual != lock['sources']:
        raise RuntimeError('shared source differs from the exact Krea2 HNDL candidate lock')
    # Delegate official PG byte/profile admission to the verified shared API.
    # A version match alone is insufficient; this host-only probe constructs no model.
    from particle_sliders.pg09 import validate_pg09_domain
    pg = validate_pg09_domain((1, 1, 1, 1, 1), response_domain='continuous_field')
    from particle_sliders.fibo_hndl import verified_builder
    _, identity = verified_builder()
    return dict(core_commit=lock['commit'], core_version=lock['version'],
                particlegan=lock['particlegan'], pg_sources_sha256=pg['pg_sources_sha256'],
                core_validation='recursive installed source/data closure; compiled caches excluded',
                reference_core_wheel_sha256=lock['reference_wheel_sha256'],
                network=identity)


def native_factory(*args, **kwargs):
    from krea2.cuda_train import CudaUNI
    return CudaUNI(*args, runtime_profile='hypergan-hndl-v1', **kwargs)
