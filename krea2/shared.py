"""Check imported installed API bytes against the reviewed product dependency."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def validate_shared_runtime():
    import particle_sliders
    lock = json.loads(Path(__file__).with_name('shared.lock.json').read_text())
    if lock['schema'] != 'krea2-shared-api-v3':
        raise RuntimeError('Krea2 requires its complete shared source/data closure lock')
    if (version('particle-sliders-core') != lock['version'] or
            version('particlegan') != lock['particlegan']):
        raise RuntimeError('install the exact shared core and ParticleGAN from requirements.txt')
    root = Path(particle_sliders.__file__).resolve().parent.parent
    # Check every Git-verified package file, including data/licenses and the
    # compatibility namespace; compiled interpreter caches are not sources.
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for package in ('particle_sliders', 'concept_slider_core')
              for p in sorted((root / package).rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    if actual != lock['sources']:
        raise RuntimeError('imported shared API source differs from krea2/shared.lock.json')
    # Official PG admission owns source/profile checks, beyond version strings.
    from particle_sliders.pg09 import validate_pg09_domain
    pg = validate_pg09_domain((1, 1, 1, 1, 1), response_domain='continuous_field')
    return dict(distribution=lock['distribution'], version=lock['version'],
                commit=lock['commit'], particlegan=lock['particlegan'],
                pg_sources_sha256=pg['pg_sources_sha256'],
                source_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest())
