"""Check imported installed API bytes against the reviewed product dependency."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def validate_shared_runtime():
    import particle_sliders
    lock = json.loads(Path(__file__).with_name('shared.lock.json').read_text())
    if (version('particle-sliders-core') != lock['version'] or
            version('particlegan') != lock['particlegan']):
        raise RuntimeError('install the exact shared core and ParticleGAN from requirements.txt')
    root = Path(particle_sliders.__file__).resolve().parent
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
              if p.suffix in ('.py', '.json')}
    if actual != lock['sources']:
        raise RuntimeError('imported shared API source differs from krea2/shared.lock.json')
    return dict(distribution=lock['distribution'], version=lock['version'],
                commit=lock['commit'], particlegan=lock['particlegan'],
                source_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest())
