"""Regenerate/check the active canonical lock from an exact Git-verified core wheel."""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tomllib
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-repo', required=True, type=Path)
    parser.add_argument('--core-wheel', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    dependency = next(x for x in project['project']['dependencies'] if x.startswith('particle-sliders-core'))
    commit = re.search(r'@([0-9a-f]{40})#subdirectory=', dependency).group(1)
    def source(path):
        return subprocess.check_output(['git', '-C', str(args.core_repo), 'show', commit + ':' + path])
    manifest = tomllib.loads(source('packages/particle-sliders-core/pyproject.toml').decode())
    sources = {}
    with zipfile.ZipFile(args.core_wheel) as archive:
        metadata = Parser().parsestr(archive.read(next(n for n in archive.namelist() if n.endswith('/METADATA'))).decode())
        if (metadata['Name'], metadata['Version']) != ('particle-sliders-core', manifest['project']['version']):
            parser.error('core wheel metadata differs from the exact Git manifest')
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            if path.parts[0] not in ('particle_sliders', 'concept_slider_core') or name.endswith('/'):
                continue
            if '..' in path.parts or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                parser.error('unexpected core wheel path: ' + name)
            value = archive.read(name)
            if value != source('packages/particle-sliders-core/src/' + name):
                parser.error('core wheel differs from exact Git bytes: ' + name)
            sources[name] = hashlib.sha256(value).hexdigest()
    if not sources:
        parser.error('empty core source/data closure')
    prefix = 'packages/particle-sliders-core/src/'
    tracked = subprocess.check_output(['git', '-C', str(args.core_repo), 'ls-tree', '-r',
        '--name-only', commit, '--', prefix], text=True).splitlines()
    expected = {name.removeprefix(prefix) for name in tracked
                if name.removeprefix(prefix).split('/')[0] in ('particle_sliders', 'concept_slider_core')}
    if set(sources) != expected:
        parser.error('core wheel omits or adds files relative to the exact recursive Git closure')
    pg = next(x.split('==', 1)[1] for x in manifest['project']['dependencies'] if x.startswith('particlegan=='))
    lock = dict(schema='krea2-shared-api-v3', repository='https://github.com/HyperGAN/particle-sliders',
        distribution='particle-sliders-core', commit=commit,
        version=metadata['Version'], particlegan=pg,
        reference_wheel_sha256=hashlib.sha256(args.core_wheel.read_bytes()).hexdigest(),
        sources=sources, hash_policy='all package files recursively; no compiled caches',
        qualification='bounded CPU product integration; native/GPU quality unqualified')
    destination = root / 'krea2/shared.lock.json'
    text = json.dumps(lock, indent=2) + '\n'
    if args.check:
        if destination.read_text() != text:
            parser.error('active lock differs from the exact Git-verified wheel closure')
    else:
        destination.write_text(text)
    print(json.dumps(dict(commit=commit, files=len(sources), checked=args.check,
                          reference_wheel_sha256=lock['reference_wheel_sha256'])))


if __name__ == '__main__':
    main()
