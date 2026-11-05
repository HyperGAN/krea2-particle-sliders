"""Build a separate local candidate wheel; never install or publish it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def restore_pre_lr_runtime(root, destination):
    """Keep the closed HNDL cohort's borrowed runtime bytes unchanged."""
    import hashlib
    import json
    archive = root / 'cohorts/pre-lr-hndl'
    manifest = json.loads((archive / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        data = (archive / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('archived HNDL runtime checksum differs: ' + relative)
        (destination / relative).write_bytes(data)
    (destination / 'krea2/learning_rates.py').unlink(missing_ok=True)
    # Archive ships in the sdist so its profile builder remains reproducible.
    target = destination / 'cohorts/pre-lr-hndl'
    if not target.exists():
        shutil.copytree(archive, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--core-wheel', type=Path,
                        help='optionally verify the original core wheel artifact before building')
    parser.add_argument('--sdist', action='store_true', help='also build the exact candidate source distribution')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.core_wheel is not None:
        lock = json.loads((root / 'krea2_hndl/core.lock.json').read_text())
        if hashlib.sha256(args.core_wheel.read_bytes()).hexdigest() != lock['reference_wheel_sha256']:
            parser.error('core wheel artifact differs from the recorded reference wheel SHA256')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='krea2-hndl-build-') as temporary:
        source = Path(temporary)
        for name in ('krea2', 'krea2_hndl', 'configs'):
            shutil.copytree(root / name, source / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('LICENSE', 'NOTICE', 'LIFECYCLE.md'):
            shutil.copy2(root / name, source / name)
        for name in ('docs/modernization.md', 'scripts/build_hndl_candidate.py',
                     'scripts/install_hndl_candidate.py', 'scripts/check_install.py',
                     'scripts/regenerate_hndl_lock.py', 'profiles/hypergan-hndl/pyproject.toml'):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        (source / 'MANIFEST.in').write_text(
            'include LIFECYCLE.md NOTICE\ninclude docs/modernization.md\n'
            'recursive-include scripts *.py\nrecursive-include profiles *.toml\n')
        shutil.copy2(root / 'profiles/hypergan-hndl/pyproject.toml', source / 'pyproject.toml')
        # Fresh PG0.9.5 cohort uses current source; old closed archives stay untouched.
        with (source / 'MANIFEST.in').open('a') as manifest:
            manifest.write('\nrecursive-include cohorts/pre-lr-hndl *.py *.json\n')
        if args.sdist:
            subprocess.run([sys.executable, '-I', '-c',
                'import sys; from setuptools.build_meta import build_sdist; build_sdist(sys.argv[1])',
                str(output)], cwd=source, check=True)
        subprocess.run([sys.executable, '-I', '-m', 'pip', 'wheel', '--no-index',
            '--no-deps', '--no-build-isolation', '--wheel-dir', str(output), str(source)], check=True)


if __name__ == '__main__':
    main()
