"""Install the alternative candidate only in a fresh virtual environment."""
import argparse
from importlib.metadata import distribution, PackageNotFoundError
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys


def check_environment():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('create a fresh virtual environment for the candidate')
    try:
        distribution('krea2-particle-sliders')
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError('Krea2 is already installed; candidate and supported profiles require separate environments')
    if find_spec('krea2') is not None or find_spec('krea2_hndl') is not None:
        raise RuntimeError('Krea2 modules are already available; remove inherited source paths or use a fresh environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        check_environment()
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.check_only:
        print(json.dumps(dict(status='passed', environment=sys.prefix, existing_krea2=False)))
        return
    if args.wheel is None or not args.wheel.is_file():
        parser.error('--wheel must name the built local candidate wheel')
    # Normal pip dependency resolution; no implicit --pre, --upgrade or --no-deps.
    subprocess.run([sys.executable, '-I', '-m', 'pip', 'install', str(args.wheel.resolve())], check=True)


if __name__ == '__main__':
    main()
