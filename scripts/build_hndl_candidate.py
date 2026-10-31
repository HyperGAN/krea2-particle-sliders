"""Build a separate local candidate wheel; never install or publish it."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='krea2-hndl-build-') as temporary:
        source = Path(temporary)
        for name in ('krea2', 'krea2_hndl', 'configs'):
            shutil.copytree(root / name, source / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('LICENSE', 'NOTICE', 'LIFECYCLE.md'):
            shutil.copy2(root / name, source / name)
        shutil.copy2(root / 'profiles/hypergan-hndl/pyproject.toml', source / 'pyproject.toml')
        subprocess.run([sys.executable, '-I', '-m', 'pip', 'wheel', '--no-index',
            '--no-deps', '--no-build-isolation', '--wheel-dir', str(output), str(source)], check=True)


if __name__ == '__main__':
    main()
