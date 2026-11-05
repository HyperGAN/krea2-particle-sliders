"""Freeze this new additive native cohort, keeping historical source locks intact."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "fe5f8686b6ef0f3cbcffb9bb36ff61a9ca7efaf4"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT.parent), *args])
    paths = git("ls-tree", "-r", "--name-only", BASE, "--", "krea2").decode().splitlines()
    base = {p.removeprefix("krea2/"): sha(git("show", BASE + ":" + p)) for p in paths}
    root = ROOT / "src/krea2_strict"
    own = {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*")
           if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "source-lock.json"}
    lock = dict(schema="krea2-strict-native-source-lock-v1", base_commit=BASE,
                core_commit="3dd672a7f9b8c169daef90fd63e4bb42b9ddc423", base_sources=base, product_sources=own)
    (root / "source-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(base_files=len(base), product_files=len(own), core_files=577)))


if __name__ == "__main__":
    main()
