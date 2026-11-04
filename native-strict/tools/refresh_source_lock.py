"""Freeze this new additive native cohort, keeping historical source locks intact."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "9d3ad435daa230fe7746bc0888abad8fcf2a7913"


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
                core_commit="ddb41423bcefc1fe76cd94d0f9b367353b3ea137", base_sources=base, product_sources=own)
    (root / "source-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(base_files=len(base), product_files=len(own), core_files=574)))


if __name__ == "__main__":
    main()
