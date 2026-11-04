"""Read-only CPU asset/budget preflight; never starts a native model load."""
import argparse
import json
from .assets import inspect_assets, memory_budget

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--transformer", required=True)
parser.add_argument("--skeleton", required=True)
parser.add_argument("--full-hashes", action="store_true")
args = parser.parse_args()
print(json.dumps(dict(assets=inspect_assets(args.transformer, args.skeleton, full_hashes=args.full_hashes),
                      budget=memory_budget(), native_gpu_qualified=False), indent=2))
