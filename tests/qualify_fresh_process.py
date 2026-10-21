"""Fresh-process CPU continuation oracle using the installed Krea2/core wheels.

Run from outside the checkout with python -I. Only the fixture tests directory is
added to sys.path; product/common code must resolve from installed packages.
Run full, split, then resume for each public preset, using the same output dir.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pg09_product import fixture, equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('full', 'split', 'resume'))
    parser.add_argument('preset', choices=('bispf', 'bispf_optimized'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _, _, game = fixture(args.preset)
    if args.phase == 'resume':
        game.load_state_dict(torch.load(args.output / (args.preset + '-split.pt'), weights_only=True))
    for _ in range(2 if args.phase == 'split' else 3 if args.phase == 'resume' else 5):
        game.step()
    torch.save(game.state_dict(), args.output / (args.preset + '-' + args.phase + '.pt'))
    if args.phase == 'resume':
        from krea2.shared import validate_shared_runtime
        from particle_sliders.pg09 import ConditionalPG09
        import krea2
        equal(torch.load(args.output / (args.preset + '-full.pt'), weights_only=True), game.state_dict())
        assert not torch.cuda.is_initialized()
        result = dict(preset=args.preset, exact_fresh_process_resume=True,
            completed_steps=game.completed_steps, cuda_initialized=False,
            product_source=krea2.__file__, game_source=inspect.getfile(ConditionalPG09),
            runtime=validate_shared_runtime())
        (args.output / (args.preset + '-receipt.json')).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
