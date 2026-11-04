"""Bounded random tiny actual Krea2 CPU software seam, never pretrained/CUDA."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, required=True)
parser.add_argument("--resume", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
from krea2.synthetic import staged_fixture
from krea2_strict.execution import policy, forward_guards
from krea2_strict.game import build_game
from krea2_strict.cache import verify_native_parity
from krea2_strict.provenance import verify_sources
from particle_sliders.pg09 import _hash

with torch.random.fork_rng(devices=[]), patch.object(torch, "manual_seed", torch.random.default_generator.manual_seed):
    native, cache = staged_fixture()
native.strict_execution = dict(numerical=policy(), sources=verify_sources())
native.strict_assets = {"fixture": "random tiny CPU FP32, no pretrained qualification"}
handles = forward_guards(native.model, native.strict_execution["numerical"])
parity = verify_native_parity(native, cache)
game = build_game(native, cache, cache_sha256="deterministic-random-tiny-cpu",
    model_identity={"fixture": native.strict_assets, "strict_execution": native.strict_execution},
    total_steps=2, seed=7, batch_size=2, lr=5e-5, d_lr_mult=2)
if args.resume:
    game.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
while game.completed_steps < args.stop:
    game.step()
packet = game.state_dict()
torch.save(packet, args.output)
for handle in handles:
    handle.remove()
assert not torch.cuda.is_initialized()
print(json.dumps(dict(steps=game.completed_steps, packet_sha256=_hash(packet),
                     all_teacher_rows_exact=parity["all_rows_exact"], cuda_initialized=False)))
