# Reproduce

Final Boss and Eldritch are trained rank-16 LoRAs with calibrated rank-8
distills. A finished Krea-2 slider includes its checkpoint, matched samples,
training evidence and release validation. Those files are published at
[ntc-ai/krea2-particle-sliders](https://huggingface.co/ntc-ai/krea2-particle-sliders).
The source repository is `krea2-particle-sliders`; model artifacts and logs stay out of Git.

## CPU smoke checks

```bash
python -m pip install -r requirements.txt
python scripts/train_krea2.py --help
python scripts/infer_krea2.py --help
pytest -q
python scripts/train_krea2.py --dummy --save_dir /tmp/krea2-dummy
python scripts/infer_krea2.py --dummy --load_te_lora models/smile-krea2-bbox_lora --save_dir /tmp/krea2-infer
```

That install pins `particle-sliders-core` at
`git+https://github.com/HyperGAN/particle-sliders.git@4340e28bed388d50800c469525b460a108091da0#subdirectory=packages/particle-sliders-core`.
Training calls `winning_formulation()` (gmix, provisional `particle-gmix-1600-v2`).
ParticleGAN is installed transitively and is not vendored in this repo.

The minimal smoke tests and `--dummy` do not download models. Numerical release
tests additionally use torch, safetensors and Diffusers if installed, with a
tiny randomly initialized model. A command without `--dummy` or `--live`
raises before downloading; inference also requires `--load_te_lora`.

## Base and runtime

The pinned transformer is `jimmycarter/krea2-turbo-bbox`, revision
`ec7aa643a4da7e56a08c1778e03e837a2e57a94b`, subfolder
`epoch-14-step-73184/transformer`. Pipeline components come from
`krea/Krea-2-Raw`, revision `6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603`.

The live runtime needs CUDA torch, Diffusers with `Krea2Pipeline` and
`Krea2Transformer2DModel`, transformers, peft, accelerate and safetensors.
The release's `validation/environment.json` and per-slider `evidence/*/training.json`
record tested versions. Training used Python 3.12, torch 2.13.0+cu126,
Diffusers 0.40.0.dev0, transformers 5.15.0 and peft 0.20.0.

```python
import torch
from diffusers import Krea2Pipeline, Krea2Transformer2DModel

transformer = Krea2Transformer2DModel.from_pretrained(
    "jimmycarter/krea2-turbo-bbox",
    revision="ec7aa643a4da7e56a08c1778e03e837a2e57a94b",
    subfolder="epoch-14-step-73184/transformer",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Raw",
    revision="6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.register_to_config(is_distilled=True)  # selects mu=1.15
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(
    "ntc-ai/krea2-particle-sliders",
    weight_name="distilled/native/krea2-eldritch-unit-alpha.safetensors",
    adapter_name="eldritch",
)
pipe.set_adapters("eldritch", adapter_weights=1.0)
image = pipe(
    "An armored knight in a ruined cathedral.",
    height=768, width=768, num_inference_steps=8, guidance_scale=0.0,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("eldritch.png")
```

This example demonstrates loading; exact release replay uses
`release_tools/distill.py`, the cached embeddings and `krea2/attention.py`.
The latter uses memory-efficient masked attention above 4096 tokens for larger
renders. GPU kernels and runtime versions can affect pixels.

## Train on physical GPU 0

```bash
bash scripts/train_final_boss_gpu0.sh
bash scripts/train_eldritch_gpu0.sh
```

Run the jobs sequentially. The scripts set `CUDA_VISIBLE_DEVICES=0`, use
`--live`, and save separate runs under `outputs/`. Each trains rank 16 for 400
updates at 512px and learning rate 5e-5, using six neutral/positive pairs and
two frozen trajectories per pair. Every fifth update uses preservation weight
0.1. Teachers and students see the same latent and timestep. Only attention
LoRA parameters train; base weights and the text encoder stay frozen.
On another machine, set `KREA2_PYTHON` to the Python executable in your CUDA
environment; the default points to this training machine's environment.

See [docs/final-boss.md](docs/final-boss.md) for the objective and
[PROMPTING.md](PROMPTING.md) for bbox syntax and token limits. Sampling uses
8 steps, guidance 0 and mu=1.15. The Raw model's 28-step/CFG 4.5 defaults do
not apply to this turbo transformer. The fruit-bowl prompt is a preservation
control and exposes some style drift.

## Distill, calibrate and publish

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_slider_strength.py outputs/eldritch-krea2-bbox
CUDA_VISIBLE_DEVICES=0 python release_tools/distill.py
CUDA_VISIBLE_DEVICES='' python release_tools/verify_comfy.py /path/to/ComfyUI artifacts/release
python release_tools/build.py
pytest -q
# Commit and push the source changes, leaving outputs/ and artifacts/ ignored.
python release_tools/publish.py --folder artifacts/release
python release_tools/publish.py --folder artifacts/release --publish
```

Run the ComfyUI audit in its own Python environment. Distillation needs both
training runs, their `teacher_cache.pt` files, and the Eldritch strength probes
for historical comparison. [DISTILLATION.md](DISTILLATION.md) describes rank
reduction and alpha selection. The first publication command validates and
packages locally; only `--publish` uploads. Publication requires a clean source
tree, records its commit and every artifact hash, and verifies remote files
and downloaded weight readbacks. Existing released weights are immutable.

For ComfyUI use its single-file bbox model and the ComfyUI adapter exports;
see [COMFYUI.md](COMFYUI.md). The Diffusers runtime uses the transformer
subfolder instead of interpreting a Comfy checkpoint as Diffusers weights.
