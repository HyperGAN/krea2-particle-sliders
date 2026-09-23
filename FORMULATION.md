# Formulation

`krea2-particle-sliders` (this repo) owns the Krea2 product surface for sliders on `jimmycarter/krea2-turbo-bbox`: which transformer subfolder is loaded, how it sits on the `krea/Krea-2-Raw` pipeline, the turbo sample numbers (8 steps, diffusers `guidance_scale=0.0`, `mu=1.15`, Comfy CFG 1.0), the Comfy node in `comfy_krea2.py`, the UNI prompt cards under `configs/krea2/`, and the train/infer entrypoints in `krea2/` and `scripts/`.

The gmix architecture and the formulation overlay come from `particle-sliders-core`:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@a119ca1ecd3d5d6c437065839d22739b04f2f4d8#subdirectory=packages/particle-sliders-core
```

`krea2/train.py` is the product entry. It locks training with:

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require(stamp.as_dict())
```

At this pin, `winning_formulation()` is gmix plus the provisional `particle-gmix-1600-v2` parameters. When ParticleGAN #38 crowns a full-board winner, the overlay changes in [HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders) and this repo bumps that pin. This repo does not keep a second copy of those knobs.

There is no local copy of `RoutedMLP`, `GradRegularizer`, or `locked_shared`. `--dummy` runs one CPU step through `stamp.bridge()`, `stamp.critic()`, `stamp.losses()`, `stamp.regularizer()`, and `stamp.noise_std_at()`. ParticleGAN is a transitive dependency of `particle-sliders-core`. This repo does not vendor ParticleGAN.

The published Final Boss and Eldritch weights are linear LoRAs. `scripts/train_final_boss_gpu0.sh` and `scripts/train_eldritch_gpu0.sh` still run that DiT objective, and they call `winning_formulation()` before the first update. Music and Anima products keep their own defaults.

Formulation toys, ablations, and pass/fail gates stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod).
