# Formulation

`krea2-particle-sliders` (this repo; renamed from `krea2-concept-sliders`) owns **model integration and recipes** for sliders on `jimmycarter/krea2-turbo-bbox`: which transformer subfolder is loaded, how it sits on the `krea/Krea-2-Raw` pipeline, the turbo sample numbers (8 steps, diffusers `guidance_scale=0.0`, `mu=1.15`, Comfy CFG 1.0), the UNI prompt cards under `configs/krea2/`, and the train/infer entrypoints in `krea2/` and `scripts/`.

Formulation toys, ablations, and pass/fail gates stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). Do not add them here, and do not put them in ParticleGAN.

ParticleGAN is the core primitive (the particle branch and its regularizers). This product does not fork those primitives and does not vendor a second copy. If a Krea2 slider needs a new game, the experiment lands in conceptmod, the reusable primitive lands in ParticleGAN, and only the winning recipe comes back here as a config and a docs change.

The distilled trainer is in this repo (`krea2/train.py`, `scripts/train_krea2.py`). It is not a wrapper around particle-sliders. Music and Anima products keep their own defaults.
