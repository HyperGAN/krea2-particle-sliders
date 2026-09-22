# Formulation

This repo owns **model integration and recipes** for sliders on `jimmycarter/krea2-turbo-bbox`: which transformer subfolder is loaded, how it sits on the `krea/Krea-2-Raw` pipeline, the turbo sample numbers (8 steps, diffusers `guidance_scale=0.0`, `mu=1.15`, Comfy CFG 1.0), and the UNI prompt cards under `configs/krea2/`.

Formulation toys, ablations, and pass/fail gates stay in [HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod). Do not add them here, and do not put them in ParticleGAN.

ParticleGAN is the core primitive (the particle branch and its regularizers). This product does not fork those primitives and does not vendor a second copy. If a Krea-2 slider needs a new game, the experiment lands in conceptmod, the reusable primitive lands in ParticleGAN, and only the winning recipe comes back here as a config and a docs change.

`train_lora_krea.py` on particle-sliders is the stock Raw / official-Turbo backend. It is sibling work, not the implementation of this product. The bbox finetune needs `train_lora_krea2` (see [REPRODUCE.md](REPRODUCE.md)). Music and Anima trainers keep their own defaults.
