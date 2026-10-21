# Krea2 training formulation

Current `krea2-particle-sliders` live training executes the installed
`particle_sliders.pg09.ConditionalPG09` API with released ParticleGAN **0.9.1**.
The exact shared source pin is in requirements.txt. This product does not vendor ParticleGAN. There is no vendored ParticleGAN,
local GAN loss, product optimizer, or copied shared backend.

`python scripts/train_krea2.py --live` stages the frozen
`jimmycarter/krea2-turbo-bbox` transformer and `krea/Krea-2-Raw` skeleton,
attaches DiT PEFT weights, and records frozen positive and neutral velocities on
identical states and native timesteps. The installed `ImageResponseAdapter`
recomputes the student using neutral conditioning and those actual PEFT leaves.
Native pack/unpack preserves `[1,C,1,H,W]`; model time is `raw_t/1000`.
The installed field critic receives those spatial fields and full native
conditioning projections. Native `guidance_scale=0` means direct conditional;
the shared mathematical CFG coefficient for that operation is 1.

This is an **adapter GAN**. Krea2 has no routed particle bank. The common API
owns the public BISPF/BISPF optimized losses, critic penalty, optimizers, RNG,
balanced row sampling, and full-state checkpoints. The edit contrast RMS from
training rows supplies the response scale. Full balanced cycles contain exactly
four edited rows per preservation row. No velocity MSE, reconstruction loss, or
historical hold coefficient is optimized. Only enhance rows with target equal
to neutral are supported. Scale +1 is the positive teacher, zero disables the
adapter exactly, and negative strengths are extrapolation, not a trained minus pole.

Checkpoints bind actual frozen transformer/VAE tensor hashes, encoded contexts,
teacher cache bytes, local model bytes or exact Hub revisions, source/runtime,
recipe, domain and sampler state. Resume rejects historical MSE state and changed
cohorts. Crash recovery may continue into a new output directory while preserving
the original log. Native and PEFT exports retain alpha/rank metadata.

`--dummy` produces stand-in images and one real shared GAN CPU diagnostic. Its
receipt explicitly says no product weights were trained. CPU tests use real tiny
Krea2 modules and PEFT; they do not establish full model GPU rendering or image quality.

## Historical releases

The published Final Boss and Eldritch LoRAs were produced by the historical
velocity MSE trainer. The old `winning_formulation()` / `stamp.require` metadata
(`particle-gmix-1600-v2`, gmix) and separate CPU smoke did not make its live loop
a GAN. The reproduction pin was
`a119ca1ecd3d5d6c437065839d22739b04f2f4d8`, from
`HyperGAN/particle-sliders`, package `subdirectory=packages/particle-sliders-core`.
Earlier work also used the `HyperGAN/conceptmod` namespace. Those historical
weights and source archives retain their original identity. This change creates
a new training cohort and does not relabel or resume their optimizer state.
`comfy_krea2.py` remains the native Comfy adapter loader.
