# Canonical ParticleGAN 0.9.4 adoption

The ordinary package and active `krea2` training/inference backend use
`particle-sliders-core` **0.12.0** at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137` and released official ParticleGAN
**0.9.4**, source `330eb94ee748cdcad131b0275d8de75fd5a7f96f`.
Python **3.11 or newer** is required. The shared pin includes the merged unified
training dispatcher. This product keeps its established native CLI and backend;
product-specific profile work on the separate feature branch is independent.

`pyproject.toml` owns the exact dependency. The default `requirements.txt`
installs the product and its console/backend registrations. The complete 574-file
core package source/data closure, including licenses and YAML data, was checked
against the immutable Git tree before generating `krea2/shared.lock.json`.
Runtime admission checks every installed package file (excluding interpreter
caches), exact distribution versions and official PG source/profile admission.
The reference core wheel SHA256 is
`0a78d4f88e0c5e5436acfb9f3aca2ae75a7889d9b4f6525cda4a38c42e39257f`.

```bash
python -m pip install -r requirements.txt
python -I scripts/check_install.py
python -I scripts/regenerate_shared_lock.py --check \
  --core-repo /path/to/particle-sliders \
  --core-wheel /path/to/particle_sliders_core-0.12.0-py3-none-any.whl
```

Krea2 remains a legitimate **adapter GAN**, `particles=None`. Its FP32 PEFT
leaves, frozen native base, `[1,C,1,H,W]` velocity fields, balanced
edit4:preserve1 sampling, conditioning projections, calibrated contrast and
native turbo policy (8 steps, CFG=0, mu=1.15) are retained. The shared game owns
the BISPF objectives, optimizer, public penalty and checkpoint semantics.
No dummy particle bank or local replacement loss is introduced.

This is a new source cohort. Previous optimizer/raw/EMA packets retain their
signed original environment and sources; source mismatches must fail closed.
The prior canonical core0.8.1/PG0.9.1 product is available from exact immutable
product source `91ae74b12c123a7241142f134cc4501c5288583a`:

```bash
# Use a separate environment for the prior source/checkpoint cohort.
python -m pip install -r profiles/core081/requirements.txt
```

The separate `krea2-hndl` preview remains at core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, official PG0.9.4,
HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0.
Its lock, native shape checks and qualification gates are preserved. Follow
[the preview instructions](modernization.md) in a separate environment.
Use the exact prior product source above to reproduce previously signed preview
packets; current product source bytes create a distinct cohort even when the
preview core pin is unchanged. This upgrade does not requalify that preview.
Published Final Boss and Eldritch adapters remain historical velocity-MSE
weights; neither this dependency upgrade nor CPU fixtures retrain those weights.

Bounded installed CPU qualification and package evidence are recorded under
`evidence/pg094-canonical/`. Native full-model loading, GPU training, mixed
precision, production caches, rendering and image quality remain unqualified.
