# Supported installation and next core cohort

The supported Krea2 cohort remains core 0.8.1 at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b` with official ParticleGAN 0.9.1.
`pyproject.toml` owns this dependency and `krea2/shared.lock.json` verifies its
complete source map. The default requirements installer now installs the product
itself, including backend entry points and console commands. The old file is
preserved byte-for-byte as `requirements-legacy.txt` for dependency-only use.

Create a separate virtual environment before installing. From the repository root:

```bash
python -m pip install -r requirements.txt
python -I scripts/check_install.py
```

Native tiny-model tests and pretrained loading also need `python -m pip install
'.[native,test]'`. The native extra explicitly declares Transformers and
Torchvision: Krea2's Qwen3-VL encoder/image processor imports require both. Match
Torchvision to the selected Torch build; version ranges are installation bounds,
not proof of every possible pairing. The earlier installed CPU receipt used
Diffusers 0.40.0.dev0, PEFT 0.20.0 and Transformers 5.15.0. A matching Diffusers
build must actually expose `Krea2Pipeline` and `Krea2Transformer2DModel`:

```bash
python -I scripts/check_install.py --native
python -I -m pytest -o pythonpath= --import-mode=importlib --confcutdir=tests -q \
  tests/test_lifecycle.py tests/test_native_inference.py \
  tests/test_pg09_product.py tests/test_native_checkpoint.py
```

Use a wheel install and run outside the checkout for release qualification.
The inspection command checks the installed distribution, exact shared source
map, registrations and optional class imports. It never downloads weights,
allocates a model, trains, or selects another core profile.

## Explicit HyperGAN/HNDL candidate

The separate `krea2-hndl` backend executes the shared `HNDLFieldCritic`, built by
the real verified HyperGAN develop builder and HNDL 0.10. Its wrapper removes
only the native singleton time axis at the critic boundary. The shared game and
PG penalty still receive genuine 5D responses. Native PEFT leaves remain the
generator; no particle bank or alternate objective is introduced.

`krea2_hndl/core.lock.json` admits the complete exact joint-candidate source map,
then the core verifies the HNDL source closure and clean HyperGAN build provenance.
Independent native-only exports use backend `krea2-hndl` and protocol
`krea2-hypergan-hndl-native-v1`; source/architecture/cohort mismatches refuse.
Shared product preparation seams permit this explicit binding without replacing
the supported factory's validator, game, or inference protocol.

Build a separate local candidate wheel with the patched build tools installed:

```bash
python -I scripts/build_hndl_candidate.py --output /tmp/krea2-hndl-wheels
```

This creates version `0.3.0.dev0` with its own exact modern dependency metadata
from `profiles/hypergan-hndl/pyproject.toml`. It never installs or publishes the
wheel and leaves the default `pyproject.toml`/requirements/core081 lock intact.
Use a **separate** environment containing the exact reviewed local core wheel,
official PG 0.9.4, HyperGAN commit and HNDL 0.10.0, and compatible native extras.
The core commit is not yet published, so ordinary remote dependency resolution
for this profile remains blocked until coordinated core publication. Do not
change the SHA or disable its source guard to make installation succeed.

After installing that candidate in its separate environment:

```bash
python -I scripts/check_install.py --profile hypergan-hndl --native
```

Create a new `SessionConfig('krea2-hndl', 'new-cohort', dummy=True, steps=4)` for
bounded CPU fixtures; real local native options follow `LIFECYCLE.md`. The shared
`open_session`/`open_inference` entry points discover only this backend in the
candidate wheel. Legacy `krea2-train`/`krea2-infer` console scripts are omitted
from the candidate profile because they select the supported legacy runtime.
Native loading remains local and requires separate resource admission.

Installed candidate tests live in `tests/test_hndl_candidate.py`. They require
`KREA2_LEGACY_PACKET` pointing to an actual core081 training packet created in
its separate supported environment. Run with `-I`, `-o pythonpath=`,
`--import-mode=importlib` and `--confcutdir=tests`; the last option prevents pytest
from collecting the repository's unrelated ComfyUI root initializer.

## Adoption gates still open

The coordinated local candidate is core 0.12 / PG 0.9.4 / HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, joint commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`. It is unpublished and its shared CPU
results do not qualify this Krea2 product. Installing it here must fail the exact
runtime guard. A new product cohort needs all of the following before adoption:

1. A reviewed reachable core commit, its canonical full source map and wheel
   receipt, an exact new product lock and aligned installer metadata. Preserve
   the current lock/receipts and keep old environments available.
2. Installed-package tiny native tests for both public presets, signed/zero
   strength, frozen-base/adapter gradient ownership, unchanged observations,
   exact resume, raw/EMA native-only loading, and rejection without mutation of
   old or mismatched checkpoints. Record actual import paths and dependencies.
3. Independent review of the candidate native HNDL/HyperGAN binding and installed
   CPU evidence. The default product uses shared FieldCritic and adapter GAN,
   `particles=None`; the candidate changes critic implementation and therefore
   requires a new cohort. Do not copy GAN math or introduce a substitute loss.
4. Separately admitted pretrained GPU sampler, mixed precision, memory and
   output checks. No such execution is covered by CPU fixtures. This inventory
   task has no NVIDIA device and does not qualify those checks.

Krea2 keeps genuine `[1,C,1,H,W]` continuous fields. The upstream flat-vector
continuous-domain work has separate ownership; do not flatten/reclassify fields,
widen admission, or use categorical fallback. Historical linear-MSE LoRAs,
training packets and caches keep their original identities. Source-bound packet
rejection is intentional; old checkpoints must use their exact old environment,
with no automatic relabeling or conversion.
