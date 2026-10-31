# Supported installation and next core cohort

The active modern candidate now pins immutable core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, the coordinated transactional
lifecycle fix. Core remains at 0.12 with official PG 0.9.4. HyperGAN develop
remains at `5f368a3770cf5bdae0448995c9f76e29af8049f7`, and HNDL remains at 0.10.0.
The recursive lock is generated from a wheel whose entire source/data closure
is checked against that exact Git tree. Maintainers can verify the lock with:

```bash
python -I scripts/regenerate_hndl_lock.py --check \
  --core-repo /path/to/particle-sliders --core-wheel /path/to/reference-core.whl
```

The prior 4abc preview environment, packets and review receipts retain their old
identities. New training and raw/EMA native packets have a different signed
source/cohort identity, despite unchanged version numbers and native protocol.
Old previews fail closed without applying checkpoint tensors; no rebinding or
automatic conversion is supported. Keep the old environment to consume them.
The supported core 0.8.1 defaults and their lock are unchanged.

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
The default requirements file deliberately uses `-e .[test]` for development;
inspection reports the editable metadata and source origin. Installation tests
accept that declared editable origin. Their wheel lane separately refuses source
checkout imports, so development installation is not reported as wheel evidence.
The inspection command checks the installed distribution, exact shared source
map, registrations and optional class imports. It never downloads weights,
allocates a model, trains, or selects another core profile.

## Explicit HyperGAN/HNDL candidate

The separate `krea2-hndl` backend executes the shared `HNDLFieldCritic`, built by
the real verified HyperGAN develop builder and HNDL 0.10. Its wrapper removes
only the native singleton time axis at the critic boundary. The shared game and
PG penalty still receive genuine 5D responses. Native PEFT leaves remain the
generator; no particle bank or alternate objective is introduced.

`krea2_hndl/core.lock.json` admits the complete recursive exact candidate source/data map,
then the core verifies the HNDL source closure and clean HyperGAN build provenance.
PG's version is checked first, then the shared `validate_pg09_domain` API verifies
the official PG Python source map before any model is allocated. Installed core
validation hashes every package file recursively, including data and licenses,
excluding only compiled Python caches; it does not hash the original archive.
`reference_wheel_sha256` records that original build artifact; supply
`--core-wheel /path/to/exact-core.whl` to the candidate builder to verify the
archive too. Rebuilt wheels may have different archive metadata and are admitted
at runtime only when the complete installed source map matches.
Independent native-only exports use backend `krea2-hndl` and protocol
`krea2-hypergan-hndl-native-v1`; source/architecture/cohort mismatches refuse.
Shared product preparation seams permit this explicit binding without replacing
the supported factory's validator, game, or inference protocol.

Build a separate local candidate wheel with the patched build tools installed:

```bash
python -I scripts/build_hndl_candidate.py --sdist --output /tmp/krea2-hndl-wheels
```

This creates version `0.3.0.dev0` with its own exact modern dependency metadata
from `profiles/hypergan-hndl/pyproject.toml`. It never installs or publishes the
wheel and leaves the default `pyproject.toml`/requirements/core081 lock intact.
Use a **separate** environment containing the exact reviewed local core wheel,
official PG 0.9.4, HyperGAN commit and HNDL 0.10.0, and compatible native extras.
The supported and candidate wheels intentionally share distribution name
`krea2-particle-sliders` and top-level module `krea2`: they are mutually exclusive
alternative builds and cannot be coinstalled. `pip --pre` or `--upgrade` in the
supported environment could replace its distribution and is not an adoption
path. Keep the original environment intact. Use the guarded installer in a fresh
venv, with no inherited Krea2 installation or source paths:

```bash
python -m venv /tmp/krea2-hndl-env
/tmp/krea2-hndl-env/bin/python -I scripts/install_hndl_candidate.py --check-only
# Normal dependency resolution needs a compatible native dependency environment:
/tmp/krea2-hndl-env/bin/python -I scripts/install_hndl_candidate.py \
  --wheel /tmp/krea2-hndl-wheels/krea2_particle_sliders-0.3.0.dev0-py3-none-any.whl
```

The installer refuses an existing distribution or importable Krea2 module and
performs normal pip dependency resolution. It adds no `--pre`, `--upgrade` or
`--no-deps` flag. Direct pip installation bypasses this environment check and
must not be used to replace the supported profile. Build-only commands remain
safe in an existing environment because they install nothing.

For a clean CPU base installation, constrain `torch==2.6.0+cpu`,
`setuptools==84.2.1`, `particlegan==0.9.4` and `hndl==0.10.0` in a separate
constraints file. Run the guarded installer with `PIP_CONSTRAINT` pointing to
that file and `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu` (or use
`PIP_FIND_LINKS` pointing to the verified official CPU wheel). Use the normal
PyPI index for other dependencies and normal build isolation; do not use
`--no-deps` for clean resolution. The generated candidate sdist can also be
installed with normal pip in a second fresh environment and needs no checkout.
This base lane does not install native model extras or qualify pretrained work.
The upstream target's synthetic CUDA software tests do not establish Krea2
native/GPU sampler, model, width or quality qualification. This product migration
runs CPU checks only and preserves all native and domain admission gates.
The exact core commit is now publicly reachable, verified during the final PR6
review-fix handoff. The earlier publication blocker is cleared; a clean full
native dependency resolution has not been run here. Do not change the SHA or
disable its source guard to make installation succeed.

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

The HNDL training critic is resolved for one native channel count and H/W. It
accepts different batch sizes but rejects changed spatial dimensions, including
an equal-area H/W swap, before graph execution. A different training resolution
requires a fresh graph, cache and cohort. This is stricter than the legacy
convolutional FieldCritic. Native-only inference exports have no critic; this
training restriction does not establish or prohibit any unqualified pretrained
multi-resolution rendering behavior.

Installed candidate tests live in `tests/test_hndl_candidate.py`. The optional
external legacy-packet test skips if `KREA2_LEGACY_PACKET` is absent; a supplied
but missing/invalid file fails. Required tests still check a valid incompatible
cohort packet and unchanged target state. Full cross-cohort qualification must
set `KREA2_LEGACY_PACKET` to an actual core081 packet from the separate supported
environment and obtain zero skips; a missing-fixture run is not full evidence.
Run with `-I`, `-o pythonpath=`,
`--import-mode=importlib` and `--confcutdir=tests`; the last option prevents pytest
from collecting the repository's unrelated ComfyUI root initializer.

## Adoption gates still open

The coordinated local candidate is core 0.12 / PG 0.9.4 / HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, joint commit
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. It is now publicly reachable; its
shared CPU results do not qualify this Krea2 product. Installing it into the
supported environment must fail that profile's exact runtime guard. Default
adoption still needs all of the following:

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
