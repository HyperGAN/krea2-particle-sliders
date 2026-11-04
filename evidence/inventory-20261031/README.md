# Inventory integration qualification, 2026-10-31

Base: public `afc226f4710db5e4fe9379f22606cd5ccb8ee917`. Original checkout
`/ml2/krea2-particle-sliders` remained clean and untouched. No repository/ancestor
AGENTS.md or relevant local .agents instructions were present. Work used a separate
clone and branch with separate virtual environments. Public main matched the
inventory, and GitHub's open-PR list was empty before publication.

Installed supported wheel: **50 passed** (37 native/lifecycle cases, 6 docs,
3 config, 4 installer/lock/admission cases). Installed HyperGAN/HNDL candidate:
**5 passed**. There were no skips in either selected final run. Exact commands,
JUnit results, installation origins and wheel source maps accompany this file.

The candidate executes actual tiny random Krea2/PEFT models, the real released
PG0.9.4 game and real HNDL0.10 graph through the clean verified HyperGAN develop
builder. Both public presets pass exact split/full resume, frozen-base and
unique adapter ownership, nonzero adapter derivatives, zero-strength frozen
response, state-preserving observations, raw/EMA native-only loading without
training construction, and actual core081 packet rejection without mutation.
Matched weights preserve the shared FieldCritic's outputs, response gradients
and second derivatives through the singleton-axis wrapper. Source-tamper
admission is an explicit negative software test, not native GPU evidence.

Supported runtime: Python3.11 / Torch2.11.0+cu128, Diffusers0.40.0.dev0,
PEFT0.20.0, Transformers5.15.0, core0.8.1/PG0.9.1. The candidate uses
Python3.12 / Torch2.14.1+cpu, Diffusers0.40.0.dev0, PEFT0.21.2,
Transformers5.22.1, core0.12/PG0.9.4/HNDL0.10. Core source maps, full
HyperGAN builder provenance/closure, imports and registrations pass inspection.
CUDA stayed uninitialized; the executor has no `/dev/nvidia*`.

Both virtual environments reuse existing native libraries read-only. Supported
core and PG were installed locally; the modern core/PG were installed into its
own venv, with native dependency libraries reused by explicit .pth paths.
This is installed-package CPU conformance, not a fresh full dependency-resolution
or pretrained/GPU qualification. The installer itself also passed in a third
separate venv with already staged exact core/PG, using `--no-index --no-deps
--no-build-isolation -r requirements.txt`, then checking registrations.

Both wheel source maps match the final package bytes. The supported sdist builds
back into a wheel with matching package bytes and includes the candidate build
inputs. The supported wheel excludes the candidate package/registrations;
candidate metadata explicitly selects its exact new core/HNDL extra. The old
requirements file and supported source lock remain byte-for-byte unchanged.

Initial runs are retained in the task's external evidence folder: one supported
pytest collection failed on the unrelated root ComfyUI initializer before
`--confcutdir=tests` was supplied; one candidate test used the wrong packet-key
path and was corrected. The selected final runs above pass. No GAN math, recipe,
width gate or historical packet metadata was changed.

Remaining gates: the joint core commit is unpublished, so normal remote
resolution of the candidate cannot succeed yet. Root owns coordinated core
publication, Claude/Ada review and merge. Pretrained CUDA memory, precision,
sampler/decoder, image quality and production native caches remain unrun.
No model weights, deployment, package release or merge occurred here. There is
no active repository workflow that would automatically release these source
changes. Existing source-bound packets keep their original environments; helper
source changes intentionally require a new source cohort rather than relabeling.
