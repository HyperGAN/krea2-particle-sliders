# Fresh Krea2 integration audit and bounded migration plan

2026-10-21. Source inspection only; no tests, installs, builds, native loads or GPU work.

## Base and preserved work

The live GitHub PR inventory contains PR4 (`audit/krea2-shared-gan-20261017`),
open against main at `afb329014caffac825b9c29b9ae85ec23e0d7f72`.
The isolated checkout starts from reference draft
`f75ca0ac8b14bd0ec8f5b818ac1b89f62b5ed152`, which contains that public payload
unchanged followed by three unpublished lifecycle commits. Neither PR4 nor the
reference owner's checkout is edited. Main `0b489f6` is inventory evidence only.
Shared target is exactly `fd89e1c780ab5210cd1be98ec3b70977e6349379`, core 0.7.0;
compact inference and the clean main candidate remain parent-owned follow-ups.

## Actual entrypoints inspected

- Loading: `live.load_krea2_bbox_pipeline` loads bbox DiT + Raw skeleton, frozen
  BF16; `CudaUNI` caches native text encodings, releases the text encoder, attaches
  FP32 DiT PEFT leaves, native large-image attention and checkpoint hooks.
- Training: `krea2-train`, module/script aliases and existing shell launchers reach
  `train.train` -> `cuda_train.train_cuda` -> `pg09.build_game` -> common
  `ConditionalPG09` via `lifecycle.advance`. Legacy envelopes remain original.
- Resume: native `--resume` validates source/model/prompt/run/cache/log contracts
  and PG full state; common sessions use v1 training packets. No format coercion.
- Evaluation: common `session.evaluate` returns native velocity fields; native
  probe/recovery/release tools render from PEFT folders and validate GAN sidecars.
- Export: native checkpoint writer emits PEFT/Diffusers alpha-bearing weights,
  PG state/export and complete artifact hashes; common session emits separate
  raw/EMA inference packets. Comfy consumes native projection exports.
- Inference: sample CLI -> `CudaUNI.generate`; common factory returns the same
  native callback. It currently only encodes teacher-cache prompts, so arbitrary
  new-prompt rendering needs an explicit product encoding design.
- Auxiliary `LiveKrea2BboxBackend` is attachment/loading, not an alternate GAN
  trainer; static galleries/release publication are independent retained tools.

`docs/entrypoint-audit.md` retains the full original PR4 audit and CDF evidence;
its version 0.6.0 statement describes that older cohort, not this fresh draft.

## Native invariants

Krea is an adapter GAN, `particles=None`; actual PEFT A/B leaves are owned once.
The native field remains `[1,C,1,H,W]`, with reversible native packing and full
conditioning projections. Cached positive velocity is evaluated at identical
latent/time; time is raw/1000, mathematical guidance=1 means native CFG=0.
Balanced edit4:preserve1 rows remain unchanged. Native turbo policy stays eight
steps, mu=1.15, CFG=0; all finite signed strengths are supported as executable
extrapolation, no negative teacher is invented. References and text-encoder
training are unsupported. Base BF16/native versus FP32/tiny gates stay distinct.

## Bounded implementation

1. Keep the installed backend registration and exact source lock. Tighten local
   option validation, native render output declarations, and failure cleanup.
2. Bind actual current product/native callback identity into new session packets;
   validate cache geometry/sampler/model selectors against native arguments.
3. Keep renderer integration external through common `on_step`, manifest and
   packet seams. Inventory shared spatial FieldCritic and native PEFT/attention
   as separate HNDL parity boundaries; no new math, reverse imports or DSL.
4. Ask parent to schedule later installed-wheel CPU conformance, one backend at
   a time: exact split resume including metrics/EMA/RNG/coverage; rejected restore
   before mutation; raw/EMA export load; zero/signed/fractional strengths;
   sampler/RNG preservation; native setup/update failure cleanup; dependency and
   unsupported-operation diagnostics. No validation result is claimed here.

## Core rough spots for parent

- `SliderSession.step/train` at this exact milestone do not enforce
  `inference_only`, despite the documented inference packet training prohibition.
  Core must gate this; do not hide it in a Krea callback.
- `Backend.open` chooses RNG devices from `options.device`, whereas Krea's actual
  native CUDA device lives inside `native_args`. Preparation can consume CUDA RNG
  outside that guard. Need a common preflight device declaration/resolver seam.
- Common render capability describes outputs/references/precision but not prompt
  scope. Krea currently releases its encoder and cannot encode unseen prompts.
  Keep an explicit cached-prompt gate/documentation until a product-owned staged
  encoder design is reviewed; do not claim new-prompt inference works.
- Source manifest records factory file only, not arbitrary callback/helper bytes.
  Product can provide explicit source identity through existing model identity,
  but a declared factory identity seam would make this less error-prone.
- Compact inference still reconstructs training cache/game in core 0.7.0. Await
  the parent's interface iteration; do not assemble an alternate loader here.

## Implemented source follow-up

The common factory now validates option types and local cache presence; validates
cache model/rank/resolution/turbo-policy selectors before native construction;
checks frozen weights inside the cleanup boundary; restores caller model modes
and checkpoint attributes; and releases the PEFT attachment once on close.
It binds current product Python/JSON source hashes and native library versions
through the existing model-identity field without changing core or GAN math.

Native rendering declares PIL-only output, explicitly gates uncached prompts,
validates native resolution and uses a temporary native scheduler restored even
on failure. `CudaUNI` gains a default-preserving prompt-receipt switch so common
session preparation writes no files into legacy `--save_dir`; native CLI behavior
retains its original default. Existing launchers, native train/resume/export,
Comfy and release consumers retain their entrypoints and packet formats.

Four focused regression cases were authored for later admission: cache/frozen
identity cleanup, render failure scheduler restoration plus idempotent close,
output/prompt gates, and turbo sampler mismatch before native ownership transfer.
They have **not** been executed. Full split-resume/EMA and prior qualification
receipts still require rerun against this new source cohort. No new wheel or
sdist was built and no installed-runtime validation is claimed.

Light verification completed: Python AST parsing of the three edited Python
files; `git diff --check`; and SHA256 comparison of all 39 core Python/JSON source
members in `shared.lock.json` against `git show fd89e1c:<path>`. This reads the
exact local milestone, never edits its worktree and never imports a model.
The dependency-direction test is deferred with the other scheduled tests; source
inspection finds product-to-core imports and no core edits/reverse imports.

Coordination delivery to parent thread `01a157c0-24ba-7273-9be8-c36fa69e301f`
failed with `thread not found`; no message was delivered by that action. The
plan and rough spots therefore accompany the final task result for automatic
parent notification. This is a failed lookup/delivery, not an approval denial.
