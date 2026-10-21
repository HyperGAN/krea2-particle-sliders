# Krea2 clean core 0.8.1 adoption proposal

Parent authorized source/light edits against exact candidate
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, following its independent observation
review. The owner's CORE081-HANDOFF.md/build receipt still say rereview pending;
parent's newer handoff states that review passed. No new test slot is authorized.

The preserved derivative parent is `7a50383569de1e343b977a417df972fd6dfca45b`.
This local checkout retains public PR4 `afb3290` and reference draft `f75ca0a`.
No shared-owner checkout, installed environment or active training pin is edited.

## Compatibility and exact lock

Krea uses released PG 0.9.1, adapter GAN/particles=None and continuous native image
fields. Core 0.8.1 remains on that exact PG profile. `krea2/pg09.py` is unchanged;
no loss, penalty, prior, optimizer, scheduler policy or parameter arm is altered.
PG 0.9.3 compatibility is not claimed or selected.

`requirements.txt` and `pyproject.toml` propose the exact clean core commit.
`shared.lock.json` contains the owner's canonical **30-file source_sha256 map**
without generating a derivative-specific subset. Runtime validation now checks
both complete namespaces, including `concept_slider_core`, with distribution
version 0.8.1 and PG 0.9.1. The canonical wheel SHA256 is
`279aaab83da05b6ebeca030dfe115e1ae2232ca4924fe4278ea48103918fbd8d`.
No artifact was installed or built in this phase.

## Adoption delta

- Separate `particle_sliders.inference_backends` registration. Training creates
  a native-only specification; v2 raw/EMA export and independent base/adapter
  preparation use core's public protocol. No game/rows/critic/optimizer is built
  by the inference factory. It accepts local paths, native card, explicit device
  and new prompt text, with no teacher-cache dependency.
- Tiny fixture construction is split into `staged_native` and teacher capture;
  model geometry, seed and original teacher procedure remain unchanged.
- Native encoder/tokenizer identity is captured before encoder release. New
  native cache writers preserve it, and common factories require/verify it.
  Inference identities contain current native sources/versions/frozen components,
  architecture and protocol, without embedding cache or GAN state. Legacy
  sidecars without this new field retain their original provenance; new sidecar
  consumers verify the field when present.
- Explicit top-level CUDA device must agree with the nested CLI index. This
  resolves product preparation RNG discovery using core's existing public input
  seam; raw CLI strings are never assumed to be parsed by core.
- Native request preflight checks output/reference/prompt/geometry constraints
  before preparation. Optional update admission is passed directly to Binding.
  Product observation restores native buffers/modes/checkpoint flags, scheduler,
  sampler fields and position cache outside the adapter network. Core 0.8.1 owns
  registered adapter buffers/RNG, including nonpersistent and buffer-only devices.
- Native model loading remains BF16 with FP32 trainables. Signed strengths, 5D
  field geometry, CFG0, eight steps/mu1.15, conditioning and PEFT ownership stay
  native. HyperGAN presentation and HNDL definition parity remain separate.

## Verification and next admitted slot

Only AST parsing, canonical Git/source-map comparison, metadata/TOML inspection
and `git diff --check` are performed here. No Python runtime imports of torch,
Diffusers, PEFT or models, tests, dependency installs, builds, downloads, GPU work,
public writes or merges are executed. Existing installed dependencies still carry
core 0.7.0; do not relabel the previous 26 passes as 0.8.1 evidence.

Source cases are authored for v2 raw/EMA response without training construction,
explicit CUDA-device preflight without actual CUDA, pure admission refusal with
unchanged state, and native external buffer/cache cleanup on observation failure.
The four original callback regressions are adapted to the new public cleanup and
specification seam. These tests are deferred. The next slot must provide the
exact canonical core 0.8.1 wheel/source distribution, PG 0.9.1 and the existing
compatible Krea native stack; no global-package changes are requested.

## Remaining concrete rough spots and boundaries

1. Core IMAGE-LIFECYCLE.md still says compact inference is outside the slice and
   describes preparation through the cache; it lacks the new native registration,
   preflight/admission/observation API. Its Krea example also suggests arbitrary
   new prompts without explaining encoder lifetime. Parent/core documentation
   should distinguish v1 loading from native-only v2. Product docs now do so.
2. Training `Backend.preflight` runs only when opening; later session renders use
   the product callback's repeat validation. This is wired deliberately; native
   inference has core-level per-request preflight as well.
3. Native encoder identity hashing adds a streaming read of frozen encoder
   tensors. Its cost and full-model memory/placement/sampler behavior remain
   unqualified and require separately admitted native execution. It is never run
   in this source phase or for the CPU fixture's nonexistent pretrained encoder.
4. Native old teacher caches lack encoder/tokenizer identity. They remain at
   their original source cohort; new native training needs a fresh, receipted
   capture. This proposal does not fabricate receipts or bind old checkpoints.
5. Source wiring is untested on this pin. Installed-wheel conformance, actual
   native sampler/decode and GPU RNG/mixed-device behavior, quality, matched-weight
   HNDL, and the HyperGAN reporting bridge remain separate gates.

No denied or uncertain operation occurred in this phase. No parent messaging was
attempted; parent consumes the task output as instructed.


## Completed installed CPU qualification

The parent subsequently admitted one serial CPU slot. Exact runtime source
`c5bf060bd15925ff2e9ea31415ceabf232717699` was archived, built as an sdist, built
from that sdist as a wheel, and installed with the canonical core 0.8.1 wheel and
local released PG 0.9.1 into a separate workspace venv. Existing native libraries
were reused; no global environment, active pin or historic source lock changed.
Tests ran from /tmp with Python isolated mode and source injection disabled.

Results: **37 product integration cases + 6 exact-core dependency-direction
cases = 43 passed**, zero failures/errors/skips. Separate producer and fresh
consumer probes passed nine named checks each, covering installed CLI
capabilities/train/resume/evaluate/refusals, raw/EMA v2 exports, fresh native-only
loading after training inputs were deleted, exact response parity, no game/
teacher/critic/optimizer construction, signed/zero/fractional strengths, RNG
isolation, early preflight rejection and failure cleanup before weight mutation.
Both installed legacy console help entrypoints passed. No production source
correction was needed. All 35 packaged Python/JSON/YAML files and the canonical
30 core source files match the exact archives/wheels/installed package.

Product wheel SHA256:
`890f579113e1812a2189eb743fb3273706e78adec4746e83de97ebb1363db387`.
Product sdist SHA256:
`7c0b8b6e6f65d1b2e004fb8a8e33d2b278fd18b00ed97053ce1642f56805f7d8`.

Resource policy: one CPU/thread, offline mode, 2 GiB RAM cap with a conservative
1.75 GiB process-group RSS stop guard and hard 1.5 GiB data-allocation limits,
16 GiB host reserve, 50 ms monitoring. Observed peak process/group RSS was
795,787,264 bytes (0.741 GiB); minimum host available memory was 49,486,483,456
bytes (46.09 GiB). No guard triggered. CUDA remained uninitialized throughout;
no pretrained model or MOSS preparation was executed. The CPU slot is released.

`evidence/core081-installed/summary.json`, resource/command receipts, logs, JUnit
XML, byte checks, reproducible probes and the exact wheel/sdist retain the proof.
Extracted build source and pytest temporaries were removed after verification.
Earlier source-only receipts remain historical. The core documentation mismatch
and separate native/pretrained, HNDL and HyperGAN gates remain; no new core code
blocker, denial or uncertain action was observed.
