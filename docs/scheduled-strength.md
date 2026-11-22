# Scheduled strength for Krea2

With an open, owned Krea2 RAW inference session, pass one gain per native
guidance-zero sampling step to `krea2.scheduled_render.render`. Its returned
receipt records the applied strength of every denoiser call.

Use the owner's actual native integration intervals to derive equal-nominal-
strength comparisons. Reuse the saved RAW and match the prompt, seed and native
geometry across scheduled, constant and OFF images. [Shared qualification guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)

## Warm Amber RAW8 comparison

The native 512×512, eight-step, guidance-zero comparison used the same saved
RAW8 adapter and prompts, seeds and OFF images across four cases. Each arm
in the timing panel had nominal strength 0.5 and the same total strength over
the native sigma intervals. Late-heavy strength gave the highest mean
intended-effect gain.

| Schedule | Nominal strength | Mean effect ON minus OFF | Images that passed integrity |
| --- | ---: | ---: | ---: |
| Constant | 0.5 | −0.000169472 | 4/4 |
| Early-heavy | 0.5 | −0.000403319 | 4/4 |
| Late-heavy | 0.5 | +0.000007518 | 4/4 |
| Saved constant | 1.0 | −0.000565414 | 4/4 |

The late-heavy mean improved by +0.000176991 over equal-dose constant,
with one of four cases improving. It improved by +0.000572933 over the
saved strength-1 constant, with three of four cases improving. The
strength-1 images used the same RAW8 adapter, native render settings and
exact OFF images. The same instrument scored all four profiles. To reproduce
the late-heavy schedule, split the traversed sigma distance in half. The
experiment uses gain `0.5` in the first half and `1.5` in the second. Each
step gets the weighted average of the gains across its interval. The interval
from sigma `0.5128441453` to `0.3109010756` crosses the midpoint, producing
the penultimate gain of about `1.4364`. This preserves the total nominal
strength of the constant `0.5` arm. The midpoint and bounds are chosen
experiment settings. [Shared interval resolver](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/src/particle_sliders/strength_schedule.py)

Regenerate the preset from the saved sigma grid. Use the same eight-step
owner, prompt, seed, and `resolution=512` for replay.

```python
import json
from pathlib import Path

from krea2.scheduled_render import render

evidence = json.loads(Path("evidence/warm-amber-raw8-scheduled-strength.json").read_text())
sigmas = evidence["actual_sigmas"]
design = evidence["preset_design"]
early_gain, late_gain = design["late_profile_gains"]
widths = [abs(end - start) for start, end in zip(sigmas, sigmas[1:])]
remaining_early = sum(widths) * design["split_fraction"]
step_gains = []
for width in widths:
    early_fraction = min(width, remaining_early) / width
    step_gains.append(early_fraction * early_gain + (1 - early_fraction) * late_gain)
    remaining_early = max(0.0, remaining_early - width)

image, receipt = render(owner, prompt=prompt, seed=seed, strength=0.5,
                        resolution=512, step_gains=step_gains)
```

[Exact schedule, checkpoint and per-case evidence](../evidence/warm-amber-raw8-scheduled-strength.json)

## Activation-dependent allocation

`krea2.adaptive.render` uses the shared CFP threshold or CLABS feedback gate
from *Right Token, Right Time*. Here the generation event is a guidance-zero
Euler interval. Each decision reads the current packed image latents and
normalized timestep before any LoRA applies in that interval. An on decision
applies the selected strength to the existing Krea2 PEFT LoRA. The paper edits
along a fixed residual direction, while Krea2 uses its saved RAW adapter as
that intervention. [Shared equations and calibration](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

```python
from krea2.adaptive import render
from particle_sliders.adaptive_steering import ActivationGate

gate = ActivationGate("cfp", calibrated_threshold)
image, receipt = render(
    owner, prompt=prompt, seed=seed, resolution=resolution,
    strength=selected_strength, gate=gate, alignment=calibrated_alignment,
)
```

Fit or select the alignment readout on a separate calibration split. The
callback receives packed `[1, positions, channels]` latents and the native
normalized timestep tensor. It returns one finite signed score for the
interval. Record the readout fitting data, parameter values and the rule that
maps spatial activations to one score. With the readout and RAW fixed, run
`calibrate_gate_budget` with active steering and a fresh gate per calibration
image. Each rollout aggregates the receipt's fired and eligible intervals.
For CLABS, supply measured projection statistics and the selected controller
parameters, then calibrate its bias. For the matched random control, use
`ActivationGate("random", target_rate, seed=gate_seed)` and omit the readout.

The adaptive receipt carries the actual sigma endpoints, alignment, fire
probability and effective strength per interval. Compare its realized gate
rate and `sigma_weighted_dose` with the controls, then score matched finished
outputs using the frozen intended-effect measure and a separate integrity
check. Keep the Warm Amber RAW8 fixed-schedule cases for evaluation and use
separate cases to fit signals and tune budgets. That panel supplies saved
images and denoiser settings.

## Measured Warm Amber RAW8 CFP replay

The four held-out cases used the archived Warm Amber adapter after eight RAW
updates, with EMA disabled and zero evaluation optimizer updates. The native
sampler used 512×512 images, eight Euler intervals, guidance zero, mu 1.15
and nominal strength 0.5. The CFP gate used the archived 64-dimensional
direction and threshold `-0.03601886751130224`. The readout takes the mean
of the packed BF16 latents over batch and image positions in float64, casts
the pooled vector to float32 and applies the shared signed projection.
The separate calibration panel accepted 17 of 32 firings at a target of
16 with a one-interval tolerance. On the held-out panel CFP fired on 23 of
32 intervals. The constant strength matches the realized sigma-weighted
dose separately for every case. [Exact calibration, direction, sigma grid and RAW identity](../evidence/warm-amber-raw8-cfp-heldout.json)

| Held-out case | Seed | CFP ON minus OFF | Constant ON minus OFF | CFP minus constant |
| --- | ---: | ---: | ---: | ---: |
| Blue bicycle beside a stone wall | 29001 | +0.000421529 | −0.001411031 | +0.001832560 |
| White teapot on a kitchen counter | 29002 | −0.000444029 | −0.000030213 | −0.000413816 |
| Orange cat beside a green door | 29005 | −0.000053709 | +0.000129849 | −0.000183559 |
| Wooden bridge over a quiet stream | 29004 | +0.000643981 | +0.001674267 | −0.001030286 |
| Equal-case mean | | +0.000141943 | +0.000090718 | +0.000051225 |

CFP was ahead of OFF in two of four cases and ahead of the same-dose
constant in one of four. All eight CFP and matched constant images passed
the separate integrity check. These results support an opt-in replay for
this adapter and panel.

Use the source-bound archived Krea2 runtime and its guarded CUDA owner with
the exact RAW already loaded. The archived runtime uses
`jimmycarter/krea2-turbo-bbox` at
`ec7aa643a4da7e56a08c1778e03e837a2e57a94b`. The evidence identifies the
77,730,047-byte RAW packet by SHA-256
`47dabb83bad2026c0e254639fd5a4609725c54252f79d6cb3ad145fbbe3bb2e3`.
Access to that packet and its archived owner is required for GPU replay.
The repository provides the calibration, exact prompts, seeds and output
hashes. The helper checks the native adaptive renderer and shared gate
source against their measured versions. [Measured product source](https://github.com/HyperGAN/krea2-particle-sliders/tree/da5bb91ec7081a7bc058bfa337b914b6d7b57bf9) · [Shared runtime gate](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/packages/particle-sliders-core/src/particle_sliders/adaptive_steering.py) · [Shared calibration source](https://github.com/HyperGAN/particle-sliders/tree/7b26e5b06f5db91e8d3f51a1602d4e459f0f6a22/packages/particle-sliders-core)

From a checkout, call the helper inside the archived owner's existing
context. Pass its native generator as `native` and its admitted guard
as `owner`. Use an original OFF PNG when available.

```python
from scripts.replay_warm_amber_raw8_cfp import replay_warm_amber_raw8_cfp

result = replay_warm_amber_raw8_cfp(
    native, owner=guard, raw_path=raw8_path, case_id="dev0-seed29001",
    archived_off=original_off_path,
)
```

The helper verifies the live RAW weights and the realized gate trace before
rendering the dose-matched constant. It returns the three images and the
per-image archive hash matches. A hash mismatch records a difference from
the measured PNG. [Replay source and exact prompts](../scripts/replay_warm_amber_raw8_cfp.py)


## Warm Amber RAW8 random timing control

The seeded random gate used the same four cases, saved RAW8 adapter, OFF
images, 512×512 native sampler, and nominal strength 0.5 as the CFP and
CLABS comparisons. It drew independently at each Euler interval with
probability 0.5 and fired on 18 of 32 intervals. Each constant matched
its case's realized sigma-weighted dose. All eight new finished images
passed independent integrity checks. [Exact random control, per-case gains, and receipts](../evidence/warm-amber-raw8-random-heldout.json)

| Method | Firings | Gate ON minus OFF | Constant ON minus OFF | Gate minus constant | Cases ahead of constant |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFP | 23/32 | +0.000141943 | +0.000090718 | +0.000051225 | 1/4 |
| CLABS | 22/32 | +0.000022331 | +0.000436411 | −0.000414080 | 0/4 |
| Random | 18/32 | +0.000276118 | −0.000109414 | +0.000385532 | 2/4 |

The table gives equal weight to the four cases. Read each method against
its own constant because realized doses differ across methods.
[CFP evidence](../evidence/warm-amber-raw8-cfp-heldout.json) · [CLABS evidence](../evidence/warm-amber-raw8-clabs-heldout.json)
