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
