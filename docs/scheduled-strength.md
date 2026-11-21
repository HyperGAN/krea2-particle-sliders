# Scheduled strength for Krea2

With an open, owned Krea2 RAW inference session, pass one gain per native
guidance-zero sampling step to `krea2.scheduled_render.render`. Its returned
receipt records the applied strength of every denoiser call.

```python
from krea2.scheduled_render import render

# Eight native steps for this inference owner.
image, receipt = render(owner, prompt=prompt, seed=7, strength=0.5,
                        resolution=768, step_gains=[1.5] * 4 + [0.5] * 4)
```

Use the owner's actual native integration intervals to derive equal-nominal-
strength comparisons. Reuse the saved RAW and match the prompt, seed and native
geometry across scheduled, constant and OFF images. The example shows the
input format. [Shared qualification guide](https://github.com/HyperGAN/particle-sliders/blob/main/docs/strength-schedules.md)

## Warm Amber RAW8 comparison

The native 512×512, eight-step, guidance-zero comparison used the same saved
RAW8 adapter and prompts, seeds and OFF images across four cases. Each arm
had nominal strength 0.5 and the same total strength over the native sigma
intervals. Late-heavy strength gave the highest mean intended-effect gain.

| Schedule | Mean effect ON minus OFF | Images that passed integrity |
| --- | ---: | ---: |
| Constant | −0.000169472 | 4/4 |
| Early-heavy | −0.000403319 | 4/4 |
| Late-heavy | +0.000007518 | 4/4 |

The late-heavy mean improved by +0.000176991 over constant strength. One
of four cases improved individually. To reproduce this late-heavy schedule,
use `strength=0.5`, `resolution=512` and
`step_gains=[0.5] * 6 + [1.4363971969064732, 1.5]` on the same native grid.
[Exact schedule, checkpoint and per-case evidence](../evidence/warm-amber-raw8-scheduled-strength.json)
