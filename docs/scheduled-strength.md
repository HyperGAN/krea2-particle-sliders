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
