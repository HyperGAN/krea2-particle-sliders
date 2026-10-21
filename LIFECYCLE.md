# Krea2 common lifecycle

Exact tested shared pin: `32f6830f6402e85aa77b6631db64fcbbb2131f6b`,
clean core **0.8.1**, ParticleGAN **0.9.1**. The canonical source map and wheel hash
are consumed directly from the core owner's `core081-build/receipt.json`.
The tested pin is publicly reachable through merged common-core PR144 ancestry.
This source remains a draft for exact-head review, not a package release; see
[qualification status](PUBLICATION-STATUS.md). No PG 0.9.3 profile is downgraded.

The registered image backend owns native loading/conditioning, geometry and
sampler differences. Core owns session updates, EMA, resume, export and cleanup;
ConditionalPG09 owns the unchanged GAN math. Krea remains an adapter GAN with
`particles=None`, actual FP32 PEFT leaves, native `[1,C,1,H,W]` fields, full
conditioning projections and balanced edit4:preserve1 cache rows. Native turbo
policy stays eight steps, CFG=0, mu=1.15. All finite signed strengths remain
supported as executable extrapolation, without a negative teacher.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('krea2', 'new-krea-core081', dummy=True, steps=4, ema_decay=.9)
with open_session(config) as session:
    session.train()
    session.checkpoint('new-krea-run/training.pt')
    session.export_inference('new-krea-run/native-ema.pt', selection='ema')
```

The dummy path constructs an actual random tiny Krea2/PEFT model and internal
teacher fixture. It supports native velocity response, with rendering disabled.
The new native-only registration builds its tiny base/adapter separately, without
capturing teacher responses or constructing a game:

```python
import torch
from particle_sliders.inference import open_inference
with open_inference('new-krea-run/native-ema.pt', operation='response') as inference:
    response = inference.response(prompt='neutral', latent=torch.zeros(1, 4, 1, 4, 4),
                                  timestep=500., strength=-.5)
```

Installed qualification passed on exact runtime source `c5bf060`: **43 pytest
cases passed**, zero failures/skips, plus fresh-process raw/EMA and CLI probes.
All 35 packaged Python/JSON/YAML files match source and installed bytes, with the
canonical 30-file core map verified. See `evidence/core081-installed/summary.json`.
This qualifies bounded CPU fixtures, not pretrained/native GPU behavior. The
prior 26-test pass remains scoped to its original 0.7.0 cohort in
`evidence/core070/`.

For staged native training, supply a local `cache_file`, existing native CLI
`native_args`, and explicit `device='cuda:N'` matching `--device N`. Core can then
include the device in preparation RNG isolation before model allocation. The
factory rejects Hub permission, legacy resume/adapter flags, dummy/card modes,
incorrect precision, and cache model/rank/geometry/turbo-policy mismatches.
The new cache must contain its original `conditioning_identity`, recorded by the
new native writer and compared with actual encoder/tokenizer identity. Old
unreceipted caches are rejected; no identity is fabricated for them.

Optional `render_prompts=[...]` are encoded during native training preparation in
addition to cache prompts. `open_session(..., operation='render',
native_request={'prompt': ..., 'resolution': 768})` validates declared prompts
and geometry before native loading. Already opened sessions accept pre-encoded
prompts, including cache prompts. New prompt text is never silently encoded after
the encoder has been released.

Native-only inference uses `particle_sliders.inference.open_inference`, the v2
artifact, and only local runtime options:

```python
runtime = dict(native_args=local_native_args, device='cuda:0',
               prompts=[new_grounded_prompt])
with open_inference('native-ema.pt', options=runtime,
                    native_request={'prompt': new_grounded_prompt}) as inference:
    image = inference.render(prompt=new_grounded_prompt, strength=-.5, seed=7,
                             resolution=768)
```

Runtime `native_args` must match the exported rank/native card/revisions; local
weight paths may relocate, with actual frozen DiT/VAE and encoder/tokenizer
identity checked before applying exported tensors. `prompts` are new inference
inputs and are not part of a training cache. The factory never calls teacher
capture, `build_game`, a critic or an optimizer. It rejects cache/training options.
Native output is PIL only; references and tiny rendering remain unsupported.
Velocity response takes an explicit native field and raw timestep, never claims
image output and performs no solver step. Native execution still needs separate
resource admission and qualification; this source change runs no native models.

A resource owner may inject `before_update=callback` into the image backend's
`open` call. Core invokes that pure admission callback before each update and
keeps a rejected session usable. Observation hooks preserve the external native
model modes/flags/buffers, native scheduler, sampler attributes and position cache
without cloning pretrained base parameters. Common-session preparation writes
nothing to the legacy CLI `--save_dir`. HyperGAN run presentation stays external
through `on_step`, manifests and artifact receipts; native image sampling stays
a product callback. HNDL spatial critic/native attachment parity remains separate.

V1 training and inference packets retain their original loaders. V2 native-only
artifacts cannot resume training. The 0.7.0 and CDF source cohorts, published
adapters, PR4 payload, historical receipts and active environments are preserved.
No automatic checkpoint/cache rebinding, publication or merge is authorized here.
See [the 0.8.1 adoption record](docs/core081-adoption.md) and the preserved
[original audit](docs/core070-integration-plan.md).
