# Krea2 common lifecycle

Proposed local shared pin: `fd89e1c780ab5210cd1be98ec3b70977e6349379` / core 0.7.0.
Unpublished; exact-head review is required before draft publication.

This derivative registers backend `krea2` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, EMA, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('krea2', 'new-krea-cpu-v1', dummy=True, steps=4)
with open_session(config) as session:
    session.train()
    response = session.evaluate(strength=-.5)
    session.export('new-krea-run/adapter.pt')
```

This builds the actual random tiny Krea2/PEFT response fixture. It declares
render=false; native rendering remains available in the native configuration.
Krea stays an adapter GAN with particles=None, balanced edit4:preserve1 cache
rows and finite signed strength support.

For locally staged native use, set `cache_file` to the frozen teacher cache and
`native_args` to the existing native CLI option list with local transformer and
skeleton paths. The factory constructs the backend and checks frozen weight
identity. It rejects `--allow_hub`, old resume and adapter-loading flags; use
session resume/artifact arguments. Native loading requires separate resource
admission. The factory never enables a Hub download.

Write `config.as_dict()` as JSON to use `particle-slider capabilities`,
`particle-slider train`, `particle-slider evaluate` and `particle-slider render`.
Run `particle-slider --help` for the common arguments. Use a fresh output
directory for each CLI continuation to preserve old metrics and receipts.
Training packets and inference artifacts have different versioned schemas;
exports cannot resume training. New source pins require a new cohort or a
reviewed explicit migration.

The core sdist includes IMAGE-LIFECYCLE.md with the public factory, capability,
packet and error contracts. Backend dependencies stay isolated: Qwen and Fibo
use different Diffusers Git revisions and are not a joint environment promise.
Use CPU thread limits and offline mode for these fixtures. They do not qualify
pretrained behavior, GPU memory, convergence or image quality.

Fresh derivative audit: see [the bounded migration plan](docs/core070-integration-plan.md).
This source-only follow-up has not run the new conformance cases. The reference
milestone's receipts remain scoped to its exact original heads.

Native rendering currently returns a PIL image and accepts only prompts encoded
from the teacher cache. An unseen prompt raises `UnsupportedOperation` because
the native loader releases its text encoder after preparation. Latent output and
references are rejected; the tiny fixture continues to reject rendering entirely.
Sampling uses a temporary native scheduler and restores the prepared scheduler
on success or failure. Common-session preparation does not write into the legacy
CLI's `--save_dir`; common checkpoint/export calls own their output paths.

The factory checks the cache's model selectors, rank, geometry and turbo sampler
policy before loading the native model, then checks the actual frozen weight
identity. Current product Python/JSON bytes and native library versions are bound
into the new packet identity alongside the immutable teacher-cache identity.
This is a new source cohort, with no automatic old-packet rebinding.

Core `fd89e1c` still needs its inference-only training gate and a preparation RNG
device declaration for Krea's nested native CUDA argument. These are parent/core
fixes; this adapter does not conceal them with product-specific session classes.
Until those fixes and scheduled CPU conformance are reviewed, this is a bounded
integration draft, not a completed qualification or approval for native execution.
