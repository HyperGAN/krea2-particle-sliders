# Krea2 entrypoint audit

Snapshot: 2026-10-17, full source checkout including the previously omitted scripts,
release tools and Comfy loader. Generated `build/`, package metadata and bytecode
are copies, not additional product entrypoints. This describes the proposed source
change; it is not an approved merge, GPU admission or image-quality qualification.

The current dependency files propose `particle-sliders-core` **0.6.0** at
`42a2e371c537180bd189fd4a288b640f9f1c98fc`, with released ParticleGAN **0.9.1**.
That shared pin/lock is provisional while its replacement and exact-head review are
pending. `krea2/shared.py` checks installed distribution versions and module bytes
against `krea2/shared.lock.json`; an old checkout or decorative import is insufficient.

## Training and inference surfaces

| Entrypoint | Actual dispatch and common API execution |
| --- | --- |
| `krea2-train`, `python -m krea2.train`, `scripts/train_krea2.py` | Aliases of `krea2.train.main`. `--live` dispatches `train_cuda`; `--print_card` prints configuration without training. |
| `scripts/train_final_boss_gpu0.sh` | Sets the existing GPU-0 launcher environment and calls `scripts/train_krea2.py --live` with pinned bbox/Raw revisions and Final Boss prompts. It now reaches the GAN path, not the historical original's MSE loop. Requires resource-owner admission before execution. |
| `scripts/train_eldritch_gpu0.sh` | Same live dispatch for Eldritch; uses cached Hub weights unless downloads are separately enabled. Also requires admission. |
| `krea2.cuda_train.train_cuda` | Stages the frozen native model, caches neutral/positive trajectories, then calls `krea2.pg09.build_game` and the installed `ConditionalPG09.step`. No local MSE optimizer or GAN update is substituted. |
| `krea2.pg09.build_game` / `Krea2Rows` | Product binding of native PEFT leaves, cached target fields and conditioning to shared `ImageResponseAdapter`, `image_lora_parameters`, `CampaignFieldCritic` and `ConditionalPG09`. The common API owns losses, penalties, optimizers, sampling, RNG and full-state resume. |
| Train `--dummy`, `krea2.formulation.cpu_gan_step` | Stand-in images plus a real shared GAN step on CPU fixture tensors. The receipt says diagnostic-only and no Krea2 weights trained. It does not qualify native training. |
| `krea2-infer`, `python -m krea2.infer`, `scripts/infer_krea2.py` | Aliases of the sample-only CLI. `--live --load_te_lora` reaches `CudaUNI` and native PEFT sampling; `--dummy` produces stand-ins. `--resume`, conflicting modes and missing explicit mode are rejected. |
| `CudaUNI.generate`, `CudaUNI.trajectory` | Native pipeline rendering and frozen trajectory capture. Rendering performs no GAN updates; shared reversible LoRA scaling preserves alpha, nested strengths and parameter flags. |
| `krea2.live.load_krea2_bbox_pipeline` | Loads the pinned bbox transformer into the Raw skeleton and sets `is_distilled=True`. It is the live trainer's active loader. |
| `krea2.live.load_live_backend` / `LiveKrea2BboxBackend` | Additional callable loading/attachment surface, not the CLI's training loop or a complete alternate training protocol. No `ConditionalPG09.step` call here. |
| `scripts/finish_final_boss_samples.py` | Historical fixed-run checkpoint-400 recovery: reloads `CudaUNI`, compares a 768px render and finishes showcase images. Inference only; writes recovery/sample artifacts. |
| `scripts/probe_slider_strength.py` | Reloads a saved run through `CudaUNI`, renders a strength sweep and a frozen positive-caption reference. Inference only, no adversarial update. |

The GAN target is frozen positive velocity at the student's identical state and
timestep; the student sees neutral conditioning. Native pack/unpack preserves
`[1,C,1,H,W]`, including the singleton temporal axis, rather than reshaping an
arbitrary vector into an image. The shared field critic supports this genuine 5D
domain. Direct native `guidance_scale=0` corresponds to shared mathematical CFG
coefficient 1. Training supports DiT LoRA, enhance rows with target equal to neutral,
and scale +1. Negative strengths are extrapolation, not a trained negative pole.

The new path binds exact Hub revisions or resolved local file hashes, actual frozen
transformer/VAE tensors, encoded conditioning, target-cache bytes, source/runtime
and shared game configuration. Crash recovery can copy the validated log prefix and
cache into a new continuation directory while preserving the original log. Legacy
MSE optimizer state is rejected. Checkpoints contain full shared game state,
inference-only export, PEFT/native LoRA files with alpha metadata and artifact hashes.

New GAN adapter identity validation lives in the `CudaUNI` constructor, so callers
such as the recovery, probe and release runtime receive the same check when loading
a new checkpoint folder. Historical PEFT folders without that receipt retain
unverified training provenance. The sample-only CLI requires separate fresh output
and records the loaded adapter, rather than overwriting training evidence.

## Serving, validation and release surfaces

| Entrypoint | Actual work; no shared GAN training |
| --- | --- |
| Root `__init__.py` → `comfy_krea2.py` (`NTCKrea2BboxLora`) | Registers an inference-only Comfy node. Clones the model, applies local LoRA projection patches and translates uniform native alpha metadata; zero strength bypasses loading. Standard Comfy Load LoRA is another documented consumer of the Comfy export. No training or automatic weight download. |
| `scripts/serve_final_boss.py` | Builds a gallery from existing images/metadata and serves static files with `ThreadingHTTPServer`. There is no model or inference endpoint. |
| `scripts/build_eldritch_gallery.py` | Builds static gallery files and status/preview links. Browser polling reads run status; it does not start training or generation. |
| `scripts/verify_slider_run.py` | Validates existing logs, exported weights, sample metadata and a supplied visual-review statement. Handles historical MSE versus new GAN log keys. It neither trains nor performs an independent human visual review. |
| `release_tools.distill` (`Runtime`, `main`) | Historical two-slider pipeline: loads existing adapters, captures native activations, performs activation-weighted PCA rank reduction and development-set alpha selection, then renders matched samples. Approximation MSE is an evaluation/calibration metric here, not a slider-training loss. |
| `release_tools.refresh_preview` | Uses `release_tools.distill.Runtime` to render matched released-adapter previews and update the local release catalog. Inference only. |
| `release_tools.select_preview` | Same native rendering runtime for candidate previews; waits for a saved visual-selection decision and builds curation evidence. Its `--resume` reuses validated render files, not a training optimizer. |
| `release_tools.lora` | Callable factor reading, compression/error measurements and alpha-bearing PEFT/Diffusers/Comfy exports. No CLI, model training or network publication. |
| `release_tools.build` | Builds cards, comparison assets, evidence, local documentation and a Comfy plugin ZIP from existing release data; also rewrites the repository README. It is a historical release builder, not a GAN trainer or new-cohort quality gate. |
| `release_tools.verify_comfy` | CPU integration against real Comfy Krea topology and patch math: projection mapping, alpha, clone isolation and zero bypass. Does not render the full model. |
| `release_tools.publish` | Validates local release artifacts, creates source/hash archives and, only with `--publish`, uploads to the Hub and verifies remote hashes/readbacks. It does not train or render models. The historical fixed two-slider release schema is not blanket approval to publish a new GAN cohort. |

## Historical provenance and remaining gates

Published Final Boss and Eldritch originals came from the historical velocity-MSE
trainer. Their rank-8 distills compress those linear LoRAs. A
`winning_formulation()` stamp or disconnected CPU smoke never made those original
live updates adversarial. Preserve their archived source, weights and evidence;
current launcher names do not retroactively change their objective. Historical
`docs/final-boss.md`, `docs/eldritch.md` and release-generation templates must be read
with that distinction; regenerated cards must preserve it too.

CPU regressions cover tiny real Krea2/PEFT graphs, both shared presets, adapter
updates, frozen base weights, exact shared-state continuation, inference export,
alpha round trips and reversible sampling state. They do not establish full 12B
mixed-precision GPU training, production native sampler parity, 1536px attention,
full ComfyUI GPU generation, long-run stability or image quality. The custom large
image attention path and historical rendering evidence need qualification for each
new cohort. Resource-owner admission, exact installed-package review, independent
exact-head review and those native/quality gates remain necessary before a new
training/release claim. No GPU job, service, publication or merge was started for
this entrypoint audit.

There are no direct HyperGAN develop/HNDL imports in this product. Optional
develop/HNDL reuse assessment remains with the shared-package owner/task 14; it is
not claimed as executed here. Krea2 already reuses the common native field critic
and 5D response API. An unrelated flat readout would not preserve that geometry and
is not a justified replacement merely to add an import.
