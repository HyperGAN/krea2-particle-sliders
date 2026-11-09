> Current source: released PG0.9.5/core0.13.0. Earlier PG0.9.4 mechanics and versioned readiness/resource receipts retain their original scope. The [current Warm Amber RAW32 finished-output record](../docs/finished-output-learning-rates.md) has its own captured native source/model/RAW identities; it does not relabel those receipts.

# Krea2 strict native source candidate

This separate package registers `krea2-native-strict`. The released `krea2`
route and closed HNDL archives retain their original source bytes and commands.
The inherited G/base default remains `5e-5`; D is derived from the shared recipe
or `lr * d_lr_mult`. The scoped existing Warm Amber result instead uses explicit
G=0.036 and `d_lr_mult=0.5000000000000001`, without changing that default.
This adapter-only game has no bank or prior rate. EMA defaults to disabled;
new configuration explicitly uses `ema_decay=None` and RAW weights throughout.
The preparation details and historical readiness notes below retain their
original source scope; they do not require replaying completed native work.

Use Python3.12 in a fresh environment, actual Torch2.9.1+cu128 and
TorchVision0.24.1+cu128, then install this subproject with dependencies. Its
manifest pins base `fe5f8686b6ef0f3cbcffb9bb36ff61a9ca7efaf4`, core
`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official ParticleGAN0.9.5, Diffusersd5baa4,
Transformers5.15.0 and PEFT0.20.0. Source/origin/runtime locks must pass before
model construction. Candidate PG801 and the old fixture profiles are separate
environments, not interchangeable dependency substitutions.

```sh
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
# From a checkout of this PG0.9.5 source revision:
python -m pip install ./native-strict
```

The [CPU readiness receipt](docs/cpu-readiness-v2.json) records the tested source,
package parity and limits, including the separate inherited-suite failures.

The metadata command imports no ML framework and allocates no model:

```sh
CUDA_VISIBLE_DEVICES='' python -m krea2_strict \
  --transformer /local/turbo-snapshot --skeleton /local/raw-snapshot
```

The transformer root must contain `epoch-14-step-73184/transformer`, pinned to
`ec7aa643a4da7e56a08c1778e03e837a2e57a94b`. The Raw skeleton is pinned to
`6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603`. All selected weights, configuration,
tokenizer and scheduler assets are hashed again before a real load. No Hub
download, alternative weight variant or Comfy file is admitted.

The caller must explicitly set deterministic algorithms with warn-only disabled,
highest FP32 matmul precision, TF32 disabled, reduced-precision reductions
disabled, cuDNN deterministic with benchmark disabled, and only math SDPA
enabled with low-precision math reductions disabled. Native RMSNorm frozen
weights retain the pinned Diffusers FP32 exceptions and are audited explicitly;
all other transformer base weights are BF16. `CUBLAS_WORKSPACE_CONFIG`
must be `:4096:8`; TF32 environment overrides must be absent or0. The package
checks and binds these settings without changing them. Checks run before load,
every update, native model/attention forward, and observation. Loader seed41
saves/restores CPU and only owned visible CUDA0 RNG, including PEFT init.

Every resolution must be a multiple of16 through768; final0 disables the
legacy showcase. Actual latent plus context tokens and each attention input
must stay at most4096. This keeps the inherited attention processor on its
stock path, where the required math SDPA policy applies. Full native contexts
and spatial fields are preserved.

A resource owner must separately admit a single exact GPU UUID and provide
`KREA2_STRICT_GUARDIAN=<parent PID>` and `KREA2_STRICT_PROCESS_CAP_GIB=36`.
The proposed process cap is36GiB plus a5GiB global reserve, requiring41GiB
free at admission. The transformer alone is23.879GiB BF16 and the inherited
loader overlaps it with the8.267GiB text encoder. LoRA, optimizer, critic,
activations, math attention and loading peaks remain unmeasured. This is not a
claim that training fits. Closing releases owned GPU storage to CPU, including
retained game/optimizer aliases; retaining a closed session can keep roughly
24 GiB of base weights in host RAM. The host peak remains unmeasured. No
model/GPU admission is authorized by this document.

For an admitted future cohort, use `StrictCudaUNI` with explicit local native
args including `--device 0 --resolution 512 --sample_resolution 768
--final_resolution 0 --transformer /local/turbo-snapshot --skeleton_model
/local/raw-snapshot`. Capture a **new** cache with `krea2_strict.cache.capture`.
It records current-source OFF conditional velocities from both neutral and
positive trajectories and preservation rows. On every training preparation,
all stored responses must exactly match the actual shared adapter OFF path;
a claimed receipt alone cannot qualify an old cache.

`krea2_strict.lifecycle.config(cache_file=..., native_args=[...],
expected_gpu_uuid=...)` produces the common SessionConfig. Use the core
`open_session`/checkpoint/raw-export API. The native-only inference backend
uses local prompts/args/UUID and refuses old source, numerical, asset or card
identity before base allocation; it never constructs cache rows or a critic.
Old MSE/PEFT folders and default/HNDL packets cannot be relabeled for this
cohort. Keep the existing source-bound update/resume/RAW-response receipts;
do not repeat them for the already measured rate. Selection uses one equal-case
mean intended-effect ON−OFF score in finished outputs as described in
[the LR guide](../LEARNING-RATES.md). Reuse saved RAW checkpoints, completed
outputs and first scores; no new confirmation is required. Teacher-response
SSE, preservation and generic beauty are diagnostics, never selectors or tiebreaks.

Version0.1.1 corrects the strict text encoder identity to the exact
`transformers.Qwen3VLModel` declared by the pinned skeleton model index,
encoder architecture and native Krea pipeline. The generation wrapper and
subclasses remain refused; frozen BF16 and resolved attention checks remain.
All713 source weight keys/shapes match that native class on a meta model.
Tiny random-weight public native loading and the pipeline's12 hidden-state
taps pass CPU checks without loading pretrained tensors or initializing CUDA.

The first full native attempt using version0.1.0 loaded the DiT and encoder,
then refused this correct native class before encoding, teacher capture or
training. Its source, environment and failure/unload receipts are preserved.
At that historical readiness stage, version0.1.1 needed its own
source/environment/cache and bounded native evidence. The current scoped
finished-output record is indexed separately above; no old packet is relabeled
and no proof replay or retry is implied.
