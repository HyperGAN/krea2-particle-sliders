# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Prepare cached conditioning | 1 | 44 | 40 | 512px training;≤32 prompts; ≤512 tokens; B=1; BF16 frozen; FP32 LoRA/critic/activation envelope; no checkpointing |
| Train default512px | 1 | 60 | 40 | 512×512; ≤512 tokens; B=2; BF16 frozen; FP32 LoRA/critic/activation envelope; checkpointed |
| Render1536px tiled VAE | 1 | 44 | 40 | 1536×1536; ≤512 tokens; B=1; BF16 frozen; FP32 LoRA/critic/activation envelope; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`jimmycarter/krea2-turbo-bbox@ec7aa643a4da7e56a08c1778e03e837a2e57a94b`**. cuda_train.Runtime streams the DiT to the chosen GPU, temporarily adds the4B text encoder, caches prompts, then deletes that encoder before backpropagation. The VAE stays CPU until sampling finishes; the DiT remains resident while VAE decode runs. Therefore use max(encode,train,decode), with the actual overlap in each row, rather than summing every phase peak.

These are Diffusers turbo-bbox profiles, not a Comfy FP8-equivalence or quantized-training claim. Prompt length≤512 (507 user-token budget after template),≤32 cached prompts,≤128 response rows. Main training checkpointing is enabled. LargeImageAttention forces efficient masked attention above4096 tokens; at smaller sizes the stock backend may still choose dense math.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- DiT index tensor bytes: **25,640,146,072 bytes** (23.879 GiB).
- skeleton text encoder serialized bound: **8,875,715,136 bytes** (8.266 GiB).
- VAE conservative BF16 bound: **507,591,892 bytes** (0.473 GiB).

The turbo transformer index supplies25,640,146,072 tensor bytes. Main width6144 with48Q/12KV heads;28 main blocks. Four text-fusion blocks have width2560; a conservative32-block activation envelope uses the larger width throughout. The12 text layers are a tensor axis, not12 separately owned copies of each fusion block.

Trainable allowance: **19,402,048 parameters**. 28×16×(6×6144+2×1536) +4×16×(8×2560) +≤200,000 field-critic parameters = ≤19,402,048; no particle bank. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The current `bispf_optimized` PG0.9.5 recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Prepare cached conditioning | 32.145 | 0.248 | 1.043 | 0.000 | 28.737 |
| Train default512px | 23.879 | 20.250 | 1.043 | 0.000 | 28.737 |
| Render1536px tiled VAE | 24.352 | 7.125 | 0.072 | 0.000 | 28.737 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **128 rows**, each with three FP32 native fields, plus **32 cached prompts**. Tensor-only cache allowance: `100,663,296+1,006,632,960` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The auditable inputs and per-profile subtotals are in [source-facts.json](docs/memory/source-facts.json). For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **75,497,472 bytes** (0.070 GiB): bounded Qwen3VL text KV. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [configs/krea2/config-final-boss.yaml](configs/krea2/config-final-boss.yaml)
- [configs/krea2/model.lock.json](configs/krea2/model.lock.json)
- [krea2/live.py](krea2/live.py)
- [krea2/cuda_train.py](krea2/cuda_train.py)
- [krea2/attention.py](krea2/attention.py)
- [krea2/pg09.py](krea2/pg09.py)
- [Captured architecture, published file/index sizes and arithmetic](docs/memory/source-facts.json) retains exact model revisions, metadata SHA256s and direct Hugging Face config URLs. No weights were downloaded or opened for this calculation.
- Optimizer facts: the pinned ParticleGAN0.9.5 `recipes.py` (`bcap`/`bispf_optimized`, lines1435–1451), `optim/dualnorm.py` momentum/state branch, and `optim/direction_blend.py` FP64 normals. These are part of the dependency selected by the repository's existing core lock; their source facts are recorded in the combined calculation report.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.
