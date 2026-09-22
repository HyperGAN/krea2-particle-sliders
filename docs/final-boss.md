# Final boss on turbo-bbox

Run `bash scripts/train_final_boss_gpu0.sh` from this checkout. The launcher uses
the existing `/ml2/ntc-image-studio/.venv-anima/bin/python` environment without
modifying it, and exposes only physical GPU 0. Outputs are in
`outputs/final-boss-krea2-bbox/`.

The bbox transformer is pinned to `ec7aa643a4da7e56a08c1778e03e837a2e57a94b`,
subfolder `epoch-14-step-73184/transformer`. The Raw skeleton is pinned to
`6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603`. Setting the pipeline's
`is_distilled=True` selects mu=1.15; this installed Diffusers pipeline has no
`mu` keyword in its image-generation call. Generation uses 8 steps and CFG 0.

Six pairs cover a knight, robot, sorceress, wolf warrior, prose cave warrior,
and a two-panel knight comic. The neutral and final-boss captions share layout,
background and identity. Grounded captions follow the model's
[prompting guide](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md):
x-first 0–1000 boxes, one element per line, panel elements first, numeric IDs,
quoted text, all contents within their panel. The actual tokenizer and chat
prefix are checked before training; prompts exceeding 507 content tokens fail
instead of silently truncating. Attributes are not prepended.

The CUDA loss trains `adapter(neutral, scale=1)` toward the frozen model's
positive velocity at the same latent and timestep. Frozen neutral and positive
8-step trajectories supply cached states, using two seeds per pair. This is a
small fixed teacher cache, so the held-out bridge prompt and second seed in the
showcase matter when checking generalization. Scale 0 disables the adapter
exactly. Negative captions are not teachers.

Every fifth update preserves the frozen velocity on a mountain lake, bicycle,
or cat prompt, with weight 0.1. The fruit prompt is reserved for verification.
This DiT preservation loss differs from the dummy backend's attribute/token
bookkeeping; the frozen text encoder is not trained. Rank 16 attention LoRA,
AdamW at 5e-5, 400 updates, 512px training, gradient checkpointing. Both base
weights and cached text embeddings use bf16; trainable adapters use fp32.

Each 50-step checkpoint contains a PEFT adapter, a Diffusers LoRA named
`final-boss-krea2-bbox.safetensors`, and optimizer/RNG state. These are Diffusers
and PEFT exports; Comfy loading is not validated by this run. Resume with:

```bash
bash scripts/train_final_boss_gpu0.sh \
  --resume outputs/final-boss-krea2-bbox/checkpoint-0050
```

`status.json` records the phase and last completed step. `train.jsonl` records
loss, baseline teacher gap, adapter change, gradient norm, runtime and GPU
memory. `run.json` records package versions, model pins and CLI arguments.
`prompt_token_counts.json` records every encoded caption. Checkpoints also
contain optimizer state for continuation.

After the first checkpoint, a fixed-seed knight comparison is saved. At the
end, `grid.png` compares scales 0 / 0.5 / 1 at 768px for the knight, cave warrior,
held-out bridge and fruit control. Two 1536px showcase images use the held-out
bridge with seed 1234. Exact prompts and seeds are saved in `samples/metadata.json`.
Inspect these outputs before judging the slider's strength or generalization.

At large resolutions, `krea2/attention.py` explicitly expands grouped key/value
heads so PyTorch's memory-efficient masked attention can run on the RTX A6000.
This avoids the native GQA path's quadratic attention allocation. Sequences of
4096 tokens or fewer retain the stock processor, including this run's 512px
training and 768px comparisons. The large-image processor was compared against
stock masked GQA on GPU (relative RMS difference 0.00227 in bf16).

`scripts/finish_final_boss_samples.py` reloads checkpoint 400, checks its 768px
held-out rendering against the training process's result, and finishes the
1536px samples. The original training source is in `outputs/final-boss-krea2-bbox/source/`;
the recovery source and prior out-of-memory status are preserved separately in
`showcase_recovery/`.
