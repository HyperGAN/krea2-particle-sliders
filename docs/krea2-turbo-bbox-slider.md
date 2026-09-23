# Krea2 turbo-bbox slider

In-repo trainer for [`jimmycarter/krea2-turbo-bbox`](https://huggingface.co/jimmycarter/krea2-turbo-bbox). The GitHub repo is **krea2-particle-sliders**.

This does not change Music or Anima defaults. Stock Krea Raw (CFG 4.5 / 28 steps) is a different card and is not implemented here.

CPU runs use `--dummy`. No Hub weights and no GPU train in CI. This repo does not vendor the transformer.

## Load

The Hub repo is the DiT only. The pinned variant is `epoch-14-step-73184/transformer`. `is_distilled` is a pipeline flag and is not stored on that upload, so `krea2/live.py` sets it when a pipeline is built.

```python
tf = Krea2Transformer2DModel.from_pretrained(
    "jimmycarter/krea2-turbo-bbox",
    subfolder="epoch-14-step-73184/transformer",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Raw", transformer=tf, torch_dtype=torch.bfloat16,
)
```

| input | how it loads |
|---|---|
| default Hub id | `from_pretrained(model_id, subfolder=--transformer_subfolder)` |
| local diffusers directory | that directory, or `<dir>/<subfolder>` when `config.json` is nested |
| Comfy `.safetensors` | read by `load_comfy_krea_transformer` and refused as a diffusers transformer; ComfyUI loads the file itself |

`krea/Krea-2-Raw` is gated. Use `--live` to opt into CUDA training and `--allow_hub` for the first fetch. Without `--dummy` or `--live`, the CLI refuses before downloading. See [the final boss recipe](final-boss.md).

ComfyUI's Krea-2 Turbo template labels guidance-off as **CFG 1.0**. This trainer uses the diffusers convention **guidance_scale 0**. mu=1.15 is the distilled timestep shift, not a CFG scale.

## UNI

| scale | teacher |
|---|---|
| **+1** | `v(z, t, pos)` at CFG 0 |
| **0** | `v(z, t, neu)` |
| **−1** | canary only |

Unused tokens (yaml `attributes`) are bookkeeping. Concept words are not held. Captions are bare: attributes are not prefixed. The fruit-bowl `control_prompt` is verify-only.

Yaml `guidance_scale` is not the train/sample card. The CLI defaults are CFG 0 / 8 / mu 1.15 even if a row still says 4.5. An explicit `--sample_guidance` / `--sample_steps` / `--mu` overrides those defaults.

`--lm_target embed` forces `--lora_targets te`. Sample guidance stays 0.

About 90% of the base model's training used the grounding DSL in the Hub [PROMPTING.md](https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md): plain text, boxes `[x0,y0,x1,y1]` on a 0–1000 grid, **x first**. Captions are passed through unchanged. `configs/krea2/prompts-smile.yaml` includes one grounded row.

## Commands

```bash
python -m pip install -r requirements.txt
python scripts/train_krea2.py --help
python scripts/train_krea2.py --print_card
python scripts/train_krea2.py --dummy --save_dir /tmp/krea2-bbox-dummy
python scripts/infer_krea2.py --dummy --load_te_lora models/smile-krea2-bbox_lora
pytest -q
```

`--print_card` prints the distilled card and exits. It does not download weights.

## Shared formulation

Training calls `winning_formulation()` from the pinned `particle-sliders-core`
package: gmix architecture and the provisional `particle-gmix-1600-v2`
parameters. See [FORMULATION.md](../FORMULATION.md). Hub id, Comfy node, turbo
sample numbers, and prompt cards stay in this repo.

The product entrypoints were first drafted in particle-sliders PR #131
(`train_lora_krea2.py`, `krea2_bbox_live.py`) and now live in `krea2/`.
