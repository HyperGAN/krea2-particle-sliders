# Reproduce

What this tree can check today, and what it cannot.

## Honest scope

| | In this repo | Not in this repo |
|---|---|---|
| Product docs, starter UNI yamls, fail-closed wrappers | yes | |
| CPU tests that parse those files and never touch the Hub | yes | |
| `jimmycarter/krea2-turbo-bbox` weights | | on the Hub only |
| `train_lora_krea2` / `infer_lora_krea2` | | not on [particle-sliders](https://github.com/HyperGAN/particle-sliders) `main` yet |
| A trained slider, sample grid, or calibrated Comfy strength | | not produced |

Do not describe a GPU run from this checkout as a finished Krea-2 slider.

## CPU smoke (no Hub download)

```bash
python -m pip install -r requirements.txt
pytest -q
```

`pytest` reads the yamls and the docs. It does not import `diffusers` or `torch` and does not call Hugging Face.

The wrappers refuse to train until the backend file exists. From a clean shell:

```bash
python scripts/train_krea2.py --dummy
python scripts/infer_krea2.py
```

Both exit **2**. Stderr names `jimmycarter/krea2-turbo-bbox`, `epoch-14-step-73184/transformer`, and the intended CLI. They set `HF_HUB_OFFLINE=1` unless you pass `--allow_hub`, and they do not execute `train_lora_krea.py`.

To prove the forward path without a real backend, point `PARTICLE_SLIDERS_ROOT` at a checkout that actually contains `conceptmod/textsliders/train_lora_krea2.py`. A checkout that only has `train_lora_krea.py` still exits 2.

## Intended train CLI

When particle-sliders grows the entrypoint, the wrapper injects these defaults unless you override them:

```bash
export PARTICLE_SLIDERS_ROOT=../particle-sliders

python scripts/train_krea2.py --dummy \
  --prompts_file configs/krea2/prompts-expression.yaml \
  --config_file configs/krea2/config-expression.yaml
```

which forwards to:

```bash
python conceptmod/textsliders/train_lora_krea2.py \
  --model_id jimmycarter/krea2-turbo-bbox \
  --transformer_subfolder epoch-14-step-73184/transformer \
  --pipeline_id krea/Krea-2-Raw \
  --sample_steps 8 \
  --sample_guidance 0.0 \
  --mu 1.15 \
  --hold_weight 0.1 \
  --prompts_file configs/krea2/prompts-expression.yaml \
  --config_file configs/krea2/config-expression.yaml \
  --control_prompt "a bowl of fruit on a table" \
  --dummy
```

The same flags apply to `configs/krea2/prompts-lighting.yaml` and `configs/krea2/prompts-panel-clarity.yaml`.

Load the base the way the Hub card specifies:

```python
tf = Krea2Transformer2DModel.from_pretrained(
    "jimmycarter/krea2-turbo-bbox",
    subfolder="epoch-14-step-73184/transformer",
    torch_dtype=torch.bfloat16,
)
pipe = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Raw",
    transformer=tf,
    torch_dtype=torch.bfloat16,
)
```

`--dummy` must stay free of that download, same rule as `train_lora_krea.py --dummy`. A live run is opt-in with `--allow_hub` and a CUDA box large enough for the 12B transformer plus the parked Qwen3-VL encoder (stock Krea notes say ~48 GB with the text encoder frozen, more if the text encoder trains). This scaffold does not claim that measurement for the bbox transformer.

Starter recipe assumptions, not live results:

- UNI: scale +1 tracks the plus caption, scale 0 tracks the neutral caption, minus is a canary.
- `--hold_weight 0.1` so unused gender tokens do not dominate the loss the way the age card's default `1.0` did on smile-krea.
- `--lm_target v` and `--lora_targets dit` until a gap check on this transformer says otherwise.
- Sample the distilled transformer at 8 steps, guidance 0, `mu=1.15`. The `raw_steps: 28` / `raw_guidance: 4.5` keys in the yaml are the stock Raw card, recorded so nobody confuses the two. `sample.mode` is `turbo`.

## Intended infer CLI

```bash
python scripts/infer_krea2.py \
  --prompts_file configs/krea2/prompts-panel-clarity.yaml \
  --load_lora models/panel-clarity-krea2
```

That forwards to `conceptmod/textsliders/infer_lora_krea2.py` with the same model pin and turbo sample flags. There is no `models/` checkpoint in git. Inference prompts can be prose or the grounding DSL; see [PROMPTING.md](PROMPTING.md). The fruit-bowl control prompt is a check that the slider did not move an unrelated subject. It is not a teacher.

## What a real release would add

A later PR, not this one, would add a Hub weight (outside this git tree), a sample grid at scales `0 / 0.25 / 0.5 / 1.0`, and a one-line change if the packed Comfy file moves past `epoch-14-step-73184`. Update `configs/krea2/model.lock.json` and `krea2_defaults.py` together when that happens.
