# Reproduce

What this tree can check today, and what it cannot.

## Honest scope

| | In this repo | Not in this repo |
|---|---|---|
| Product docs, starter UNI yamls, fail-closed wrappers | yes | |
| CPU tests that parse those files and never touch the Hub | yes | |
| `jimmycarter/krea2-turbo-bbox` weights | | on the Hub only |
| `train_lora_krea2.py` (train and infer) | wrappers only | [particle-sliders #131](https://github.com/HyperGAN/particle-sliders/pull/131), guide `docs/krea2-turbo-bbox-slider.md` |
| A trained slider, sample grid, or calibrated Comfy strength | | not produced |

Do not describe a GPU run from this checkout as a finished Krea-2 slider.

## CPU smoke (no Hub download)

```bash
python -m pip install -r requirements.txt
pytest -q
```

`pytest` reads the yamls and the docs. It does not import `diffusers` or `torch` and does not call Hugging Face.

Clone the backend from PR #131. This repo does not vendor it.

```bash
git clone https://github.com/HyperGAN/particle-sliders.git
cd particle-sliders
git fetch origin pull/131/head:pr-131
git checkout pr-131
export PARTICLE_SLIDERS_ROOT="$PWD"
export PYTHONPATH="$PARTICLE_SLIDERS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHONPATH=. python conceptmod/textsliders/train_lora_krea2.py --print_card
```

`--print_card` prints the distilled card and exits. It does not download weights. From this product repo, with those exports still set:

```bash
python scripts/train_krea2.py --dummy
python scripts/infer_krea2.py --dummy --load_te_lora models/expression-krea2_lora
```

If `PARTICLE_SLIDERS_ROOT` does not contain `conceptmod/textsliders/train_lora_krea2.py`, both wrappers exit **2** and print that clone hint. They set `HF_HUB_OFFLINE=1` unless you pass `--allow_hub`, and they do not execute `train_lora_krea.py`. A checkout that only has the Raw trainer still exits 2. Infer without `--load_te_lora` also exits 2, so a sample command cannot start an 800-step train.

## Train CLI

The wrapper injects these defaults unless you override them. `--skeleton_model` is #131's name for the Raw pipeline (VAE, text encoder, scheduler). The product yaml calls that same id `pretrained_model.pipeline_id`.

```bash
python scripts/train_krea2.py --dummy \
  --prompts_file configs/krea2/prompts-expression.yaml \
  --config_file configs/krea2/config-expression.yaml
```

which forwards to `conceptmod/textsliders/train_lora_krea2.py` with `PYTHONPATH` set to the particle-sliders checkout:

```bash
python conceptmod/textsliders/train_lora_krea2.py \
  --model_id jimmycarter/krea2-turbo-bbox \
  --transformer_subfolder epoch-14-step-73184/transformer \
  --skeleton_model krea/Krea-2-Raw \
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

## Infer CLI

There is no `infer_lora_krea2.py`. Sampling without a new train is the same entrypoint plus `--load_te_lora`:

```bash
python scripts/infer_krea2.py \
  --prompts_file configs/krea2/prompts-panel-clarity.yaml \
  --load_te_lora models/panel-clarity-krea2_lora
```

That forwards to `conceptmod/textsliders/train_lora_krea2.py` with the same model pin and turbo sample flags (`--skeleton_model krea/Krea-2-Raw`, 8 steps, guidance 0, `mu=1.15`). `--load_te_lora` skips the train loop and writes the sample grid. There is no `models/` checkpoint in git. Inference prompts can be prose or the grounding DSL; see [PROMPTING.md](PROMPTING.md). The fruit-bowl control prompt is a check that the slider did not move an unrelated subject. It is not a teacher.

A local Comfy transformer can stand in for the Hub subfolder. The skeleton still comes from Raw:

```bash
python scripts/train_krea2.py \
  --transformer /path/to/krea2-bbox-turbo-comfy-latest.safetensors \
  --allow_hub \
  --sample_steps 8 --sample_guidance 0.0 --mu 1.15
```

## What a real release would add

A later PR, not this one, would add a Hub weight (outside this git tree), a sample grid at scales `0 / 0.25 / 0.5 / 1.0`, and a one-line change if the packed Comfy file moves past `epoch-14-step-73184`. Update `configs/krea2/model.lock.json` and `krea2_defaults.py` together when that happens.
