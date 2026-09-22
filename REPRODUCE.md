# Reproduce

What this tree can run today, and what it cannot.

## Honest scope

| | In this repo | Not in this repo |
|---|---|---|
| Train / infer entrypoints (`krea2/train.py`, `scripts/train_krea2.py`, `scripts/infer_krea2.py`) | yes | |
| Distilled card, load plan, CPU UNI loop, Comfy node | yes | |
| `jimmycarter/krea2-turbo-bbox` weights | | on the Hub only |
| A finished GPU slider or calibrated Comfy strength | | not produced |

Do not describe a GPU run from this checkout as a finished Krea-2 slider. The GitHub repo is `krea2-particle-sliders` (renamed from `krea2-concept-sliders`).

## CPU smoke (no Hub download)

```bash
python -m pip install -r requirements.txt
python scripts/train_krea2.py --help
python scripts/infer_krea2.py --help
pytest -q
python scripts/train_krea2.py --dummy --save_dir /tmp/krea2-dummy
python scripts/infer_krea2.py --dummy --load_te_lora models/smile-krea2-bbox_lora --save_dir /tmp/krea2-infer
```

`pytest` and `--dummy` do not import `diffusers` and do not call Hugging Face. A command without `--dummy` raises before any download. Infer without `--load_te_lora` exits with an argparse error so a sample command cannot start the train loop.

There is no sibling checkout and no path variable that points at particle-sliders. The trainer that was drafted on particle-sliders #131 lives in this repo now.

## Train CLI

```bash
python scripts/train_krea2.py --dummy \
  --prompts_file configs/krea2/prompts-expression.yaml \
  --config_file configs/krea2/config-expression.yaml
```

Defaults, unless you override them:

```bash
python scripts/train_krea2.py \
  --model_id jimmycarter/krea2-turbo-bbox \
  --transformer_subfolder epoch-14-step-73184/transformer \
  --skeleton_model krea/Krea-2-Raw \
  --sample_steps 8 \
  --sample_guidance 0.0 \
  --mu 1.15 \
  --hold_weight 0.1 \
  --lora_targets dit \
  --lm_target v \
  --dummy
```

`--skeleton_model` is the Raw pipeline (VAE, text encoder, scheduler). Product yamls also record that id as `pretrained_model.pipeline_id`.

Load the base the way the Hub card specifies. `krea2/live.py` is that loader. This CLI does not call it unless a future revision opts into a CUDA step; today `--dummy` is the executed loop.

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

Live packages (not installed by `requirements.txt`, and not used by the CPU tests): a current `diffusers` with `Krea2Pipeline` / `Krea2Transformer2DModel`, plus `torch`, `peft`, and `safetensors`.

Starter recipe, not a live result:

- UNI: scale +1 tracks the plus caption, scale 0 tracks the neutral caption, minus is a canary.
- The card CFG is 0 even if a yaml row still says `guidance_scale: 4.5`. An explicit `--sample_guidance` overrides the card.
- `--hold_weight 0.1` for bare captions.
- Sample the distilled transformer at 8 steps, guidance 0, `mu=1.15`. `raw_steps: 28` / `raw_guidance: 4.5` in the yaml are the stock Raw card, recorded so the two are not mixed.

## Infer CLI

```bash
python scripts/infer_krea2.py \
  --dummy \
  --prompts_file configs/krea2/prompts-panel-clarity.yaml \
  --load_te_lora models/panel-clarity-krea2_lora
```

`--load_te_lora` skips the train loop and writes `samples/final_meta.json` plus one PNG per neutral caption and the fruit-bowl control, at scales `0 / 0.25 / 0.5 / 1.0`. On `--dummy` the adapter path is recorded and not downloaded. There is no checkpoint in git.

Inference prompts can be prose or the grounding DSL. See [PROMPTING.md](PROMPTING.md). The fruit-bowl line is a check that the slider did not move an unrelated subject. It is not a teacher.

## Comfy single file

`krea2-bbox-turbo-comfy-latest.safetensors` is what Comfy loads. The diffusers path uses the transformer subfolder. `load_comfy_krea_transformer` reads a local safetensors file and refuses to pretend it is already a diffusers transformer. See [COMFYUI.md](COMFYUI.md).
