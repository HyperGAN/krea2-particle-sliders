# Eldritch on turbo-bbox

Run `bash scripts/train_eldritch_gpu0.sh` from this checkout. It uses the same
pinned model, environment, GPU 0, training method and sampling settings as
[Final Boss](final-boss.md). Existing model files must be cached; pass
`--allow_hub` if downloading them on a new machine.

The six neutral prompts and their boxes are identical to the Final Boss run.
Only the positive descriptions change: iridescent chitin, clusters of eyes,
curling tentacles, organic plating and asymmetrical alien anatomy. Scene,
background, character IDs, panels and dialogue remain matched between the
neutral and positive captions. The actual tokenizer checks the 507-token
budget before training.

This trains a separate rank-16 adapter from the frozen bbox base. Final Boss
is not loaded into the Eldritch run. The same seeds and verification prompts
allow direct comparisons between the two sliders. The 400-update run uses
512px training, learning rate 5e-5, two teacher trajectory seeds per pair,
and a preservation update every fifth step with weight 0.1.

Outputs are under `outputs/eldritch-krea2-bbox/`. Each checkpoint exports
`eldritch-krea2-bbox.safetensors` for Diffusers and
`adapter_model.safetensors` with its PEFT configuration. Use 8 inference steps,
guidance 0, and the distilled shift mu=1.15. In this installed pipeline the
distilled flag selects the shift; `mu` is not an image-call keyword.

The first checkpoint produces a 50-update preview. Final evaluation renders
strengths 0, 0.5 and 1 for a cathedral knight, prose cave warrior, held-out
volcanic bridge, and unrelated fruit control. Two further bridge renders at
1536px use seed 1234. Scale zero disables the adapter exactly. The large-image
attention processor is installed before training, so the 1536px samples use
memory-efficient masked attention.

`python scripts/build_eldritch_gallery.py` adds `/eldritch/` to the existing
gallery on port 8780. During training it exposes progress and intermediate
previews. Once `verification.json` and `RESULTS.md` are present, rebuilding
publishes the final comparisons and checkpoint-400 download. The HTTP server
continues to listen on `0.0.0.0`; no restart is needed for new static files.

To inspect stronger settings and compare against the positive-caption teacher,
run the following after training releases GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_slider_strength.py \
  outputs/eldritch-krea2-bbox
```

Use the same CUDA environment as training. This reloads checkpoint 400, renders
the neutral knight and held-out bridge at strengths 1, 1.5 and 2, and renders
the positive knight caption with the adapter disabled. Outputs and exact
prompts are under `strength-probes/`. Rebuilding the gallery includes this
additional comparison once all probe renders are present.

In the completed run, strength 1 mainly changes armor, while 1.5 adds curling
appendages and a more alien silhouette on both inspected scenes. Strength 2
is more fragmented. Start around 1–1.5 for a visible corruption effect; extra
eyes remain weak. The unrelated fruit control at strength 1 stays recognizable
but shifts toward a painterly style. See the local gallery and `RESULTS.md`
for the actual comparisons and validation results.

The published original embeds alpha 24 at rank 16, so released strength 1
has the gain of this raw checkpoint's strength 1.5. Its rank-8 distill embeds
alpha 12. Start both released versions at strength 1; the recommendation above
applies to the original uncalibrated training checkpoint.
