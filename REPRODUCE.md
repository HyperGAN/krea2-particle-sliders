# Reproduction and new GAN training

The finished Krea-2 slider releases on the Hub retain their historical MSE
provenance. Reproduce those bytes with their archived source, not this new GAN
trainer. The original package pin was
`a119ca1ecd3d5d6c437065839d22739b04f2f4d8`;
`winning_formulation()` and `particle-gmix-1600-v2` were metadata, not the live
loss. See FORMULATION.md and the release source archive linked from README.md.

For the current source:

```bash
python -m pip install -r requirements.txt
python scripts/train_krea2.py --dummy --save_dir /tmp/krea2-diagnostic
python scripts/infer_krea2.py --help
```

The exact installed API is `particle_sliders.pg09.ConditionalPG09`, from
`subdirectory=packages/particle-sliders-core`, using ParticleGAN 0.9.1.
The diagnostic downloads no weights and trains no Krea2 slider. Tests execute
both public presets through actual tiny Krea2 models on CPU.

A resource owner must admit full model GPU work. After admission, use a new
cohort directory:

```bash
python scripts/train_krea2.py --live --allow_hub \
  --name new-gan-cohort --prompts_file configs/krea2/prompts-final-boss.yaml \
  --model_id jimmycarter/krea2-turbo-bbox \
  --transformer_subfolder epoch-14-step-73184/transformer \
  --skeleton_model krea/Krea-2-Raw \
  --gan_preset bispf_optimized --gan_batch_size 2 \
  --steps 400 --lr 0.00005 --save_dir outputs/new-gan-cohort
```

The default frozen revisions are exact commit SHAs. Native rendering remains
8 steps, CFG off, mu=1.15. CPU tests do not qualify full model mixed precision,
large image attention, native GPU sampler parity, or image quality.

Resume the full GAN checkpoint with the same objective, budget, source, runtime,
and model identities. If a crash left log records after the saved checkpoint,
choose a new directory; the old log remains intact:

```bash
python scripts/train_krea2.py --live [same training arguments] \
  --resume outputs/new-gan-cohort/checkpoint-0050 \
  --save_dir outputs/new-gan-continuation
```

Live inference now reaches the actual native PEFT loader. Use an empty output
directory. New GAN checkpoints validate their model identity and artifact hashes;
historical PEFT folders are explicitly recorded as unverified training provenance:

```bash
python scripts/infer_krea2.py --live \
  --load_te_lora outputs/new-gan-cohort/checkpoint-0400 \
  --prompts_file configs/krea2/prompts-final-boss.yaml \
  --save_dir outputs/new-gan-samples
```

Full GAN state and inference export are separate. Loading an inference export
cannot resume training. There is no routed particle bank or trained negative pole.
