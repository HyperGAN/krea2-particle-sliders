# Final frozen CDF consumer gate

This is a narrow installed-wheel qualification of shared core
`cdfb80a76b882b4b73861b12f555263bd64ffeb7` with released ParticleGAN 0.9.1.
The earlier `../pg091` receipts remain scoped to 4be0b5b2. Their broad test
counts do not qualify CDF. No old checkpoint or active source cohort was rebound.

The replacement core changes only `pg09.py`, `pg09_images.py`, and wheel RECORD
relative to the tested 4be wheel. The changes validate G/D Adam clocks and avoid
unnecessary Diffusers metadata lookup for Iris. The held image payload is absent.
Krea2 retains its actual installed Diffusers metadata.

Eight narrow tests passed from `/tmp` under isolated Python using separately
installed product/core/ParticleGAN wheels. These exercise both public presets,
actual 5D Krea2 fields and trainable PEFT leaves, common GAN updates, export,
continuation, live-entrypoint crash/recovery, atomic rejection of negative second
moments and zero/future G/D clocks, followed by valid exact next updates.
Both presets also matched complete state exactly in separate five-step and
2+3-step processes. `origin-and-cohort.json` records installed origins and atomic
rejection of an unchanged real 4be checkpoint before a valid CDF update.

## Reproduce the narrow gate

Build and install the product wheel with the exact frozen core and released
ParticleGAN wheels into a separate environment. Copy `tests/test_pg09_product.py`
and `tests/qualify_fresh_process.py` into an otherwise empty fixture directory.
Use that environment's Python, outside the checkout:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
"$CDF_PYTHON" -I -m pytest -q -c /dev/null --import-mode=importlib \
"$FIXTURE_DIR/test_pg09_product.py" \
-k 'native_shared_gan_updates_and_exact_resume or full_live_dispatch_crash_and_new_directory_resume_on_cpu or shared_adam_history_rejection_is_transactional or cdf_adam_clock_rejection_and_valid_next_update'
```

For each `bispf` and `bispf_optimized`, run the qualifier's `full`, `split`, and
`resume` phases in that order, each in a fresh isolated Python process:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 "$CDF_PYTHON" -I \
"$FIXTURE_DIR/qualify_fresh_process.py" "$PHASE" "$PRESET" "$OUTPUT_DIR"
```

`receipt.json` records wheel hashes and verifies every packaged product runtime
file against this tree. No GPU was initialized. Native full-model rendering,
mixed precision parity and image quality remain coordinator-owned gates.
This receipt does not authorize merging; Ada and the parent route exact heads.
