# Canonical core0.12 / official PG0.9.4 CPU qualification

Canonical Krea2 source now pins shared core
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, core0.12.0 and released official
ParticleGAN0.9.4 (`330eb94ee748cdcad131b0275d8de75fd5a7f96f`).
The complete 574-file core source/data closure was compared byte-for-byte with
that Git tree by `scripts/regenerate_shared_lock.py`, then rechecked with
`--check`. Canonical wheel SHA256 is recorded in `krea2/shared.lock.json`.

The installed product wheel passed **60 tests, zero failures/skips**, selected
from test_lifecycle, test_native_inference, test_pg09_product, test_installation,
test_train_infer and test_docs_contract. These run both public presets through
actual random tiny Krea2/PEFT CPU models, exact split/full resume, frozen base
and adapter ownership, OFF/signed responses, raw/EMA native-only inference,
cache/checkpoint gates, complete source/data and official PG admission, and
product/package/CLI contracts. `tests.txt` and `tests.xml` preserve the result.
One stale source assertion was corrected to recognize the already-existing
shared `advance(game, ...)` helper; the runtime update path was not rewritten.

`installed.json` records actual import origins and native versions. The final
venv has no system-site-packages. It reuses the separately staged clean CPU
Torch2.14.1+cpu and dependencies, plus staged Diffusers0.41.0.dev0, read-only via
explicit .pth paths. The product and exact core wheels are installed into this
venv, with PEFT0.21.2 installed separately. This is installed-wheel conformance
with reused dependencies, not a new full native dependency resolution. Pip
check reports no broken requirements. CUDA remains uninitialized.

An initial host-native environment was discarded because its old system
bitsandbytes import initialized CUDA before qualification. No pretrained model,
GPU model operation or training was launched in that environment. The reported
passing lane uses CPU-only Torch and does not expose that system environment.

The merged shared dispatcher was checked with the explicit fixture profile in
this directory: plan, train to step2, periodic checkpoint, resume to step4 in a
new directory and raw/EMA v1+v2 export. The resumed v2 packet was independently
opened through native-only inference, producing a finite signed response of
shape [1,4,1,4,4], with CUDA uninitialized. No product feature branch was merged.
`dispatcher-*.json/txt` retain the plan and results; generated tensors stay local.

The source distribution and its rebuilt wheel package the legacy source recipe
and lock regeneration script. All 35 packaged product/config files match source,
the first tested wheel and the wheel rebuilt from sdist. See `packaging.json`.

Typical exact commands (ROOT and ENV refer to this checkout and the dedicated
CPU environment) were:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$ENV/bin/python" -I \
  -m pytest --import-mode=importlib -q \
  "$ROOT/tests/test_lifecycle.py" "$ROOT/tests/test_native_inference.py" \
  "$ROOT/tests/test_pg09_product.py" "$ROOT/tests/test_installation.py" \
  "$ROOT/tests/test_train_infer.py" "$ROOT/tests/test_docs_contract.py"
"$ENV/bin/python" -I "$ROOT/scripts/check_install.py" --native
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -I -c 'from particle_sliders.session_cli import main; main()' \
  train --config "$ROOT/evidence/pg094-canonical/training.json" \
  --profile "$ROOT/evidence/pg094-canonical/training_profile.json" \
  --out /tmp/krea2-new-run --export-format both --selection ema --no-reporting
```

Full pretrained/native GPU training, mixed precision, production caches, native
rendering and image quality remain unqualified. Existing Final Boss/Eldritch
weights remain historical velocity-MSE artifacts. The separate HNDL preview
retains its pin/gates; its past packets require their original immutable product
source. No release, model publication, deployment or merge is performed.
