# Krea2 PG 0.9.1 qualification

Runtime implementation: `fa407b32eef08df73d0966cf7ad64335dd479c15`.
Shared API: repaired immutable `4be0b5b29d2e2c741431f468d71ba8a7c884b724`,
producer wheel SHA256 `2e7621c370e790f65702acf76cfa2f428b393cd3f863245aa26d8a76ff096ada`.
Official ParticleGAN 0.9.1 wheel SHA256
`b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.

The source suite passed74 tests and7 subtests. One additional focused real tiny
adapter-swap test passed after that suite, verifying PEFT unload/Diffusers reinsert,
frozen tensor identity, nonunit alpha and nested negative/zero strength. The
wheel-only native/API suite passed28 tests and7 subtests. Both public presets also
passed exact five-update versus two-plus-three continuation in three separate
processes per preset. `imports.json` proves product and common code resolved only
from installed wheels with isolated Python and no checkout imports. CUDA remained
uninitialized. See `receipt.json` for exact payload hashes and limits.

Source test command (CUDA hidden, Hub/Transformers offline, bytecode disabled):

```sh
python -m pytest -q --junitxml=source-tests.xml
```

Wheel-only qualification uses an isolated venv with the official PG wheel, the
producer core wheel, and the product wheel installed with `--no-deps`. Existing
qualified native runtime dependencies are inherited; no active environment was
modified. Copy `test_pg09_product.py`, `test_native_checkpoint.py`,
`test_infer_live.py`, and `qualify_fresh_process.py` into a separate tests directory,
then run outside both source trees:

```sh
python -I -m pytest -q -c /dev/null --import-mode=importlib /path/to/copied-tests
python -I /path/to/copied-tests/qualify_fresh_process.py full bispf /tmp/krea2-proof
python -I /path/to/copied-tests/qualify_fresh_process.py split bispf /tmp/krea2-proof
python -I /path/to/copied-tests/qualify_fresh_process.py resume bispf /tmp/krea2-proof
# Repeat the three commands with bispf_optimized.
```

These tests train actual tiny Krea2/PEFT CPU modules. The full live-entrypoint test
substitutes only resource loading/rendering, executes native trajectory fields and
shared GAN updates, simulates a crash, and verifies a new continuation directory
without changing the original log. Recovery/render integrity seam tests are labeled
as such. Stand-in images are not quality evidence. No full model GPU job, service,
model publication or merge occurred. Earlier42a2e371 receipts remain provisional;
these replacement receipts do not qualify that superseded source.
