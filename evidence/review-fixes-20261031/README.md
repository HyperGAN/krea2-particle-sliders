# Source-review corrections

Review base: `567fb4044e2b782217d5af09d5e0525253e2c1f9` (draft PR6).
The following findings were checked against that source. No shared-core files,
supported runtime Python/JSON files, historical locks or existing packet metadata
were modified. `packaging.json` verifies that every supported runtime file remains
byte-identical to the reviewed head and that the core081 lock matches public main.

| Finding | Resolution and evidence |
|---|---|
| Editable installer versus source-origin assertion | Concrete test defect fixed. Inspection reports PEP610 editable metadata. The test accepts its declared source origin and keeps the wheel-only origin restriction for wheel installs. Supported-wheel suite: **53 passed**. The documented editable environment independently passes all **7 installer cases**. |
| Candidate shares distribution/module identity | Intentional alternative builds, not coinstallable packages. Docs explicitly prohibit upgrading the supported environment with `--pre`/`--upgrade`. New guarded installer requires a fresh venv with no existing Krea2 distribution or importable module; it uses normal pip resolution with no implicit upgrade/dependency override. Fresh-venv admission passes. Existing supported wheel, supported editable and candidate environments are refused. |
| Unused core-wheel hash | Renamed to `reference_wheel_sha256`, explicitly distinguishing archive identity from installed-source admission. Builder `--core-wheel` verifies the reference artifact when supplied. Correct artifact passes both builds; an incorrect artifact is rejected before output creation in the installer tests. |
| PG version versus source validation | Runtime admission now explicitly calls the verified shared `validate_pg09_domain` API, which checks the official PG source map without allocating a model. The returned PG source digest is recorded. A copied PG source tree with a changed policy file and unchanged distribution version is rejected. The game still independently admits the actual native geometry. |
| EMA only checked for finiteness | Both preset cases now compute an expected response after loading the actual EMA parameter values into the training adapter, restore the raw state, verify unchanged checkpoint hash, and compare native-only EMA inference exactly with that response. Raw and EMA parameter values must differ; the exported EMA tensors must equal the recorded EMA values. |
| Missing external legacy packet | Only that optional cross-environment case skips when the variable is absent; setting a bad path fails. A separate required case creates a valid differently-cohorted packet and verifies rejection with unchanged target state. Full qualification ran with the actual old packet supplied: **9 passed, zero skips**. The explicit missing-variable run gives **8 passed, 1 optional skip** and is not counted as full qualification. |
| Package include pattern | No current `krea2` subpackages existed, so the old pattern had not dropped current modules. It would drop future subpackages. The pattern is now `krea2` plus `krea2.*`, with a discovery fixture proving subpackages are retained and sibling `krea2_hndl` is excluded from the supported wheel. Both wheel profiles rebuild from the supported sdist with matching package bytes. |
| Fixed critic H/W | Concrete missing preflight clarified and fixed. The wrapper records channels/singleton-time/H/W, signs it in `extra_repr`, and rejects changed geometry before HNDL graph execution. Separate 3x4 and 4x3 graphs work; equal-area transposed inputs refuse without state mutation. This is a training-critic contract; no pretrained multi-resolution inference claim follows. |

Exact commands, JUnit XML, logs, installation origins, fresh-env admission and
packaging hashes accompany this receipt. Tests use installed wheels with source
injection disabled, plus a separately identified editable lane. Dependencies
remain reused read-only from the previously qualified native runtimes; no clean
full resolver or pretrained/GPU qualification is claimed. Candidate runtime is
still exact core012/PG094/HyperGAN5f368/HNDL010. The original inventory receipt
and its 50/5 results remain historical and unchanged.

The candidate's normal remote installation still requires coordinated publication
of exact shared commit `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`. No new native
training, model downloads, GPU work, release, deployment or merge was performed.
The repository has no checked-in CI/release/deployment workflows. Root owns
review routing and merge decisions.
