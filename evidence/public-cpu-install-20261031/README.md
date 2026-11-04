# Clean public-source CPU candidate installation

Runtime source remains the reviewed `e43907c1df41df9a29976aa811d690bc08c973d1`.
This follow-up adds two tests and installation receipts; no runtime, profile,
dependency declaration, historical lock or package identity changed.

A new Python3.12 venv with `include-system-site-packages=false` and no dependency
.pth reuse passed the guarded installer. Normal pip resolution fetched exact
public core `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` and HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, built both wheels with normal build
isolation, and installed core0.12.0/PG0.9.4/HNDL0.10.0. No `--no-deps`, metadata
override, source patch, local core/HyperGAN wheel, or pin widening was used.

CPU-only selection used the previously verified official Torch2.6.0+cpu wheel,
SHA256 `59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91`,
through `PIP_FIND_LINKS` and an external CPU constraint. Setuptools84.2.1,
PG0.9.4 and HNDL0.10.0 were also constrained to the existing qualified versions.
Other declared dependency ranges resolved normally; all actual versions are in
`provenance.json`. Neither supported manifests nor historical locks changed.

Results: dry-run resolution passed; both VCS wheel builds passed; actual guarded
installation passed; `pip check` returned **No broken requirements found**.
PEP610 records both exact commits. HyperGAN build provenance is `dirty:false`.
Installed core/PG/HNDL/builder source admission passed, and both registered
backend objects load. Every inspected package origin is inside the new venv.
Torch reports `2.6.0+cpu`, no CUDA build, and CUDA remains uninitialized.

No Diffusers, PEFT, Transformers, Torchvision, NVIDIA packages, model weights,
or pretrained models were installed or loaded. This closes the clean CPU base
installation check, not optional native dependency resolution or native/GPU
qualification. There is no resolver blocker in this bounded lane.

The small polish tests also pass: **54 supported-wheel cases, 10 candidate
cases, 8 editable-installer cases**. The new actual base-interpreter subprocess
must produce the intended fresh-venv error, rather than an unrelated failure.
The new geometry case initializes two training updates and verifies the complete
game/optimizer/sampler/EMA checkpoint hash is unchanged after rejection.

`commands.json` contains actual install/resolution/inspection commands and CPU
environment settings. Build hashes, provenance, exact polish commands, JUnit and
logs accompany this receipt. Complete pip logs/reports and built wheels remain
under `/ml2/codex-storage/future-workspaces/2026-10-31/task-9/evidence/public-cpu-install`.
No merge, release, deployment or weight publication was performed.
