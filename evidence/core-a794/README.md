# Coordinated final modern-core migration

Only the active modern Krea2 candidate moved from shared commit
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` to immutable target
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. The prior product head was
`116ebac47fcc929d052290a65e1d5cc0388f7ad6` on existing public draft PR6.
The stable manifest, requirements, core 0.8.1 lock and all stable runtime files
remain byte-identical. Historical review notes and the old candidate lock are
preserved; no umbrella or other owner's files were edited.

The modern dependency uses the exact target Git SHA and the existing
`hypergan-hndl` extra. Core remains at 0.12.0, official PG remains at 0.9.4,
HyperGAN develop remains at `5f368a3770cf5bdae0448995c9f76e29af8049f7`, and
HNDL remains at 0.10.0. No objective, parameter bank, native model, domain or
rendering admission changed. The candidate version remains 0.3.0.dev0; no package
release or model-weight publication is authorized or performed.

## Lock and artifact identity

The lock was regenerated once from the reference core wheel, SHA256
`22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a`.
The generator compares every wheel package file and its complete file set with
the exact Git tree. The resulting recursive closure contains 51 files, including
source, data and licenses. Runtime admission checks all installed package files
recursively, excluding compiled Python caches, and rejects additional data.
It still delegates official PG source admission and HyperGAN/HNDL closure checks
to the unchanged shared contracts. `--check` reproduces the lock without writing.

New training and raw/EMA native artifacts have a different signed source/cohort
identity even though numeric versions and the native protocol name are unchanged.
Actual old preview packets were produced using the unchanged prior environment.
Their hashes and original identity are recorded in `old-preview-identities.json`;
the packets remain local and unchanged. New sessions reject old training state
without changing the complete checkpoint hash. Old raw and EMA packets are
rejected before applying their tensors, with fresh network values and RNG
unchanged. No old packet, checkpoint metadata or teacher cache was relabeled.

## Completed CPU checks

- Installed modern conformance: **16 passed, zero skips**, using copied tests
  outside the checkout and importing product/core packages from the new venv.
  Both public presets execute real tiny Krea2/PEFT updates through the shared
  HyperGAN/HNDL critic. OFF, signed strengths, raw/EMA output and tensors, exact
  resume, frozen-base/adapter ownership, output derivatives, native-only loading,
  geometry refusal and complete state preservation pass. The suite also checks
  recursive source/data tampering, lock-digest tampering, old source packets,
  observation cleanup failure/retry and a close callback failure before product
  cleanup followed by retry. The last case does not qualify arbitrary native
  unload failures.
- Stable product: **54 passed**; stable editable installer: **8 passed**. Their
  dependency declarations, source lock and runtime bytes were not changed.
- Candidate wheel and candidate sdist both build. A wheel rebuilt from the
  sdist has identical runtime package bytes. The sdist includes its reproducible
  build/install/check/lock tools, exact profile and current instructions.
- Fresh CPU wheel installation and a second fresh CPU sdist installation both
  pass normal public dependency resolution, normal build isolation and
  **`pip check`: No broken requirements found.** No dependency override,
  source patch, source checkout injection or `--no-deps` was used in these lanes.
  Exact core and HyperGAN VCS commits appear in PEP610 metadata. Both source/data
  maps match. HyperGAN build provenance reports the exact commit and `dirty:false`.
  All inspected package origins in those two lanes are inside their own venv.
- The fresh lanes use the existing verified official Torch 2.6.0+cpu wheel and
  existing CPU constraints. They install no native model extras or NVIDIA
  packages. The conformance lane separately reuses the previously staged native
  libraries read-only with Torch 2.14.1+cpu; it is not a clean native resolver
  qualification. All three inspected lanes leave CUDA uninitialized.

`commands.json` contains the actual commands and environment settings.
`packaging.json` contains artifact hashes and package-byte parity.
The three provenance files record actual origins, versions, VCS metadata and
source identities. Logs and JUnit receipts retain the exact outcomes. Complete
resolver reports, build logs and local preview files remain under
`/ml2/codex-storage/future-workspaces/2026-10-31/task-9/evidence/core-a794`.

## Remaining gates and review routing

There is no resolver blocker in the clean CPU base lanes. Optional native
dependency resolution, pretrained CUDA execution, mixed precision, sampler and
decoder behavior, quality, native widths and embedded historical variants remain
outside this qualification. The shared owner's synthetic CUDA fixture receipt
does not provide any of those product admissions. This task ran no GPU jobs.

Existing draft PR6 may be updated after these tests. Root must review the exact
product diff with Claude, address any concrete findings and obtain the returned
review before landing. No merge, release, deployment, workflow/access change,
model-weight publication or HNDL optimization was performed.
