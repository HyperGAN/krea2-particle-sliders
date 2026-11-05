> Current adoption: [published PG0.9.5 / core0.13.0](docs/pg095-adoption.md). Versioned earlier receipts below retain their original source scope; none establishes a PG0.9.5 native result.

# Current canonical PG 0.9.4 source upgrade

The canonical source dependency now pins core0.12.0 / official PG0.9.4 at
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`. This creates a new source cohort;
Final Boss/Eldritch weights remain historical velocity-MSE artifacts.
The HNDL preview remains separate at its recorded a794 core pin.
No native model/GPU training, model publication or image-quality qualification
was performed for this upgrade. See [the adoption record](docs/pg094-adoption.md).

## Prior qualification records

The following records retain their original source/cohort meaning.

# Publication and qualification status

The active modern candidate now pins exact shared core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Its installed CPU conformance suite
passes 16 cases with no skips. Fresh wheel and sdist installations pass normal
public dependency resolution, exact source/data admission, import/provenance
checks and `pip check`. The stable product still passes 54 cases and its editable
installer passes 8 cases. See [the coordinated pin receipt](evidence/core-a794/README.md).
Prior preview packets and review notes remain historical. Old preview packets
are rejected without rebinding or tensor mutation. Native model, width, sampler
and quality admission gates are unchanged. This migration ran no GPU jobs and
requires root's Claude actual-diff RETURN review before landing.

PR6 final polish: **54 supported-wheel cases and 10 candidate cases pass**, with
zero skips; the editable installer lane separately passes 8 cases. A fresh CPU
base installation from the exact public core/HyperGAN sources also passes normal
resolution, isolated builds, source admission and `pip check`; see
[clean CPU installation receipts](evidence/public-cpu-install-20261031/README.md).
Earlier [issue-by-issue source/test receipts](evidence/review-fixes-20261031/README.md)
retain the previous selected suites and optional missing-fixture probe.
Supported runtime source bytes and its historical lock remain unchanged from
review head `567fb4044e2b782217d5af09d5e0525253e2c1f9`. Exact candidate core
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` became publicly reachable during the
final handoff; that publication blocker is cleared. Full native dependency
resolution and pretrained/GPU qualification remain unrun.

2026-10-31 inventory continuation: public main at
`afc226f4710db5e4fe9379f22606cd5ccb8ee917` includes merged PR4 and PR5; there
was no open PR at the start of this continuation. The proposed installer repair
and explicit, separate HyperGAN/HNDL candidate are described in
[modernization](docs/modernization.md). The supported pin remains core0.8.1.
The following paragraphs preserve the earlier PR4 publication-stage receipt;
they do not describe the current PR state or qualify the new candidate.

This source is proposed through [the existing draft PR](https://github.com/HyperGAN/krea2-particle-sliders/pull/4).
Its qualified candidate is `b1f42f17f028cab70d3b4517e8f51aa3b5396355`; tested runtime is
`c5bf060bd15925ff2e9ea31415ceabf232717699`. The publication follow-up changes Markdown
only: runtime, tests, dependency declarations, locks and capability gates are
unchanged. No new wheel or test execution is claimed for this docs-only commit.

Selected completed CPU evidence: **43 passed**. 37 installed cases plus6 dependency-direction guards. Adapter-only GAN with particles=None; no routed-bank substitution or pretrained quality claim.
Detailed immutable receipt basename: `summary.json`;
SHA256 `204353bb0c9a4416310b35f7474938036d7ee9d29fb2b56f63da1d643d7ef663`. Original evidence retains its exact tested source,
package/artifact identities, initial failures, scope and resource limits.
These are selected software checks, not a universal product or quality suite.

Tested core: `32f6830f6402e85aa77b6631db64fcbbb2131f6b` / core0.8.1; exact official
ParticleGAN0.9.1 source profile. This core commit is publicly reachable as an
ancestor of [merged common-core PR144](https://github.com/HyperGAN/particle-sliders/pull/144)
at `ee5c2157074681367f8d4c80de1ab7018b6a5d43`. Keep the tested pin: current core0.8.3 is not substituted.
Native libraries remain isolated per product. Existing checkpoints and result
receipts retain their original source/model/architecture/profile/cohort.

Publication is a Git draft update, not a package release, model release,
pretrained qualification or merge approval. Original historical/source-only
assessment notes describe their earlier stage; this receipt index supersedes
only their current CPU/public-source status. HyperGAN reporting and matched-
weight HNDL migrations remain separately coordinated. No unofficial PG88/
shared2960 candidate is activated or included; the official wide-response and
other unsupported-operation gates remain unchanged.
