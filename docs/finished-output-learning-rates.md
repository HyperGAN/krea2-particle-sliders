# Current finished-output learning-rate guidance

Use the existing **Warm Amber G=0.036 RAW32** checkpoint for this tested setup.
The primary mean intended-effect ON−OFF gain is **0.012435747931400934** over
four examined selection cases, best among five saved matched-budget rates.
A separately completed fresh four-case mean is **0.011814538389444351**. Keep
that fresh result separate from the selection comparison; no new confirmation
or LR search is required. Defaults and released Final Boss/Eldritch weights
retain their original identities.

Select by one number: intended-effect intensity ON minus matched OFF in
finished outputs, averaged equally across the declared cases. Higher wins.
Teacher-response SSE, preservation drift, generic beauty and completion are
not selectors or tiebreaks. Reuse completed searches, saved RAW checkpoints,
outputs and first scores. Set `ema_decay=None` and use RAW current weights for
training, scoring, export and inference; refuse weight EMA.

## Exact choice and scope

The exact option values are:

```json
{"lr": 0.036, "d_lr_mult": 0.5000000000000001, "ema_decay": null}
```

This gives actual D=0.018000000000000002. Preserve the original multiplier
literal instead of recomputing it from rounded printed rates. Krea2 is an
adapter-only game with **no particle bank**; `prior_lr_mult` is rejected, not
set to zero. The inherited package G/base default remains 5e-5 and its D
multiplier remains 1.5; the explicit measured choice does not change them.

RAW32 is the saved checkpoint within the original declared horizon128, not a
new 32-update schedule. Native finished images used 512px, eight steps, CFG0,
mu1.15. The result applies to this existing Warm Amber source/model/RAW cohort
at fixed D. It is not a joint optimum, universal LR, human/statistical
validation, training-prompt membership claim or new package default. The four
fresh prompts/seeds differ from the examined selection cases; their training
or teacher membership was not audited.

The [portable metadata](finished-output-rate.json) retains the two result
records separately, exact options, fresh per-case gains and evidence hashes.
It also retains the fresh3 monitoring exception: the owner recorded actual
entry identities, but Root missed independent live recapture and an ordinary
peer was never registered. Existing active guards and evidence remain stated;
all four cases keep equal weight, with no retrospective peer, rerender or
invented observation.

## Actual strict common API option placement

The installed strict package registers `krea2-native-strict`. Its
[`krea2_strict.lifecycle.config`](../native-strict/src/krea2_strict/lifecycle.py)
factory accepts `lr`, `d_lr_mult`, `ema_decay` and the existing local preparation
options, then returns a common `SessionConfig`. The configuration shape is:

```python
from krea2_strict.lifecycle import config

options = dict(caller_owned_strict_options)  # Verified cache/native args/UUID.
options.update(lr=0.036, d_lr_mult=0.5000000000000001,
               ema_decay=None, steps=128)
session_config = config(cohort=caller_declared_cohort, **options)
```

This shows accepted option placement, not a self-contained reproduction or an
instruction to train again. The original paired cache/conditioning, batch and
stochastic settings, verified local model/skeleton assets, native arguments,
source/numerical identities and resource-owner admission remain required.
See [the strict route](../native-strict/README.md), [lifecycle guide](../LIFECYCLE.md)
and [learning-rate controls](../LEARNING-RATES.md). This document changes no
loader, code, preset, lock or capability gate and does not qualify arbitrary
installs merely by copying the numbers.

The original RAW checkpoint, conditioning and media are separately managed,
not bundled here. Evidence JSON uses basename/byte-count/SHA256 references,
not machine-private paths or download locations. Keep original captured source
and artifact identities; never relabel an old packet. New future recipes need
their own identity, while exact resume retains the original recipe and remaining
budget. Historical SSE screens and CPU/source-readiness receipts retain their
original meaning; none becomes a new selection or confirmation requirement.
