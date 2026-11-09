# Learning rates

Current scoped choice: **Warm Amber G=0.036 at RAW32**, with exact
`d_lr_mult=0.5000000000000001` and `ema_decay=None`. The comparison used five
saved matched-budget rates and four examined finished-output cases; mean
intended-effect ON−OFF gain was 0.012435747931400934. A separately completed
fresh four-case mean is 0.011814538389444351. Keep these groups separate;
no further confirmation is required. See the [scope and actual option placement](docs/finished-output-learning-rates.md)
and [portable metadata](docs/finished-output-rate.json). Defaults below are unchanged.

Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **5e-5**, D multiplier = **1.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2); Krea2 retains its explicit base override of 5e-5. This is an adapter-only game; there is no particle bank and `prior_lr_mult` is rejected.

Common interface: put the same keys in session `options`, for example `"lr": 0.00002, "d_lr_mult": 1.5`. The product training CLI accepts `--lr` and `--d-lr-mult`. 

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

Select using one value: intended-effect intensity ON minus matched OFF in
finished outputs, averaged equally across the declared cases. Compare matched
budgets and keep the original initialization, data, stochastic streams,
precision and declared horizon identities. Reuse completed searches, saved RAW
checkpoints, outputs and first scores; do not replay completed updates or add a
new confirmation requirement for the existing choice. Historical teacher/student
relative SSE remains diagnostic only, never a selector or tiebreak.

Preservation/no-edit drift, generic beauty, late/weak-field scores and endpoint
net displacement are diagnostics only; they cannot select, reject, delay or
break ties. The finished-output effect itself is the selector. Retain existing
exact source/model/data/numerical identities and mechanical receipts without
rerunning them. Tiny CPU software checks are not native LR evidence. Use RAW
weights for training, scoring, export and inference; refuse weight EMA.
