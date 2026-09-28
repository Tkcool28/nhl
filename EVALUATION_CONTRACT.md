# NHL SOG — Pre-Holdout Evaluation Contract

Frozen 2026-09-28. This contract defines how the v1.2 baseline will be
judged on the sealed 2025-26 holdout. It was written **before** the holdout
was opened. Do not revise it after seeing holdout results; if the holdout
suggests the contract itself was wrong, record that as a finding, not as
an edit.

## Evaluation slices

Two populations are always reported side by side:

1. **all-skater** — every skater-game in the holdout frame (diagnostic).
2. **production-eligible** — `toi_l10 >= 13.0` minutes (the frozen
   eligibility rule; see `DIAGNOSTIC_BUCKETS.md`). This is the population
   the production/market-logging path will concern.

Neither slice may be redefined after holdout opening.

## Primary metrics (per slice)

Computed on P(over) pooled across lines 1.5 / 2.5 / 3.5 / 4.5, plus
line-level reporting:

- **Brier score** (pooled; primary probability-quality metric)
- **Log loss** (pooled)
- **Calibration / ECE** (10-bin ECE, pooled)
- **AUC** (pooled; reported, not gated)
- **Expected-SOG error** — MAE and RMSE of `mu_gbm` vs actual shots
  (line-independent; reported overall and by diagnostic bucket)
- **EB comparison** — every metric above is also computed for the
  empirical-Bayes baseline (`mu_eb` + its per-line alphas) on the
  identical rows. The baseline wins a metric only by beating EB on it.

## Line-level evaluation

For each of 1.5, 2.5, 3.5, 4.5: AUC, Brier, log loss, base rate, plus the
preset 5pp calibration buckets from 50%–95%
(`[0.50,0.55) … [0.90,0.95)`, plus edge bins below/above) with n, mean
predicted P(over), and empirical hit rate per bucket.

## Pre-registered gates (unchanged from v1.1)

On the holdout, P(over) at lines 1.5/2.5/3.5/4.5:

1. Brier beats EB.
2. Log loss beats EB.
3. Per line-bucket |predicted − actual| ≤ 3pp for all buckets with n ≥ 200.
4. NB count log-likelihood beats EB (summed over the four line-heads,
   identical treatment for both models).

v1.2 validation verdict was 3/4 (calibration failing); the holdout verdict
is whatever these gates say — no post-hoc adjustments.

## Diagnostic reporting (no gates)

Report Brier / log loss / ECE by every frozen diagnostic bucket in
`DIAGNOSTIC_BUCKETS.md` (probability, mu, TOI, form, history depth,
position, home/road, line). Buckets are for understanding strengths and
weaknesses, not for triggering model changes.

## What the holdout will NOT be used for

- Tuning any threshold, edge minimum, line preference, or bankroll rule.
- Redefining eligibility, buckets, or metrics.
- Selecting among challenger designs.

Interesting holdout observations become hypotheses in `HYPOTHESES.md`,
not adjustments to v1.2.
