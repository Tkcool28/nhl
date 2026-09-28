# NHL SOG — Eligibility Rule and Diagnostic Buckets

Frozen 2026-09-28, before the 2025-26 holdout was opened. Boundaries were
chosen for interpretability and reasonable population sizes on 2024-25
validation — not to optimize any metric. Do not move them after seeing
holdout results.

## Production-eligible population (frozen)

**Rule: `toi_l10 >= 13.0` minutes** (trailing-10-game average time on ice).

Rationale: matches intended production use — regular-rotation skaters, the
population for which sportsbooks actually post SOG lines. On 2024-25
validation this covers 79.3% of skater-games. Chosen from the standing plan
reference (~13:00), not from metric comparisons across candidate thresholds.

The `eligible` flag is emitted per prediction by `src/predict.py`.
Evaluation always reports both **all-skater** and **production-eligible**
slices.

## Diagnostic buckets (frozen)

Applied per player-game (and per line where noted). Bucket assignment uses
only information available at prediction time.

### P(over) probability — per line
`< 0.40`, `0.40–0.50`, `0.50–0.60`, `0.60–0.70`, `0.70–0.80`, `≥ 0.80`

### Expected SOG (mu_gbm)
- `low` — mu < 1.0
- `medium` — 1.0 ≤ mu < 1.75
- `high` — 1.75 ≤ mu < 2.5
- `elite` — mu ≥ 2.5

### TOI / role (toi_l10, minutes)
- `low_usage` — < 14
- `normal` — 14 ≤ toi < 17
- `high` — 17 ≤ toi < 20
- `very_high` — ≥ 20

### TOI trend (toi_trend, min/game)
- `declining` — < −0.5
- `stable` — −0.5 to +0.5
- `rising` — > +0.5

### Recent shot volume (shots_l10, per game)
- `cold` — < 1.0
- `cool` — 1.0 ≤ s < 1.75
- `warm` — 1.75 ≤ s < 2.5
- `hot` — ≥ 2.5

### Short-vs-long form (shots_l5 − shots_l10, per game)
- `slumping` — < −0.4
- `neutral` — −0.4 to +0.4
- `surging` — > +0.4

### History depth (n_trailing, prior games in frame)
- `low_sample` — < 20 (recently entered population; EB-dominated)
- `developing` — 20–150
- `established` — > 150

### Position
`D` (defense) vs `F` (forward); forward split retained as `C`, `L`, `R`.

### Home / road
`home` (1) vs `road` (0). Diagnostic only.

### Betting line
Always preserved: `1.5`, `2.5`, `3.5`, `4.5`.

## Reliability flags (emitted per prediction)

- `cold_start` — `n_trailing < 20`
- `imputed_features` — any model feature was NaN before median fill
- `ineligible` — `toi_l10 < 13.0`

These buckets and flags are diagnostic. They do not trigger model changes.
