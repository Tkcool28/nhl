# v1.2 Correctness Refreeze Protocol — 2026-09-28

## Starting point

- main at handoff: `5fd8eb07ec185b75d27782ba9347b57ef6a2eb83`
- Frozen lineage preserved: `v1-frozen`, `v1.1-frozen` (tags never moved)

## Bugs repaired (correctness only, no redesign, no new tuning)

### 1. Rolling-boundary temporal leakage (src/features.py)

Old pattern:

```python
g = df.groupby("player_id", group_keys=False)
def trailing(g, col, n):
    return g[col].shift(1).rolling(n, min_periods=1).mean()
```

`shift(1)` is group-scoped, but `.rolling(n)` ran on the flattened
frame sorted by `["player_id", "game_date"]`. Rolling windows therefore
crossed from the end of Player A into the start of Player B: early
Player B rows inherited Player A's future-game values. Same structural
defect in team rolls (`Team A -> Team B`). Genuine temporal leakage.

Fix: the whole shift+rolling operation now stays inside the group via
`.transform()`:

```python
def trailing(gb_col, n):
    return gb_col.transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
```

Applied to every rolling path: shots_l5/l10/l20, toi_l10, toi_trend,
shots_per60_l10 (sum/sum), pp_share_l10 (sum/sum), shots_home_l10,
shots_road_l10, eb_pred trailing sum (s10) and count (n10), naive_l10,
team_shots_for_l10, team_shots_against_l10, team_pim_l10, team_pace_l10
(derived). `n_trailing` (cumcount) and `rest_days` (group diff) were
already group-scoped; untouched.

### 2. Cross-season carryover is INTENTIONAL and retained

Groups are by `player_id` only, never `(player_id, season)`. Opening
night uses prior-season history; veterans do not reset to zero history.
Pinned by `test_season_carryover_intact` (must NOT assert a reset).

### 3. Stage A median leakage (src/model.py, src/tune_alpha_split.py)

Old: imputation medians computed over the full training frame (through
2023-24), then used to impute the Stage A fit frame (<=2022-23) AND the
2023-24 selection frame. The frame being evaluated contributed its own
imputation statistics. Poisoning 2023-24 moved 13/19 medians
(e.g. toi_l10 16.50 -> 17.68).

Fix: `stage_a_medians()` from the fit population (<=2022-23) only, used
for both Stage A frames. `stage_b_medians()` from the full training
population (through 2023-24) for final fit + 2024-25 validation.
`tune_alpha_split.py` no longer reads medians from the previous
config.json; it computes Stage A medians from its own fit frame
(and was wrapped in `main()` so it is importable/testable without
side effects).

## Tests (written BEFORE the fix, run against pre-fix code)

`tests/test_rolling_boundaries.py`:
- A. player A->B poison: FAILED pre-fix (victim first-row shots_l5 =
  111.5, poison's values). Passes post-fix (NaN = no prior history).
- B. team A->B poison: FAILED pre-fix (ZZZ first-game
  team_shots_for_l10 = 106.0, AAA's values). Passes post-fix.
- C. season carryover: PASSED pre- and post-fix (carryover already
  worked; now pinned: opening night shots_l10 == 5.0 from prior season,
  current game excluded, future-row poison invariance).
- Poison-player own-row exactness: PASSED both (sanity + current-game
  exclusion).

`tests/test_pipeline_invariants.py`:
- holdout season dir refused; holdout gameDate refused.
- Stage A medians immune to 2023-24 poisoning (model.py +
  tune_alpha_split.py); Stage A == fit-population medians; Stage B ==
  full-train medians.
- Prediction monotonicity P(over 1.5) >= ... >= P(over 4.5) on all
  validation rows.

Post-fix: 13/13 pass.

## Rerun procedure (deterministic, exactly as committed)

1. `src/reconcile_boxscores.py` -- must exit 0 (zero missing
   player-games across 8,469 official regular-season games).
2. `src/features.py` -- rebuild train/valid parquets.
3. `src/tune_alpha_split.py` -- Stage A alpha grid rerun (grid unchanged).
4. `src/model.py` -- Stage A selection report, Stage B fit, validation.

No feature, architecture, threshold, grid, hyperparameter, or
post-result changes. If the fixed pipeline selects a different
already-preregistered alpha-grid entry, that stands.

## Results

(Filled after the rerun; compare against v1.1-frozen below.)

| Metric | v1.1-frozen | v1.2 (corrected) |
|---|---:|---:|
| Train rows | 257,567 | 257,567 |
| Validation rows | 47,224 | 47,224 |
| Validation players | 920 | 920 |
| AUC | 0.8304 | 0.8305 |
| Brier | 0.1213 | 0.1212 |
| Log loss | 0.3797 | 0.3796 |
| ECE | 0.0042 | 0.0038 |
| Brier edge over EB | 0.0014 | 0.0015 |
| GBM-NB count LL | -292,504.3 | -292,410.0 |
| EB count LL | -294,878.9 | -294,905.8 |

Gates: 3/4 (brier ✓, logloss ✓, calibration ✗, nb_ll ✓) -- same as v1.1.

Calibration buckets failing the 3pp gate (n>=200):
- Over 1.5, [0.85,0.90): n=392, miss 4.3pp (v1.1: n=393, 4.3pp -- unchanged)
- Over 2.5, [0.65,0.70): n=316, miss 5.2pp (v1.1: n=312, 6.2pp -- improved)
- Over 2.5, [0.50,0.55): n=1190, miss 3.5pp (new marginal bucket)

The corrected pipeline selected different preregistered alpha-grid entries
on two lines (allowed per protocol, no grid change):
- line 2.5: lo 0.02->0.05 (hi 0.05 unchanged)
- line 4.5: hi 0.05->0.02 (lo 0.02 unchanged)

Net: the leakage was slightly *hurting* aggregate performance (it injected
other players' values as noise into early-season rows). Corrected metrics
are a touch better across the board; the high-end calibration caveat
remains and is recorded, not tuned around.

## Unexpected findings during the rerun

1. `clean_mu()` had a latent index-misalignment bug: it built a fresh
   RangeIndex Series and filled NaNs from `df["prior_pg"]`, which
   misaligns whenever df is a non-contiguous slice, leaving NaNs. The
   rolling leakage had masked it (NaN eb_pred was rare and clustered at
   the frame start where indices coincided). Fix: index the working
   Series like df. Pinned by `test_clean_mu_fill_aligns_on_slices`.
2. The boxscore reconciliation gate's `get()` did not retry on
   connection-level failures (ReadTimeout/ProxyError) -- two transient
   failures killed gate runs. Added RequestException to the
   backoff-and-retry path (same spirit as the v1.1 ingest hardening).
   Gate then passed: 0 missing player-games, exit 0.

## Freeze record

- v1.1 baseline preserved: `models/v1/metrics_v1_1_frozen.json`
  (in addition to the `v1.1-frozen` tag).
- New tag: `v1.2-frozen` on the refreeze commit.
- Old tags `v1-frozen`, `v1.1-frozen` verified unchanged.
- 2025-26 holdout: never pulled, read, reconciled, scored, or inspected.
  7 season directories on disk before and after.

## Follow-on audit items (NOT in this fix, per handoff Part 9)

A. Probability head is four per-line NB heads, not one coherent count
   distribution -- separate model-version work.
B. Production-eligible validation slice (trailing TOI >= ~13:00) --
   predeclare the filter, then report both slices.
C. High-end calibration weakness -- remeasure honestly post-fix; any
   tuning is a separate preregistered experiment.
D. PLAN.md two-stage architecture description vs actual residual-target
   GBM -- update docs or build the intended architecture separately.
E. pp_share_l10 is a weak PP-role proxy -- later research only.
F. GBM edge over EB is small (~0.0014 Brier) -- correctness and
   population matching matter more than ever.
