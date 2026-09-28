# Freeze protocol — v1 bug-fix rerun (2026-09-27)

## The bug
`src/features.py` computed the league-mean fallback over **all** seasons,
including validation (2024-25). Any validation player with no prior season
(rookies, returnees) got `prior_pg` filled with the **validation season's own
mean** — validation information leaking into validation features (and
`eb_pred`, which is built from `prior_pg`). Training rows had a milder form:
early-season games filled with the full current-season mean (future info).

## The fix (bug correction, NOT a tuning round)
- `league` means and the fallback scalar are computed on **training seasons
  only** (≤2023-24) and written to `data/processed/fallbacks.json`.
- `prior_pg` fallback hierarchy: prior season's league mean (strictly prior
  information) → frozen training-only mean for 2018-19 (no prior season
  exists in the dataset).
- `src/model.py::clean_mu` no longer computes any mean over the validation
  frame; its safety-net fill uses the frozen training mean.
- Row-count assertions added after every merge in `features.py` (guards the
  groupby-alignment fragility; behavior-neutral).
- Nothing else changes: feature set, model class, α grids, split threshold,
  early-stopping seed, and the 3-round budget are all untouched.

## Commitments (made before the rerun)
1. **The current numbers stay on record as the "pre-fix" baseline.**
   Saved as `models/v1/metrics_prefix.json`; summarized below.
2. **No re-tuning around whatever comes out.** The deterministic pipeline
   (features → α grid search on 2023-24 → GBM fit → validation scoring) is
   re-executed exactly as coded. No human choices are revisited, no knobs
   touched, no rounds added. Whatever the single rerun produces is frozen
   as v1.

## Pre-fix baseline (frozen 2026-09-28, before the fix)
- n_train=252,351 · n_valid=46,538 · 907 validation players
- 2023-24 selection: resid-GBM Brier 0.1276 vs EB 0.1294
- Pooled 2024-25: gbm_nb AUC 0.8298 / Brier **0.1213** / logloss **0.3798** /
  ECE 0.0039 · eb 0.8259 / 0.1227 / 0.3839 / 0.0150 ·
  naive_l10 0.8117 / 0.1258 / 0.3953 / 0.0082 · ridge2stage 0.8119 / 0.1258 / 0.3951 / 0.0081
- Gates: brier_beats_eb ✓ · logloss_beats_eb ✓ · calibration_3pp ✗
  (2 misses: over-1.5 [0.85,0.90) pred 0.871 hit 0.826 n=334;
   over-2.5 [0.65,0.70) pred 0.671 hit 0.623 n=236)
- Per-line gbm_nb: 1.5 → base 0.434 / auc 0.7110 / brier 0.2126 / ll 0.6143;
  2.5 → 0.221 / 0.7433 / 0.1491 / 0.4623;
  3.5 → 0.103 / 0.7755 / 0.0830 / 0.2853;
  4.5 → 0.046 / 0.8073 / 0.0405 / 0.1574
- α (gbm_nb, split at μ=3): 1.5 lo 0.05/hi 0.05 · 2.5 lo 0.05/hi 0.10 ·
  3.5 lo 0.05/hi 0.10 · 4.5 lo 0.02/hi 0.05
