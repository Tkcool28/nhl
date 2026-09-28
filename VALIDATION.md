# v1 Validation Report — 2024-25 season (frozen)

**Verdict: 2 of 3 pre-registered gates pass. Calibration gate fails on 2 of 17
qualifying buckets (both extreme top-end). Model is frozen as-is; the full
bucket table below is the apples-to-apples record for future models.**

## Setup

- Train: 2018-19 → 2023-24 (252,351 player-games). Validate: 2024-25
  (46,538 player-games, 907 players). **2025-26 sealed holdout never touched**
  (no files, no API calls — hard guards in every script).
- Main model: residual-target GBM — `mu = eb_pred + GBM(shots − eb_pred)` —
  with negative-binomial head → P(over/under) at lines 1.5/2.5/3.5/4.5.
- Adjustment rounds used: 3 of 3 (all tuned on 2023-24 only, never 2024-25):
  1. Per-line NB dispersion (moment-matched alpha was overdispersed).
  2. Residual target (GBM-poisson overpredicted stars: mu 3.33 vs actual 3.15
     in the top bucket).
  3. Mu-dependent dispersion (alpha_lo for mu≤3, alpha_hi for mu>3) — did NOT
     transfer from 23-24 to 24-25; kept for pipeline consistency, pooled
     metrics identical. Likely noise; revisit in v2.

## Pooled metrics (all four lines, 186,152 predictions)

| model       | AUC    | Brier  | Log loss | ECE   |
|-------------|--------|--------|----------|-------|
| gbm_nb (v1) | 0.8298 | 0.1213 | 0.3798   | 0.0039|
| eb baseline | 0.8259 | 0.1227 | 0.3839   | 0.0150|
| naive L10   | 0.8117 | 0.1258 | 0.3953   | 0.0082|
| ridge2stage | 0.8119 | 0.1258 | 0.3951   | 0.0081|

Gates: Brier beats EB ✓ · Log-loss beats EB ✓ · Buckets ≤3pp ✗ (2 misses)

## Per-line (v1)

| line | base rate | AUC    | Brier  | Log loss |
|------|-----------|--------|--------|----------|
| 1.5  | 0.434     | 0.7110 | 0.2126 | 0.6143   |
| 2.5  | 0.221     | 0.7433 | 0.1491 | 0.4623   |
| 3.5  | 0.103     | 0.7755 | 0.0830 | 0.2853   |
| 4.5  | 0.046     | 0.8073 | 0.0405 | 0.1574   |

## Preset calibration buckets (v1) — the apples-to-apples record

Predicted P(over) in 5pp bins vs actual hit rate, per line:

**Over 1.5**
| bucket | n | pred | hit |
|---|---|---|---|
| <0.50 | 29664 | 0.319 | 0.315 |
| 0.50–0.55 | 3224 | 0.526 | 0.526 |
| 0.55–0.60 | 3329 | 0.575 | 0.573 |
| 0.60–0.65 | 2886 | 0.624 | 0.621 |
| 0.65–0.70 | 2549 | 0.674 | 0.680 |
| 0.70–0.75 | 2050 | 0.724 | 0.723 |
| 0.75–0.80 | 1533 | 0.774 | 0.768 |
| 0.80–0.85 | 843 | 0.821 | 0.807 |
| 0.85–0.90 | 334 | 0.871 | 0.826 ⚠ miss by 4.5pp |

**Over 2.5**
| bucket | n | pred | hit |
|---|---|---|---|
| <0.50 | 43121 | 0.194 | 0.194 |
| 0.50–0.55 | 1354 | 0.525 | 0.497 |
| 0.55–0.60 | 1147 | 0.570 | 0.593 |
| 0.60–0.65 | 454 | 0.622 | 0.593 |
| 0.65–0.70 | 236 | 0.671 | 0.623 ⚠ miss by 4.8pp |

**Over 3.5**: all mass <0.50 (n=46250, pred 0.101, hit 0.101) ✓
**Over 4.5**: all mass <0.50 (n=46509, pred 0.043, hit 0.045) ✓

## Known limitations (production-relevant)

1. **Model runs ~4–5pp hot at the extreme top end** (P(over 1.5) > 0.85,
   P(over 2.5) > 0.65 — i.e., superstar overs). Treat high-confidence over
   edges with extra skepticism; the 8pp "missing info" rule catches the worst
   of it, but a 4pp phantom edge can survive it.
2. Lines 3.5/4.5 have almost no high-confidence predictions — the model rarely
   sees a player as >50% to clear 3.5. Alt-line value detection will be thin.
3. Line 1.5 AUC is only 0.71 — the 1.5 line is inherently noisy (one shot
   either way); the model discriminates much better at 2.5+.

## Reproduce

```
.venv/bin/python src/ingest_nhl.py  # 2018-19..2024-25 only; refuses 2025-26
.venv/bin/python src/features.py
.venv/bin/python src/tune_alpha_split.py  # round 3 tuning (2023-24 only)
.venv/bin/python src/model.py             # trains, validates, writes models/v1/
.venv/bin/python tests/test_leakage.py
```

Full bucket tables live in `models/v1/metrics.json`.
