# Refreeze protocol — 2026-09-28 (MDT)

## Trigger

Second independent audit (2026-09-27) ran the never-before-run ingest
completeness check (boxscore reconciliation, ~8,500 games) and found a real
defect: **5,916 player-games missing (~2%), 128 (season, player) pairs**,
including full seasons from Pastrnak (all 82 games of the 2024-25 validation
season), Marner, Forsberg, Makar, Eichel. Root cause: `ingest_nhl.py` caught
fetch exceptions, printed WARN, and wrote no file — transient failures
silently dropped whole player-seasons. Full writeup:
`AUDIT_RESPONSE_2026-09-27.md`.

Terry's damage note (correct): the prior is strictly season→season−1, so a
missing season also breaks the *next* season's prior for that player
(immediate prior missing → falls straight to the league-mean fallback, no
older-season lookback). The damage reaches beyond the 2%, and it hits stars.
E.g. Pastrnak's 2024-25 absence would also corrupt his 2025-26 prior —
the holdout/production season.

Terry authorized: **fix, refreeze** (2026-09-27). Frontend stays deferred
until the model is clean and the holdout is run.

## Pre-refreeze baseline (frozen, preserved)

Tag `v1-frozen` (commit `2df3fd65`) + `models/v1/metrics_prefix.json` remain
the permanent record of the frozen v1 numbers. They are not modified by
this refreeze.

- AUC 0.8300, Brier 0.1212, log loss 0.3797, ECE 0.0040
- NB count log-likelihood: −288,224.5 (v1) vs −290,669.0 (EB)
- Gates: brier ✓, logloss ✓, nb_ll ✓, calibration ✗ (2 top-end buckets)

## Commitments

1. The numbers above stay on record as the **pre-refreeze** baseline
   (`v1-frozen` tag is never moved).
2. **No re-tuning.** The pipeline re-executes deterministically as coded:
   `ingest_nhl.py` (hardened — mechanical fix, not modeling) →
   `reconcile_boxscores.py` (gate, must read zero missing) →
   `features.py` → `tune_alpha_split.py` → `model.py`. No knob changes, no
   new experiments, no grid changes. `tune_alpha_split.py` re-running its
   fixed grids on complete data is the pipeline as coded, not a new
   tuning decision.
3. Whatever comes out is frozen under the new tag **`v1.1-frozen`**.
   If the top-end calibration numbers move, that is expected to be part
   of why (more star-seasons in the top buckets) — accepted, not adjusted.

## Out of scope for this refreeze

- Model code (`features.py`, `model.py`, `tune_*.py`) is byte-identical to
  the `v1-frozen` tag except the ingest hardening (data plumbing, not the
  model). The coherent-α head, continuous α fit, two-stage comparison, and
  all other v2 items stay v2.
- The 2025-26 holdout stays sealed. The reconciliation gate for the
  2025-26 pull is built into the holdout script (to be written and
  independently audited before opening), not run now.
- Production incremental-ingest design (boxscore-driven, stale-data checks,
  nightly reconciliation + alerting) is a separate later build.

## Steps (in order; stop on any nonzero exit)

1. Harden `src/ingest_nhl.py` (rate limiter, jittered backoff ×8, failure
   manifest, nonzero exit on unresolved failure, 404-vs-failure
   distinction, deduped skater list, corrupt-cache refetch).
2. Add `src/reconcile_boxscores.py` (boxscore gate; exit 0 iff zero missing).
3. Delete the 15 partial files (present but missing 1–4 games each; listed
   in AUDIT_RESPONSE_2026-09-27.md) so the resume refetches them whole.
   The 113 fully-missing pairs have no files and are picked up by resume.
4. Run ingest (resume). Must exit 0.
5. Run the reconciliation gate. Must exit 0 (zero missing) or stop.
6. Run `features.py` → `tune_alpha_split.py` → `model.py`.
7. Verify: spot-check priors for previously-missing stars (e.g. Marner
   2023-24 prior_pg uses his real 2022-23 rate, not the league mean).
8. Write new SHA256SUMS, push, tag `v1.1-frozen`.

## Additional repair findings (2026-09-28, during execution)

The hardened ingest + gate surfaced three sub-cases beyond plain transient
failures. All are data repairs, not modeling changes:

1. **Unstable bulk skater list.** The stats API paginates unstably: the
   2022-23 list returned 997 skaters in the original ingest but 931–939 on
   re-query, omitting real players (e.g. Marner). The ingest now unions the
   bulk list with `official_rosters.json` (ground-truth boxscore rosters
   written by the gate) so the repair list is complete.
2. **Phantom game-log entries.** Lindholm's 2019-20 log contained a game
   (2020-03-11) with no `toi` field; the official boxscore confirms he did
   not dress. The ingest drops toi-less entries (recorded in the manifest);
   the gate is the backstop — a wrongly-dropped real game would be flagged.
3. **0:00-TOI boxscore listings.** ~20 remaining "missing" games were
   skaters listed in boxscores with 0:00 TOI (dressed, no shifts). The
   game-log API omits these and v1's training data never contained them,
   so the gate tolerates them (they are not real shot-rate observations).
4. **Genuine game-log gaps (boxscore backfill).** Two 2024-25 player-seasons
   have real boxscore games (ice time + shots) the game-log endpoint never
   returns: Gaucher (4 games, 2025-02-02..08) and White (3 games, endpoint
   returns an empty log). `src/backfill_from_boxscores.py` repairs these
   from the official boxscores after verifying field-for-field schema
   agreement on overlapping games (sog==shots, toi/shifts/pim/goals/
   assists/plusMinus identical; null ppPoints -> 0). Manifest:
   `data/raw/boxscore_backfills.json`.

## Refreeze results (2026-09-28)

Gate passed at **zero missing player-games** across all 8,469 official
regular-season games before the pipeline ran.

| | v1-frozen (defective data) | v1.1-frozen (complete data) |
|---|---|---|
| train rows | 252,351 | 257,567 (+5,216) |
| valid rows | 46,538 | 47,224 (+686) |
| valid players | 907 | 920 |
| AUC (gbm_nb) | 0.8300 | 0.8304 |
| Brier (gbm_nb) | 0.1212 | 0.1213 |
| log loss (gbm_nb) | 0.3797 | 0.3797 |
| ECE (gbm_nb) | 0.0040 | 0.0042 |
| Brier edge over EB | 0.0016 | 0.0014 |
| NB LL (gbm_nb vs EB) | −288,224.5 / −290,669.0 | −292,504.3 / −294,878.9 |
| gates | brier ✓ logloss ✓ nb_ll ✓ calib ✗ | brier ✓ logloss ✓ nb_ll ✓ calib ✗ |

Failing calibration buckets (same two top-end buckets, accepted per the
no-retune commitment):
- over 1.5 [0.85,0.90): v1 n=350 miss 6.6pp → v1.1 n=393 miss 4.3pp (improved)
- over 2.5 [0.65,0.70): v1 n=244 miss 4.9pp → v1.1 n=312 miss 6.2pp (worse)

Verdict unchanged: 3/4 gates pass; the model beats EB on Brier, log-loss,
and NB count likelihood with the same calibration caveat as v1.

## Metadata layout invariant

`data/raw/game_logs/` contains ONLY season directories. All ingest/gate
metadata (`manifest.json`, `official_rosters.json`, `failures.json`,
`boxscore_backfills.json`) lives in `data/raw/`, because `features.py`
iterates `game_logs/` expecting season dirs. (2026-09-28: metadata files
first landed inside `game_logs/` and crashed the feature loader; moved.)
