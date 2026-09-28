# NHL SOG Projection Model — Build Plan

Player shots-on-goal projector. Model outputs a full shot-count distribution per
player-game → P(over/under) per standard line → fair American odds. A dashboard
lets Terry type in book odds and see EV, edge (pp), and model reliability.

## Architecture

Two-stage, evidence-backed:
1. **TOI model** — project ice time (rolling TOI L10, PP role proxy, home/away,
   opponent penalty rate, back-to-back, schedule density).
2. **Shot-rate model** — shots/60 × projected TOI, plus player/opponent/context
   features. Count distribution = **negative binomial** (overdispersed), never
   Poisson.

## Data

- Source: official NHL API, no key. `api-web.nhle.com/v1` (player game-logs,
  boxscore, schedule) + `api.nhle.com/stats/rest` (bulk skater summary).
- **Train:** 2018-19 → 2023-24 (regular season, gameType=2).
- **Validate:** 2024-25. Minor adjustments allowed here only: pre-registered
  knobs (shrinkage strength, window lengths, feature inclusion), ≤3 rounds,
  every change must improve pre-registered probabilistic metrics, all logged.
  Structural changes = v2, not v1 tuning.
- **Holdout:** 2025-26. Sealed. Opened once, after freeze, zero adjustments.
  The one true test.
- Per-game PP TOI is NOT in the game-log or boxscore (verified 2026-09-27);
  PP role = trailing PP points/goals share proxy.

## Leakage discipline (HR v1.2 grade)

All rolling features strictly prior games (`shift(1)`), reset at season
boundaries, walk-forward. Timestamp assertions in the build; adversarial audit
before freeze. Season-boundary reset + shrinkage carryover solves opening night:
prior-season rate blended with current-season games (empirical-Bayes prior),
weight shifting to current as games accumulate.

## Pre-registered validation gates (locked before holdout opens)

On 2024-25 validation, P(over) at lines 1.5/2.5/3.5/4.5:
- Brier score beats empirical-Bayes shrunk-rate baseline
- Per line-bucket |predicted − actual hit rate| ≤ 3pp (buckets n≥200)
- NB log-likelihood beats baseline
- Report: AUC, Brier, log loss, ECE, preset probability buckets per line
  (5pp bins, 50–95%) — the apples-to-apples record for future models.

## v1 as baseline → residual model later

v1 is frozen and versioned (artifact + feature code + data hash) after validation.
Production logs EVERY evaluated line (fixed rule: all posted SOG lines for
players with trailing TOI L10 ≥ ~13:00, both books, morning capture, auto-join
results from NHL API) — not just bets. The residual model (inputs: frozen v1 μ
+ book line + no-vig prob → predicts departure from market, shrunk hard toward
zero) is a second-half-of-season build, tested only on post-registration data.

Pre-registered market-inefficiency hypotheses: role changes (TOI jump ≥3 min,
PP-share regime shift), injury promotions, low-profile teams/players, early
season, unders-vs-overs (both directions), alt lines, back-to-backs, goalie
matchups. Min n=200 per test.

Standing rules:
- **8+ pp model-vs-book gap = "probably missing info"** → check lineup/news,
  never an auto-bet.
- Log every line entered from day one (fixes the no-historical-odds gap).
- "The app" / "NHL EDGE" in reader-facing copy, never "Terry's model".
  Muse does not have a model — Terry's model produces the numbers.

## Pipeline (production)

Morning job: today's schedule → roster/lineup check → feature build →
projections → picks JSON → `GET /api/v1/sog/latest`. Late-scratch re-run with
lineup-confirmation badge. Frozen versioned artifact.

## Frontend (Terry: phone-only)

FastAPI + single-page app, mobile-first, dark "night at the rink" theme
(deep navy/black, ice-blue neon — not white, not generic-AI-dashboard).
Centerpiece: the predicted shot-count distribution drawn as a bar chart with
the book's line marked and P(over) as the glowing region. Slate → tap player →
bottom sheet: fair American odds big, book-odds input (both sides → no-vig),
EV gauge, edge-pp bar, reliability readout (μ uncertainty, sample size,
calibration tier), one-tap log-line button.
