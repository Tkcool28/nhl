# NHL SOG — Market Logging Framework

The model stays independent of sportsbook prices. Market data lives in a
comparison layer. All logs are append-only JSONL; nothing is ever
overwritten or mutated after writing.

## Lifecycle

`prediction → market observations → close → outcome`

1. **Prediction** — `src/predict.py` emits one record per player-game at
   prediction time (immutable once written).
2. **Market observations** — each observed SOG market is appended with its
   observation timestamp. Line or price moves append new records; the old
   ones stay.
3. **Close** — where feasible, capture first-observed, meaningful
   intermediate, and last-pregame (closing) market states.
4. **Settlement** — after the game, append actual SOG and per-line results.
   Never mutates the prediction or market records.

## Prediction record (emitted by `src/predict.py`)

| Field | Meaning |
|---|---|
| run_id | UUID for this prediction run |
| predicted_at | ISO-8601 timestamp of the run |
| game_date / game_id | official game |
| player_id / player_name / team / opp / home_away / position | identity |
| baseline_version | `v1.2` |
| model_sha / config_sha / data_sha | sha256 of `gbm.pkl`, `config.json`, input frame |
| eligible | `toi_l10 >= 13.0` |
| history_depth / toi_l10 / shots_l5 / shots_l10 | key features |
| mu_gbm / mu_eb | expected SOG, model and EB baseline |
| p_over_{1.5,2.5,3.5,4.5} | model P(over) per line |
| fair_odds_over_{...} / fair_odds_under_{...} | American odds from model p |
| reliability_flags | `cold_start`, `imputed_features`, `ineligible` (as applicable) |

## Market observation record

sportsbook/source, observed_at, game_id, player_id, line, over_price,
under_price (American), imp_over / imp_under (raw implied), devig_over /
devig_under (no-vig, when both sides available), model_p_over,
model_fair_over, edge_pp (model_p − devigged market p, percentage points),
edge_direction (`model_over` / `model_under`), baseline_version, eligible.

## Settlement record

game_id, player_id, actual_sog, result per observed line (`over`/`under`/
`push`), settled_at. Appended after the game; references the immutable
prediction and market records by run_id / observation id.

## Predeclared market-edge buckets (frozen 2026-09-28)

Absolute |model − market| probability difference, with direction kept:

`< 2pp`, `2–4pp`, `4–6pp`, `6–10pp`, `≥ 10pp` × `model_over` / `model_under`.

Do not redefine these buckets later because another set looks better.

## Scoring once real market logging starts

**Model quality** (independent of wagering): Brier, log loss, calibration,
expected-SOG error, diagnostic-bucket breakdowns, EB comparison —
per `EVALUATION_CONTRACT.md`.

**Market usefulness**: model vs de-vigged market probability; results by
predeclared edge bucket; line/price movement after prediction; closing-line
value; simulated ROI only after betting rules are frozen.

Short-run ROI alone never defines whether the model works. And per the
standing rule: the holdout must not become a tuning set for edge minimums,
line preferences, or bankroll rules — interesting observations become
hypotheses, not optimizations.
