# NHL SOG v1.2 Baseline Manifest

The frozen baseline that will be taken to the sealed 2025-26 holdout.
Written 2026-09-28, before holdout opening.

## What "the baseline" is

| Component | Pointer |
|---|---|
| Tag | `v1.2-baseline` |
| Commit (local) | `039cc88` |
| Commit (remote `origin/main`) | `f95bcd603d604abec4cf55f307bab9179cd92ce5` |
| Tree (both — content-identical) | `96240749d061e5680279947b2f1ccfc96d235778` |

Local and remote commits differ only in commit-metadata normalization from
the API push path (no credential helper in this workspace); the trees are
byte-identical. Verify with `git rev-parse <commit>^{tree}`.

The `v1.2-baseline` tag superset-includes the `v1.2-frozen` model commit
(`8820d2ca4e666490720b06474e43a72a4688fee1`): same code and artifacts,
plus the processed model inputs. `v1-frozen` and `v1.1-frozen` are
preserved and unmoved.

## Exact pieces (all inside the tagged tree)

- **Exact code:** `src/` at the tagged commit — features, model, tuning,
  reconciliation gate, prediction interface (`src/predict.py`), market
  logging (`src/market_log.py`).
- **Exact processed inputs:** `data/processed/train.parquet` (257,567 rows),
  `data/processed/valid.parquet` (47,224 rows), `data/processed/fallbacks.json`.
- **Exact model artifacts:** `models/v1/gbm.pkl`, `models/v1/ridge_toi.pkl`,
  `models/v1/alpha_tune.json` — hashes in `models/v1/SHA256SUMS`.
- **Exact model config:** `models/v1/config.json` (feature list, Stage B
  medians, per-line alpha splits, frozen flag).
- **Exact validation predictions:** `models/v1/valid_predictions.parquet`
  (47,224 rows; mu_gbm, mu_eb, P(over) per line for model and EB).
- **Exact metrics:** `models/v1/metrics.json`; v1.1 baseline preserved at
  `models/v1/metrics_v1_1_frozen.json`.

## Reproduce from a fresh clone (no NHL API needed)

```bash
git clone https://github.com/Tkcool28/nhl.git && cd nhl
git checkout v1.2-baseline
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 15 tests, includes prediction-interface checks
.venv/bin/python src/model.py                 # deterministic rerun; metrics.json regenerates identically
.venv/bin/python src/predict.py --input data/processed/valid.parquet --out predictions.jsonl
```

`data/raw/` (79 MB, ~6,500 files) is intentionally not in the repo; it is
exactly rebuildable via the hardened ingest (`src/ingest_nhl.py`) plus the
zero-missing boxscore gate (`src/reconcile_boxscores.py`, exit 0 required).

## Frozen companion documents (all pre-holdout)

- `EVALUATION_CONTRACT.md` — how the holdout will be judged; written before
  2025-26 is opened. Moving these goalposts afterward is not allowed.
- `DIAGNOSTIC_BUCKETS.md` — eligibility rule and diagnostic bucket
  boundaries, frozen before holdout.
- `HYPOTHESES.md` — H1–H6 challenger hypothesis registry.
- `MARKET_LOGGING.md` — prediction / market / settlement schemas and
  lifecycle for future forward logging.

## Holdout status

2025-26 remains sealed. Opening it is a separate milestone using the
already-frozen model, preprocessing, eligibility rule, evaluation contract,
diagnostic buckets, and hypotheses defined here.
