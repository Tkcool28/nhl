"""Frozen v1.2 baseline prediction interface.

Loads the frozen model artifacts and emits schema'd, timestamped, immutable
prediction records (JSONL) for a processed feature frame. Future challenger
models write to the same schema so outputs are comparable game-for-game.

The model itself is untouched: this reproduces src/model.py's Stage B
scoring path exactly (Stage B medians, clean_mu, residual GBM, per-line
alpha splits).

Usage:
    python src/predict.py --input data/processed/valid.parquet --out predictions.jsonl
    python src/predict.py --input ... --out ... --run-id abc123 --predicted-at 2026-09-28T12:00:00
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import clean_mu, p_over, LINES  # noqa: E402  (frozen scoring math)

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "models" / "v1"
BASELINE_VERSION = "v1.2"
ELIGIBLE_TOI = 13.0  # frozen eligibility rule (DIAGNOSTIC_BUCKETS.md)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def american_odds(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def reliability_flags(row, imputed):
    flags = []
    if row["n_trailing"] < 20:
        flags.append("cold_start")
    if imputed:
        flags.append("imputed_features")
    if row["toi_l10"] < ELIGIBLE_TOI:
        flags.append("ineligible")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="processed feature parquet")
    ap.add_argument("--out", required=True, help="output JSONL (must not exist)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--predicted-at", default=None,
                    help="ISO-8601 override (tests); default now UTC")
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite existing output: {args.out}")

    run_id = args.run_id or uuid.uuid4().hex[:12]
    predicted_at = args.predicted_at or datetime.now(timezone.utc).isoformat()

    cfg = json.load(open(ART / "config.json"))
    features = cfg["features"]
    med = pd.Series(cfg["medians"])
    alpha = cfg["alpha"]["gbm_nb"]
    with open(ART / "gbm.pkl", "rb") as f:
        gbm = pickle.load(f)

    df = pd.read_parquet(args.input)
    for col in ["game_id", "player_id"]:
        if col not in df.columns:
            sys.exit(f"input frame missing required column: {col}")

    X = df[features].fillna(med[features])
    imputed_any = df[features].isna().any(axis=1).values
    eb = clean_mu(df["eb_pred"].values, df)
    mu_gbm = clean_mu(eb + gbm.predict(X), df)

    model_sha = sha256_file(ART / "gbm.pkl")
    config_sha = sha256_file(ART / "config.json")
    data_sha = sha256_file(args.input)

    n = 0
    with open(args.out, "w") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            rec = {
                "run_id": run_id,
                "predicted_at": predicted_at,
                "game_date": str(row["game_date"])[:10],
                "game_id": int(row["game_id"]),
                "player_id": int(row["player_id"]),
                "player_name": str(row.get("name", "")),
                "team": str(row.get("team", "")),
                "opp": str(row.get("opp", "")),
                "home_away": "home" if int(row.get("home", 0)) == 1 else "away",
                "position": str(row.get("pos", "")),
                "baseline_version": BASELINE_VERSION,
                "model_sha": model_sha,
                "config_sha": config_sha,
                "data_sha": data_sha,
                "eligible": bool(row["toi_l10"] >= ELIGIBLE_TOI),
                "history_depth": int(row["n_trailing"]),
                "toi_l10": round(float(row["toi_l10"]), 3),
                "shots_l5": round(float(row["shots_l5"]), 3),
                "shots_l10": round(float(row["shots_l10"]), 3),
                "mu_gbm": round(float(mu_gbm[i]), 4),
                "mu_eb": round(float(eb[i]), 4),
                "reliability_flags": reliability_flags(row, bool(imputed_any[i])),
            }
            for L in LINES:
                p = float(p_over(np.array([mu_gbm[i]]), alpha[str(L)], L)[0])
                key = str(L).replace(".", "_")
                rec[f"p_over_{key}"] = round(p, 6)
                rec[f"fair_odds_over_{key}"] = american_odds(p)
                rec[f"fair_odds_under_{key}"] = american_odds(1 - p)
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} predictions -> {args.out} (run_id={run_id})")


if __name__ == "__main__":
    main()
