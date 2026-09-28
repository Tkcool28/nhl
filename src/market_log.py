"""Append-only JSONL logging for market observations and settlements.

The model stays independent of sportsbook prices; this module is the
comparison layer described in MARKET_LOGGING.md. Files are append-only:
records are validated and appended, never modified or overwritten.
"""
import json
import os

PREDICTION_REQUIRED = ["run_id", "predicted_at", "game_date", "game_id",
                       "player_id", "baseline_version", "model_sha",
                       "mu_gbm", "mu_eb"]
MARKET_REQUIRED = ["observed_at", "source", "game_id", "player_id", "line",
                   "over_price", "under_price", "model_p_over",
                   "baseline_version"]
SETTLEMENT_REQUIRED = ["game_id", "player_id", "actual_sog", "settled_at"]


def _validate(record, required, kind):
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"{kind} record missing fields: {missing}")
    return record


def append_records(path, records, required, kind):
    """Append validated records to a JSONL log. Creates the file if needed;
    never modifies existing records."""
    records = list(records)
    for r in records:
        _validate(r, required, kind)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(records)


def append_predictions(path, records):
    return append_records(path, records, PREDICTION_REQUIRED, "prediction")


def append_markets(path, records):
    return append_records(path, records, MARKET_REQUIRED, "market")


def append_settlements(path, records):
    return append_records(path, records, SETTLEMENT_REQUIRED, "settlement")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
