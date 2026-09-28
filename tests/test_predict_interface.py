"""Tests for the frozen v1.2 prediction interface (src/predict.py).

These pin the baseline output contract: schema, per-row probability
monotonicity across lines, fair-odds consistency, determinism, and
immutability (no overwrites). They run from repo data -- no downloads.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
REQUIRED = ["run_id", "predicted_at", "game_date", "game_id", "player_id",
            "player_name", "team", "opp", "home_away", "position",
            "baseline_version", "model_sha", "config_sha", "data_sha",
            "eligible", "history_depth", "toi_l10", "shots_l5", "shots_l10",
            "mu_gbm", "mu_eb", "reliability_flags",
            "p_over_1_5", "p_over_2_5", "p_over_3_5", "p_over_4_5"]


def _run_predict(out, run_id="testrun", predicted_at="2026-09-28T00:00:00+00:00",
                 nrows=300):
    va = pd.read_parquet(REPO / "data" / "processed" / "valid.parquet")
    tmp_in = REPO / "data" / "processed" / "_test_slice.parquet"
    va.head(nrows).to_parquet(tmp_in, index=False)
    try:
        r = subprocess.run(
            [sys.executable, str(SRC / "predict.py"), "--input", str(tmp_in),
             "--out", str(out), "--run-id", run_id,
             "--predicted-at", predicted_at],
            capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr
    finally:
        tmp_in.unlink(missing_ok=True)
    return [json.loads(l) for l in open(out) if l.strip()]


def test_prediction_schema_and_monotonicity(tmp_path):
    recs = _run_predict(tmp_path / "p1.jsonl")
    assert len(recs) == 300
    for r in recs:
        for k in REQUIRED:
            assert k in r, f"missing {k}"
        assert r["baseline_version"] == "v1.2"
        assert r["eligible"] == (r["toi_l10"] >= 13.0)
        ps = [r["p_over_1_5"], r["p_over_2_5"], r["p_over_3_5"], r["p_over_4_5"]]
        assert all(0 < p < 1 for p in ps)
        assert ps[0] >= ps[1] >= ps[2] >= ps[3], "P(over) must fall with line"
        # fair odds sign must agree with probability side
        assert (r["fair_odds_over_1_5"] < 0) == (r["p_over_1_5"] > 0.5)
        assert r["mu_gbm"] >= 0.05 and r["mu_eb"] >= 0.05
        assert r["home_away"] in ("home", "away")


def test_prediction_determinism_and_immutability(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _run_predict(a)
    _run_predict(b)
    assert a.read_bytes() == b.read_bytes(), "same inputs must give same bytes"
    # refuse to overwrite
    r = subprocess.run(
        [sys.executable, str(SRC / "predict.py"), "--input",
         str(REPO / "data" / "processed" / "valid.parquet"),
         "--out", str(a), "--run-id", "x"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr


def test_market_log_append_only(tmp_path):
    sys.path.insert(0, str(SRC))
    import market_log
    p = tmp_path / "m.jsonl"
    recs = _run_predict(tmp_path / "p.jsonl")
    assert market_log.append_predictions(p, recs[:5]) == 5
    assert len(market_log.read_jsonl(p)) == 5
    # second append adds, never modifies
    assert market_log.append_predictions(p, recs[5:8]) == 3
    rows = market_log.read_jsonl(p)
    assert len(rows) == 8 and rows[0]["player_id"] == recs[0]["player_id"]
    # schema enforced
    try:
        market_log.append_markets(p, [{"line": 2.5}])
        raise AssertionError("should have raised")
    except ValueError:
        pass
