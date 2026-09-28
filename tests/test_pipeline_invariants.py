"""Permanent pipeline invariants (v1.2 correctness refreeze).

- Holdout season is refused everywhere it can appear.
- Stage A imputation medians come from the Stage A fit population only
  (<=2022-23): poisoning the 2023-24 selection frame must not move them.
- Stage B medians come from the full training population (<=2023-24).
- Validation predictions stay monotonic across lines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import features
import model

REPO = Path(__file__).resolve().parent.parent


def _player_json(season_dir, pid, games):
    (season_dir / f"{pid}.json").write_text(json.dumps(
        {"meta": {"playerId": pid, "name": "T", "pos": "C"}, "games": games}))


def _game(date, team="AAA", shots=2, toi="10:00"):
    return {"gameId": 1, "gameDate": date, "team": team, "opp": "ZZZ",
            "homeRoad": "H", "shots": shots, "toi": toi, "shifts": 15,
            "pim": 0, "goals": 0, "assists": 0, "ppPoints": 0}


def test_holdout_season_dir_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "RAW", tmp_path)
    season_dir = tmp_path / "20252026"
    season_dir.mkdir()
    _player_json(season_dir, 1, [_game("2025-10-08")])
    with pytest.raises(AssertionError):
        features.load()


def test_holdout_game_date_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "RAW", tmp_path)
    season_dir = tmp_path / "20232024"
    season_dir.mkdir()
    _player_json(season_dir, 1, [_game("2025-10-08")])
    with pytest.raises(RuntimeError):
        features.load()


def _train():
    return pd.read_parquet(REPO / "data" / "processed" / "train.parquet")


def test_stage_a_medians_immune_to_selection_frame():
    """Poison every 2023-24 imputed column: Stage A medians must not move.
    (Old code computed medians over the full train frame, so the frame
    being evaluated contributed its own imputation statistics.)"""
    tr = _train()
    before = model.stage_a_medians(tr)
    poisoned = tr.copy()
    sel = poisoned["season"] == 20232024
    for col in model.ALL_COLS:
        if poisoned[col].dtype.kind == "f":
            poisoned.loc[sel, col] = 99999.0
    after = model.stage_a_medians(poisoned)
    pd.testing.assert_series_equal(before, after)


def test_stage_a_medians_equal_fit_population_only():
    tr = _train()
    expected = tr[tr["season"] <= 20222023][model.ALL_COLS].median()
    pd.testing.assert_series_equal(model.stage_a_medians(tr), expected)


def test_stage_b_medians_use_full_train():
    tr = _train()
    expected = tr[model.ALL_COLS].median()
    pd.testing.assert_series_equal(model.stage_b_medians(tr), expected)


def test_tune_stage_a_medians_immune_to_selection_frame():
    import tune_alpha_split as tas
    tr = _train()
    d_fit = tr[tr["season"] <= 20222023]
    cols = list(model.FEATURES)
    before = tas.stage_a_medians(d_fit, cols)
    poisoned = tr.copy()
    sel = poisoned["season"] == 20232024
    for col in cols:
        if poisoned[col].dtype.kind == "f":
            poisoned.loc[sel, col] = 99999.0
    d_fit_p = poisoned[poisoned["season"] <= 20222023]
    after = tas.stage_a_medians(d_fit_p, cols)
    pd.testing.assert_series_equal(before, after)


def test_clean_mu_fill_aligns_on_slices():
    """clean_mu must fill NaN mu with that ROW's prior_pg even when df is a
    non-contiguous slice (2026-09-28: the old RangeIndex silently misaligned,
    leaving NaNs that crashed the GBM fit once leakage no longer masked them)."""
    tr = _train()
    d = tr[tr["season"] == 20232024].copy()  # non-contiguous index slice
    d.loc[d.index[:50], "eb_pred"] = float("nan")
    mu = model.clean_mu(d["eb_pred"].values, d)
    assert not np.isnan(mu).any()
    prior = d["prior_pg"].fillna(
        float(json.load(open(REPO / "data" / "processed" / "fallbacks.json"))
                        ["train_mean_shots"])).values[:50]
    np.testing.assert_allclose(mu[:50], np.clip(prior, 0.05, None))


def test_prediction_monotonicity():
    p = pd.read_parquet(REPO / "models" / "v1" / "valid_predictions.parquet")
    cols = ["p_over_1.5", "p_over_2.5", "p_over_3.5", "p_over_4.5"]
    ok = (p[cols[0]] >= p[cols[1]]) & (p[cols[1]] >= p[cols[2]]) & \
         (p[cols[2]] >= p[cols[3]])
    assert bool(ok.all()), f"{int((~ok).sum())} rows violate line monotonicity"
