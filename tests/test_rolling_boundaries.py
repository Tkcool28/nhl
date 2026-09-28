"""Boundary poison tests for rolling feature construction (v1.2 correctness refreeze).

Bug under test (2026-09-28): src/features.py used patterns like
    g[col].shift(1).rolling(n, min_periods=1).mean()
where the shift is group-scoped but the rolling window runs over the
flattened, player-sorted frame. Rolling windows therefore cross from the
end of Player A into the start of Player B (and Team A into Team B):
genuine temporal leakage.

Intended design (preserved): player history DOES carry across seasons --
group by player_id only, never (player_id, season) -- so opening night
uses prior-season history. The season test below pins that behavior.

Run these BEFORE the fix: A and B must FAIL (contamination), C must PASS
(carryover already works). After the fix: all pass.
"""
import numpy as np
import pandas as pd
import pytest

import features

SEASON = 20232024


def make_row(pid, name, team, opp, gid, date, shots, toi, season=SEASON,
             pim=0, goals=0, assists=0, pp_points=0, home_road="H", shifts=20):
    return {
        "player_id": pid, "name": name, "pos": "C",
        "season": season, "game_id": gid, "game_date": pd.Timestamp(date),
        "team": team, "opp": opp, "home": 1 if home_road == "H" else 0,
        "shots": shots, "toi": float(toi), "shifts": shifts, "pim": pim,
        "points": goals + assists, "pp_points": pp_points,
    }


def run_build(rows, tmp_path, monkeypatch):
    """build() on synthetic rows; OUT redirected so fallbacks.json is not clobbered."""
    monkeypatch.setattr(features, "OUT", tmp_path)
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return features.build(df)


def poison_victim_rows():
    """Player 1 (AAA): extreme values. Player 2 (ZZZ): known low values.
    Same game_ids/dates so team aggregates are deterministic."""
    rows = []
    base = pd.Timestamp("2023-10-01")
    for i in range(15):
        d = base + pd.Timedelta(days=2 * i)
        hr = "H" if i % 2 == 0 else "R"
        rows.append(make_row(1, "Poison", "AAA", "ZZZ", f"G{i:03d}", d,
                             shots=100 + i, toi=30.0, pim=50,
                             goals=30, assists=30, pp_points=50, home_road=hr))
    for i in range(12):
        d = base + pd.Timedelta(days=2 * i)
        hr = "H" if i % 2 == 0 else "R"
        rows.append(make_row(2, "Victim", "ZZZ", "AAA", f"G{i:03d}", d,
                             shots=1, toi=10.0, pim=0,
                             goals=0, assists=1, pp_points=0, home_road=hr))
    return rows


PLAYER_COLS = ["shots_l5", "shots_l10", "shots_l20", "toi_l10", "naive_l10",
               "shots_per60_l10", "shots_home_l10", "shots_road_l10"]


def test_player_boundary_no_contamination(tmp_path, monkeypatch):
    df = run_build(poison_victim_rows(), tmp_path, monkeypatch)
    v = df[df.player_id == 2].reset_index(drop=True)
    train_mean = df["shots"].mean()

    # First victim game: no prior history exists anywhere for player 2.
    for col in PLAYER_COLS:
        assert pd.isna(v.loc[0, col]), f"{col} contaminated on first row"
    assert pd.isna(v.loc[0, "pp_share_l10"]), "pp_share contaminated on first row"

    # Second victim game: only victim's own game 0 may contribute.
    assert v.loc[1, "shots_l10"] == 1.0
    assert v.loc[1, "shots_l5"] == 1.0
    assert v.loc[1, "toi_l10"] == 10.0
    assert v.loc[1, "naive_l10"] == 1.0
    assert v.loc[1, "pp_share_l10"] == 0.0
    assert v.loc[1, "shots_per60_l10"] == pytest.approx(1 / 10 * 60)
    # home/road splits: game 0 was home, game 1 is road
    assert v.loc[1, "shots_home_l10"] == 1.0
    assert pd.isna(v.loc[1, "shots_road_l10"])

    # eb_pred on victim's first game: no trailing shots exist, so the raw
    # value is NaN -- identical to what the old code produced for the first
    # player in the frame. Downstream clean_mu() fills it with prior_pg.
    # (Under the bug it was ~76: poison's ~100-shot games, n10=10.)
    assert pd.isna(v.loc[0, "eb_pred"])
    # victim game 1: s10=1, n10=1 from victim's own game 0 only
    k = features.K_SHRINK
    assert v.loc[1, "eb_pred"] == pytest.approx((1 + k * train_mean) / (1 + k))


def test_poison_player_own_rows_exact(tmp_path, monkeypatch):
    """Sanity: with varying poison values, the poison player's own trailing
    means are exactly computable -- and must exclude the current game."""
    df = run_build(poison_victim_rows(), tmp_path, monkeypatch)
    p = df[df.player_id == 1].reset_index(drop=True)
    # game 5: trailing-5 = games 0..4 -> mean(100..104) = 102.0
    # (if the current game leaked in it would be 102.5)
    assert p.loc[5, "shots_l5"] == pytest.approx(102.0)
    assert p.loc[10, "shots_l10"] == pytest.approx(np.mean(range(100, 110)))
    assert p.loc[3, "toi_l10"] == pytest.approx(30.0)
    # toi_trend = trailing-5 mean - trailing-10 mean, constant toi -> 0
    assert p.loc[10, "toi_trend"] == pytest.approx(0.0)


def test_team_boundary_no_contamination(tmp_path, monkeypatch):
    df = run_build(poison_victim_rows(), tmp_path, monkeypatch)
    v = df[df.player_id == 2].reset_index(drop=True)  # team ZZZ
    # ZZZ's first team-game has no prior ZZZ history: all NaN.
    for col in ["team_shots_for_l10", "team_shots_against_l10",
                "team_pim_l10", "team_pace_l10",
                "opp_shots_for_l10", "opp_shots_against_l10",
                "opp_pim_l10", "opp_pace_l10"]:
        assert pd.isna(v.loc[0, col]), f"{col} contaminated on first ZZZ game"
    # Second ZZZ game: only ZZZ's own game 0 (shots_for=1, pim=0).
    assert v.loc[1, "team_shots_for_l10"] == 1.0
    assert v.loc[1, "team_pim_l10"] == 0.0
    # ZZZ's shots_against in game 0 = AAA's shots_for = 100 (legitimate,
    # same-game opponent total -- but only from ZZZ's own prior games).
    assert v.loc[1, "team_shots_against_l10"] == 100.0
    assert v.loc[1, "team_pace_l10"] == 101.0


def test_season_carryover_intact(tmp_path, monkeypatch):
    """Intended design: history carries across seasons (no reset), strictly
    chronological. First game of 2019-20 uses 2018-19 history only."""
    rows = []
    for i in range(10):
        rows.append(make_row(9, "Carry", "AAA", "ZZZ", f"S1{i:02d}",
                             f"2018-10-{i + 1:02d}", shots=5, toi=15.0,
                             season=20182019))
    for i in range(5):
        rows.append(make_row(9, "Carry", "AAA", "ZZZ", f"S2{i:02d}",
                             f"2019-10-{i + 1:02d}", shots=1, toi=12.0,
                             season=20192020))
    df = run_build(rows, tmp_path, monkeypatch)
    s2 = df[df.season == 20192020].reset_index(drop=True)

    # carryover: opening night sees the prior season's 5.0 rate, not a reset
    assert s2.loc[0, "shots_l10"] == 5.0
    assert s2.loc[0, "shots_l5"] == 5.0
    assert s2.loc[0, "n_trailing"] == 10
    # current game excluded: (9*5 + 1)/10 = 4.6 would mean leakage
    assert s2.loc[0, "shots_l10"] != pytest.approx(4.6)

    # future-row poison invariance: changing a LATER game in 2019-20 must
    # not change opening-night features
    rows2 = [dict(r) for r in rows]
    rows2[-1]["shots"] = 999
    df2 = run_build(rows2, tmp_path, monkeypatch)
    s2b = df2[df2.season == 20192020].reset_index(drop=True)
    for col in ["shots_l5", "shots_l10", "shots_l20", "toi_l10",
                "naive_l10", "eb_pred", "shots_per60_l10"]:
        assert s2.loc[0, col] == pytest.approx(s2b.loc[0, col]), col
