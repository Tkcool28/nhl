"""Phase 2: feature engineering. Strict prior-games-only construction.

Every rolling feature uses games strictly before the current game_date
(shift(1)), reset at season boundaries. 2025-26 is refused outright.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "game_logs"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
SEALED = 20252026
TRAIN_END = 20232024
VALID_SEASON = 20242025
K_SHRINK = 20  # empirical-Bayes shrinkage strength (pre-registered)

def parse_toi(s):
    try:
        m, sec = s.split(":")
        return int(m) + int(sec) / 60
    except Exception:
        return np.nan

def load():
    rows = []
    for sdir in sorted(RAW.iterdir()):
        season = int(sdir.name)
        assert season != SEALED, "REFUSED: sealed holdout"
        for fp in sdir.glob("*.json"):
            d = json.loads(fp.read_text())
            pid = d["meta"]["playerId"]
            for g in d["games"]:
                if g["gameDate"] >= "2025-08-01":
                    raise RuntimeError("REFUSED: holdout-window gameDate")
                rows.append({
                    "player_id": pid, "name": d["meta"]["name"], "pos": d["meta"]["pos"],
                    "season": season, "game_id": g["gameId"], "game_date": g["gameDate"],
                    "team": g["team"], "opp": g["opp"], "home": 1 if g["homeRoad"] == "H" else 0,
                    "shots": g["shots"], "toi": parse_toi(g["toi"]), "shifts": g["shifts"],
                    "pim": g["pim"], "points": g["goals"] + g["assists"],
                    "pp_points": g["ppPoints"],
                })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

def trailing(g, col, n):
    return g[col].shift(1).rolling(n, min_periods=1).mean()

def trailing_sum(g, col, n):
    return g[col].shift(1).rolling(n, min_periods=1).sum()

def build(df):
    g = df.groupby("player_id", group_keys=False)
    df["n_trailing"] = g.cumcount()  # games before this one
    for n in (5, 10, 20):
        df[f"shots_l{n}"] = trailing(g, "shots", n)
    df["toi_l10"] = trailing(g, "toi", 10)
    df["toi_trend"] = trailing(g, "toi", 5) - trailing(g, "toi", 10)
    s_sum = trailing_sum(g, "shots", 10)
    t_sum = trailing_sum(g, "toi", 10)
    df["shots_per60_l10"] = s_sum / t_sum.replace(0, np.nan) * 60
    pp = trailing_sum(g, "pp_points", 10)
    pts = trailing_sum(g, "points", 10)
    df["pp_share_l10"] = pp / pts.replace(0, np.nan)
    # home/road splits: trailing means within venue type (vectorized)
    df["_h_shots"] = np.where(df["home"] == 1, df["shots"], np.nan)
    df["_r_shots"] = np.where(df["home"] == 0, df["shots"], np.nan)
    df["shots_home_l10"] = g["_h_shots"].shift(1).rolling(10, min_periods=1).mean()
    df["shots_road_l10"] = g["_r_shots"].shift(1).rolling(10, min_periods=1).mean()
    df.drop(columns=["_h_shots", "_r_shots"], inplace=True)
    # rest / back-to-back
    df["rest_days"] = g["game_date"].diff().dt.days
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df["month"] = df["game_date"].dt.month

    # ---- team aggregates from the same logs (no extra API calls) ----
    tg = (df.groupby(["game_id", "team"], as_index=False)
            .agg(shots_for=("shots", "sum"), pim_for=("pim", "sum"),
                 game_date=("game_date", "first")))
    # attach opponent: the other team in the same game
    opp = tg[["game_id", "team", "shots_for"]].rename(
        columns={"team": "opp", "shots_for": "shots_against"})
    # attach opponent via self-merge on game_id (2 teams per game)
    m = tg.merge(tg[["game_id", "team", "shots_for"]], on="game_id",
                 suffixes=("", "_opp"))
    m = m[m["team"] != m["team_opp"]]
    m = m.rename(columns={"shots_for_opp": "shots_against", "team_opp": "opp_team"})
    team_game = m[["team", "game_date", "shots_for", "shots_against", "pim_for"]].copy()
    team_game = team_game.sort_values(["team", "game_date"])
    tg2 = team_game.groupby("team", group_keys=False)
    team_game["team_shots_for_l10"] = tg2["shots_for"].shift(1).rolling(10, min_periods=1).mean()
    team_game["team_shots_against_l10"] = tg2["shots_against"].shift(1).rolling(10, min_periods=1).mean()
    team_game["team_pim_l10"] = tg2["pim_for"].shift(1).rolling(10, min_periods=1).mean()
    team_game["team_pace_l10"] = (team_game["team_shots_for_l10"]
                                  + team_game["team_shots_against_l10"])
    key = team_game[["team", "game_date", "team_shots_for_l10",
                     "team_shots_against_l10", "team_pim_l10", "team_pace_l10"]]
    n0 = len(df)
    df = df.merge(key, on=["team", "game_date"], how="left")
    assert len(df) == n0, "team merge duplicated rows"
    oppkey = key.rename(columns={"team": "opp",
                                 "team_shots_for_l10": "opp_shots_for_l10",
                                 "team_shots_against_l10": "opp_shots_against_l10",
                                 "team_pim_l10": "opp_pim_l10",
                                 "team_pace_l10": "opp_pace_l10"})
    df = df.merge(oppkey, on=["opp", "game_date"], how="left")
    assert len(df) == n0, "opp merge duplicated rows"

    # ---- empirical-Bayes shrinkage prior (also the EB baseline) ----
    # prior season rate shrunk to league mean; no prior season ->
    # prior season's league mean, else frozen training-only mean.
    # BUGFIX 2026-09-27: league means and the fallback are computed on
    # TRAINING seasons only. Previously the fallback used the all-seasons
    # mean, leaking the validation season's own mean into validation rows.
    df_tr = df[df["season"] <= TRAIN_END]
    pg = (df_tr.groupby(["player_id", "season"], as_index=False)
            .agg(pg_shots=("shots", "mean"), pg_n=("shots", "size")))
    league = df_tr.groupby("season")["shots"].mean().to_dict()
    train_mean = float(df_tr["shots"].mean())
    (OUT / "fallbacks.json").write_text(json.dumps({
        "train_mean_shots": train_mean,
        "league_mean_by_season": {str(k): float(v) for k, v in league.items()},
        "K_SHRINK": K_SHRINK,
        "note": "frozen at feature-build time; validation must reuse these exact values"}, indent=1))
    pg["league_mean"] = pg["season"].map(league)
    pg["prior_pg"] = ((pg["pg_n"] * pg["pg_shots"] + K_SHRINK * pg["league_mean"])
                      / (pg["pg_n"] + K_SHRINK))
    pg["next_season"] = pg["season"] + 10001  # 20182019 -> 20192020
    n0 = len(df)
    df = df.merge(pg[["player_id", "next_season", "prior_pg"]],
                  left_on=["player_id", "season"],
                  right_on=["player_id", "next_season"], how="left")
    assert len(df) == n0, "prior_pg merge duplicated rows"
    fallback = {s: league.get(s - 10001, train_mean) for s in df["season"].unique()}
    df["prior_pg"] = df["prior_pg"].fillna(df["season"].map(fallback))
    assert df["prior_pg"].notna().all(), "prior_pg still has NaNs after fallback"
    df.drop(columns=["next_season"], inplace=True)
    # EB prediction: trailing-10 blended with prior
    s10 = trailing_sum(g, "shots", 10)
    n10 = g["shots"].shift(1).rolling(10, min_periods=1).count()
    df["eb_pred"] = (s10 + K_SHRINK * df["prior_pg"]) / (n10 + K_SHRINK)
    df["naive_l10"] = trailing(g, "shots", 10)
    return df

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build(load())
    assert (df["season"] != SEALED).all()
    assert (df["game_date"] < "2025-08-01").all()
    train = df[df["season"] <= TRAIN_END].copy()
    valid = df[df["season"] == VALID_SEASON].copy()
    assert len(valid) > 0 and len(train) > len(valid)
    train.to_parquet(OUT / "train.parquet", index=False)
    valid.to_parquet(OUT / "valid.parquet", index=False)
    print(f"train rows: {len(train):,}  valid rows: {len(valid):,}", flush=True)
    print(f"train seasons: {sorted(train.season.unique())}", flush=True)

if __name__ == "__main__":
    main()
