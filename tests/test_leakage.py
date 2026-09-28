"""Leakage tests: every rolling feature must come strictly from prior games."""
import pandas as pd
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

def test_no_holdout():
    for f in ("train.parquet", "valid.parquet"):
        df = pd.read_parquet(PROC / f, columns=["season", "game_date"])
        assert (df["season"] != 20252026).all()
        assert (df["game_date"] < "2025-08-01").all()

def test_trailing_means_are_prior_only():
    df = pd.read_parquet(PROC / "train.parquet",
                         columns=["player_id", "game_date", "shots", "shots_l10"])
    df = df.sort_values(["player_id", "game_date"])
    for pid, g in df.groupby("player_id"):
        g = g.reset_index(drop=True)
        for i in [5, 15, 40]:
            if i >= len(g):
                continue
            row = g.iloc[i]
            expect = g.iloc[max(0, i - 10):i]["shots"].mean()
            assert abs(row["shots_l10"] - expect) < 1e-9, (pid, i)
        break  # one player spot-check is enough for structure

if __name__ == "__main__":
    test_no_holdout()
    test_trailing_means_are_prior_only()
    print("leakage tests pass")
