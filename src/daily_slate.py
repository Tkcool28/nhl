"""Daily production slate scorer (NOT frozen -- this is the live pipeline).

Each morning:
  1. Refresh 2026-27 skater game logs (full re-pull; ~950 players).
  2. Run the FROZEN features.build() over all seasons (fallbacks.json
     snapshot/restored -- never rewritten with new values).
  3. Build placeholder rows for today's games -> score with the frozen
     model via src/predict.py (immutable prediction records).
  4. Write docs/data/slate_YYYY-MM-DD.json + docs/data/latest.json for
     the GitHub Pages frontend.
  5. Settle previously logged market observations whose games are final.

Usage: python src/daily_slate.py [--date YYYY-MM-DD] [--skip-ingest]
"""
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_nhl as ing
import features as feat

REPO = Path(__file__).resolve().parent.parent
DENVER = ZoneInfo("America/Denver")
SEASON = 20262027
SEASON_START = "2026-08-01"
SEASON_END = "2027-08-01"
SEASON_DIR = ing.RAW / "game_logs" / str(SEASON)
DOCS_DATA = REPO / "docs" / "data"
ELIGIBLE_TOI = 13.0


# ------------------------------------------------------------------ ingest
def bulk_skaters(season):
    seen, start = {}, 0
    while True:
        url = (f"{ing.STATS}/skater/summary?limit=100&start={start}"
               f"&cayenneExp=seasonId={season}")
        d = ing.get(url)
        rows = d["data"]
        if not rows:
            break
        for r in rows:
            if r.get("positionCode") != "G":
                seen[r["playerId"]] = {"playerId": r["playerId"],
                                      "name": r["skaterFullName"],
                                      "pos": r["positionCode"]}
        start += 100
    return [seen[k] for k in sorted(seen)]


def season_game_log(pid, season):
    url = f"{ing.WEB}/player/{pid}/game-log/{season}/2"
    d = ing.get(url)
    if d is None:
        return [], 0
    rows, n_phantom = [], 0
    for g in d.get("gameLog", []):
        gd = g.get("gameDate", "")
        if not (SEASON_START <= gd < SEASON_END):
            raise RuntimeError(f"gameDate {gd} outside season window")
        if not g.get("toi"):
            n_phantom += 1
            continue
        rows.append({
            "gameId": g["gameId"], "gameDate": gd,
            "team": g["teamAbbrev"], "opp": g["opponentAbbrev"],
            "homeRoad": g["homeRoadFlag"], "shots": g["shots"],
            "toi": g["toi"], "shifts": g["shifts"], "pim": g["pim"],
            "goals": g["goals"], "assists": g["assists"],
            "ppGoals": g["powerPlayGoals"], "ppPoints": g["powerPlayPoints"],
            "plusMinus": g["plusMinus"],
        })
    return rows, n_phantom


def refresh_ingest():
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    by_id = {p["playerId"]: p for p in bulk_skaters(SEASON)}
    if SEASON_DIR.exists():
        for fp in SEASON_DIR.glob("*.json"):
            pid = int(fp.stem)
            by_id.setdefault(pid, {"playerId": pid, "name": "?", "pos": "?"})
    players = [by_id[k] for k in sorted(by_id)]
    print(f"refreshing {len(players)} skaters for {SEASON}", flush=True)
    failures, done = [], [0]
    lock = threading.Lock()

    def fetch(p):
        fp = SEASON_DIR / f"{p['playerId']}.json"
        try:
            rows, _ = season_game_log(p["playerId"], SEASON)
        except Exception as e:
            with lock:
                failures.append({"playerId": p["playerId"], "error": str(e)[:200]})
            return
        meta = dict(p)
        if meta.get("name") == "?":
            try:
                old = json.loads(fp.read_text())["meta"]
                meta = old
            except Exception:
                pass
        fp.write_text(json.dumps({"meta": meta, "games": rows}))
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(players)}", flush=True)

    with ThreadPoolExecutor(max_workers=ing._MAX_WORKERS) as ex:
        list(ex.map(fetch, players))
    if failures:
        print(f"FATAL: {len(failures)} fetch failures", flush=True)
        sys.exit(1)
    print(f"ingest refresh complete: {len(players)} players", flush=True)


# ------------------------------------------------------------------- build
def load_all_seasons():
    rows = []
    for sdir in sorted(feat.RAW.iterdir()):
        season = int(sdir.name)
        for fp in sdir.glob("*.json"):
            d = json.loads(fp.read_text())
            pid = d["meta"]["playerId"]
            for g in d["games"]:
                rows.append({
                    "player_id": pid, "name": d["meta"]["name"],
                    "pos": d["meta"]["pos"],
                    "season": season, "game_id": g["gameId"],
                    "game_date": g["gameDate"],
                    "team": g["team"], "opp": g["opp"],
                    "home": 1 if g["homeRoad"] == "H" else 0,
                    "shots": g["shots"], "toi": feat.parse_toi(g["toi"]),
                    "shifts": g["shifts"], "pim": g["pim"],
                    "points": g["goals"] + g["assists"],
                    "pp_points": g["ppPoints"],
                })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def todays_games(date_str):
    d = ing.get(f"{ing.WEB}/scoreboard/{date_str}")
    games = []
    for day in d.get("gamesByDate", []):
        if day.get("date") != date_str:
            continue
        for g in day.get("games", []):
            if g.get("gameType") != 2:
                continue
            games.append({"game_id": g["id"],
                          "home": g["homeTeam"]["abbrev"],
                          "away": g["awayTeam"]["abbrev"],
                          "date": g["gameDate"][:10]})
    return games


def frozen_build(df):
    """Run the frozen features.build() with fallbacks.json snapshot/restore."""
    fb_path = feat.OUT / "fallbacks.json"
    fb_before = fb_path.read_bytes()
    built = feat.build(df)
    if fb_path.read_bytes() != fb_before:
        fb_path.write_bytes(fb_before)
        raise RuntimeError("FATAL: build() changed frozen fallbacks.json")
    return built


def build_slate(date_str, games, df):
    """Placeholder rows for today's games; trailing features then reflect
    history through yesterday (shift(1) inside build)."""
    if not games:
        return None
    hist = frozen_build(df)
    latest = (hist.sort_values("game_date").groupby("player_id").tail(1)
              .set_index("player_id"))
    # only players active in the current or prior season -- no ghosts whose
    # old team happens to play today
    latest = latest[latest["season"] >= SEASON - 10001]
    gid_map = {}
    for g in games:
        gid_map[g["home"]] = (g["game_id"], g["away"], 1)
        gid_map[g["away"]] = (g["game_id"], g["home"], 0)
    ph = []
    for pid, r in latest.iterrows():
        if pd.isna(r["toi_l10"]) or r["toi_l10"] < ELIGIBLE_TOI:
            continue
        if r["team"] not in gid_map:
            continue
        gid, opp, home = gid_map[r["team"]]
        ph.append({"player_id": pid, "name": r["name"], "pos": r["pos"],
                   "season": SEASON, "game_id": gid, "game_date": date_str,
                   "team": r["team"], "opp": opp, "home": home,
                   "shots": np.nan, "toi": np.nan, "shifts": np.nan,
                   "pim": np.nan, "points": np.nan, "pp_points": np.nan})
    if not ph:
        return None
    phdf = pd.DataFrame(ph)
    phdf["game_date"] = pd.to_datetime(phdf["game_date"])
    full = pd.concat([df, phdf], ignore_index=True)
    full = full.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return full


# ------------------------------------------------------------------- score
def score_placeholders(full, date_str):
    built = frozen_build(full)
    mask = (built["season"] == SEASON) & (built["game_date"] == date_str)
    slate = built[mask].copy().reset_index(drop=True)
    assert len(slate) > 0
    # Canonical immutable prediction log (MARKET_LOGGING.md): one record per
    # player-game, every skater on the dash, no clicks required.
    log_dir = REPO / "data" / "market_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"predictions_{date_str}.jsonl"
    tmp_in = Path(f"/tmp/slate_{date_str}.parquet")
    slate.to_parquet(tmp_in, index=False)
    if not log_path.exists():
        r = subprocess.run(
            [sys.executable, str(REPO / "src" / "predict.py"),
             "--input", str(tmp_in), "--out", str(log_path),
             "--run-id", f"slate-{date_str}"],
            capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)
    else:
        print(f"prediction log exists, reusing {log_path}", flush=True)
    preds = pd.read_json(log_path, lines=True)
    return preds


# ------------------------------------------------------------------ settle
def settle_logs():
    """Join actuals onto open market observations; append settlements."""
    log_dir = REPO / "data" / "market_log"
    if not log_dir.exists():
        return
    actual = {}
    for sdir in sorted((ing.RAW / "game_logs").iterdir()):
        for fp in sdir.glob("*.json"):
            pid = int(fp.stem)
            for g in json.loads(fp.read_text()).get("games", []):
                actual[(g["gameId"], pid)] = g["shots"]
    settled_path = log_dir / "settlements.jsonl"
    done = set()
    if settled_path.exists():
        for line in settled_path.read_text().splitlines():
            o = json.loads(line)
            done.add(o["observation_id"])
    n = 0
    with open(settled_path, "a") as f:
        for fp in sorted(log_dir.glob("obs_*.jsonl")):
            for line in fp.read_text().splitlines():
                o = json.loads(line)
                oid = o["observation_id"]
                if oid in done:
                    continue
                key = (o["game_id"], o["player_id"])
                if key not in actual:
                    continue  # game not final yet
                sog = actual[key]
                L = float(o["line"])
                result = "over" if sog > L else ("under" if sog < L else "push")
                f.write(json.dumps({
                    "observation_id": oid, "game_id": o["game_id"],
                    "player_id": o["player_id"], "line": o["line"],
                    "actual_sog": sog, "result": result,
                    "settled_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                n += 1
    if n:
        print(f"settled {n} observations", flush=True)


# -------------------------------------------------------------------- main
def main():
    ap_date = sys.argv[sys.argv.index("--date") + 1] if "--date" in sys.argv else None
    date_str = ap_date or datetime.now(DENVER).strftime("%Y-%m-%d")
    print(f"slate date: {date_str}", flush=True)

    if "--skip-ingest" not in sys.argv:
        refresh_ingest()
    df = load_all_seasons()
    games = todays_games(date_str)
    print(f"{len(games)} games today", flush=True)

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(REPO / "models" / "v1" / "config.json"))
    alpha = cfg["alpha"]["gbm_nb"]

    slate = {"slate_date": date_str,
             "generated_at": datetime.now(timezone.utc).isoformat(),
             "baseline_version": "v1.2",
             "games": games, "players": [], "note": None,
             "alpha": alpha}
    if games:
        full = build_slate(date_str, games, df)
        if full is None:
            slate["note"] = "no eligible skaters found for today's games"
        else:
            preds = score_placeholders(full, date_str)
            preds = preds[preds["eligible"]].reset_index(drop=True)
            game_by_id = {g["game_id"]: g for g in games}
            for _, r in preds.iterrows():
                rec = json.loads(json.dumps(r.to_dict(), default=str))
                rec["alpha"] = alpha
                g = game_by_id.get(rec["game_id"], {})
                rec["matchup"] = f"{g.get('away', '?')} @ {g.get('home', '?')}"
                slate["players"].append(rec)
            slate["run_id"] = f"slate-{date_str}"
            print(f"scored {len(preds)} eligible player-games", flush=True)
    else:
        slate["note"] = "no NHL regular-season games scheduled"

    out = DOCS_DATA / f"slate_{date_str}.json"
    out.write_text(json.dumps(slate))
    (DOCS_DATA / "latest.json").write_text(json.dumps(slate))
    print(f"wrote {out}", flush=True)
    settle_logs()


if __name__ == "__main__":
    main()
