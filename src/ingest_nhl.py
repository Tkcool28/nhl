"""Phase 1: pull raw NHL API data. Train 2018-19..2023-24, validate 2024-25.

HARD GUARD: 2025-26 (the sealed holdout) is never requested. Any attempt raises.
Threaded (8 workers) with per-worker rate politeness; resumable via file skip.
"""
import json, time, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

WEB = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"

TRAIN_SEASONS = [20182019, 20192020, 20202021, 20212022, 20222023, 20232024]
VALID_SEASON = 20242025
SEALED_HOLDOUT = 20252026
ALL_SEASONS = TRAIN_SEASONS + [VALID_SEASON]

assert SEALED_HOLDOUT not in ALL_SEASONS, "holdout season must never be pulled"

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

def guard_season(season: int):
    if season == SEALED_HOLDOUT:
        raise RuntimeError("REFUSED: 2025-26 is the sealed holdout")

_local = threading.local()

def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "nhl-sog-research/1.0"})
        _local.s = s
    return _local.s

def get(url, tries=4):
    s = session()
    for i in range(tries):
        r = s.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** i)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
    raise RuntimeError(f"failed after retries: {url}")

def skater_ids(season: int):
    """All skaters (no goalies) for a season via bulk stats API."""
    guard_season(season)
    out, start = [], 0
    while True:
        url = (f"{STATS}/skater/summary?limit=100&start={start}"
               f"&cayenneExp=seasonId={season}")
        d = get(url)
        rows = d["data"]
        if not rows:
            break
        for r in rows:
            if r.get("positionCode") != "G":
                out.append({"playerId": r["playerId"],
                            "name": r["skaterFullName"],
                            "pos": r["positionCode"]})
        start += 100
        time.sleep(0.15)
    return out

def game_log(pid: int, season: int):
    guard_season(season)
    url = f"{WEB}/player/{pid}/game-log/{season}/2"
    d = get(url)
    if not d:
        return []
    rows = []
    for g in d.get("gameLog", []):
        # hard guard: no holdout-season games even if API misbehaves
        if g.get("gameDate", "") >= "2025-08-01":
            raise RuntimeError(f"REFUSED: gameDate {g['gameDate']} in holdout window")
        rows.append({
            "gameId": g["gameId"], "gameDate": g["gameDate"],
            "team": g["teamAbbrev"], "opp": g["opponentAbbrev"],
            "homeRoad": g["homeRoadFlag"], "shots": g["shots"],
            "toi": g["toi"], "shifts": g["shifts"], "pim": g["pim"],
            "goals": g["goals"], "assists": g["assists"],
            "ppGoals": g["powerPlayGoals"], "ppPoints": g["powerPlayPoints"],
            "plusMinus": g["plusMinus"],
        })
    return rows

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    lock = threading.Lock()
    for season in ALL_SEASONS:
        guard_season(season)
        sdir = RAW / "game_logs" / str(season)
        sdir.mkdir(parents=True, exist_ok=True)
        players = skater_ids(season)
        manifest[str(season)] = {"n_players": len(players)}
        print(f"season {season}: {len(players)} skaters", flush=True)
        todo = [p for p in players if not (sdir / f"{p['playerId']}.json").exists()]
        print(f"  {len(todo)} remaining (skipping cached)", flush=True)
        done = [0]

        def fetch(p):
            fp = sdir / f"{p['playerId']}.json"
            try:
                rows = game_log(p["playerId"], season)
            except Exception as e:
                with lock:
                    print(f"  WARN {p['playerId']}: {e}", flush=True)
                return
            fp.write_text(json.dumps({"meta": p, "games": rows}))
            with lock:
                done[0] += 1
                if done[0] % 200 == 0:
                    print(f"  {season}: {done[0]}/{len(todo)}", flush=True)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(fetch, todo))
        print(f"season {season} complete", flush=True)
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("ingest complete", flush=True)

if __name__ == "__main__":
    main()
