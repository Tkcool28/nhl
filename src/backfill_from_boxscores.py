"""One-time data repair (2026-09-28): backfill real games the game-log API omits.

Two 2024-25 player-seasons have games in official boxscores (real ice time,
real shots) that the /player/{id}/game-log/{season}/2 endpoint never
returns, even on refetch:
  - 8481848 Jacob Gaucher: 4 games (2025-02-02 .. 2025-02-08)
  - 8478400 Colin White:   3 games (discovered by scan)

Boxscore fields were verified to match the game-log schema on overlapping
games (sog==shots, toi/shifts/pim/goals/assists/plusMinus identical;
null powerPlayPoints -> 0). Rows are built in the exact game-log schema and
merged into the player's raw file sorted by (gameDate, gameId).

This is a same-source data repair (official NHL API boxscore -> official
NHL API game-log schema), not modeling. Recorded in the manifest.
The reconciliation gate must pass at zero after this runs.
"""
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

WEB = "https://api-web.nhle.com/v1"
TEAMS = ["ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
         "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
         "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
         "VGK", "WPG"]
SEASON = 20242025
PAIRS = [(8481848, "Jacob Gaucher"), (8478400, "Colin White")]

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SESS = requests.Session()
SESS.headers.update({"User-Agent": "nhl-sog-research/1.0"})


def get(url, tries=6):
    import time
    for i in range(tries):
        r = SESS.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed: {url}")


def toi_min(s):
    try:
        m, sec = str(s).split(":")
        return int(m) + int(sec) / 60
    except Exception:
        return 0.0


def main():
    # official 2024-25 game list
    game_ids = set()

    def sched(team):
        d = get(f"{WEB}/club-schedule-season/{team}/{SEASON}")
        return [g["id"] for g in d.get("games", []) if g.get("gameType") == 2]

    with ThreadPoolExecutor(max_workers=16) as ex:
        for ids in ex.map(sched, TEAMS):
            game_ids.update(ids)
    print(f"{len(game_ids)} games to scan", flush=True)

    targets = {pid: name for pid, name in PAIRS}
    have = {}
    for pid, name in PAIRS:
        fp = RAW / "game_logs" / str(SEASON) / f"{pid}.json"
        d = json.loads(fp.read_text())
        have[pid] = {g["gameId"] for g in d.get("games", [])}

    def scan(gid):
        b = get(f"{WEB}/gamecenter/{gid}/boxscore")
        out = []
        for side in ("homeTeam", "awayTeam"):
            for k in ("forwards", "defense"):
                for p in b["playerByGameStats"][side].get(k, []):
                    pid = p["playerId"]
                    if pid in targets and toi_min(p.get("toi")) > 0 \
                            and gid not in have[pid]:
                        team = b["homeTeam"]["abbrev"] if side == "homeTeam" \
                            else b["awayTeam"]["abbrev"]
                        opp = b["awayTeam"]["abbrev"] if side == "homeTeam" \
                            else b["homeTeam"]["abbrev"]
                        out.append((pid, {
                            "gameId": gid, "gameDate": b["gameDate"],
                            "team": team, "opp": opp,
                            "homeRoad": "H" if side == "homeTeam" else "R",
                            "shots": p.get("sog", 0),
                            "toi": p.get("toi"),
                            "shifts": p.get("shifts", 0),
                            "pim": p.get("pim", 0),
                            "goals": p.get("goals", 0),
                            "assists": p.get("assists", 0),
                            "ppGoals": p.get("powerPlayGoals") or 0,
                            "ppPoints": p.get("powerPlayPoints") or 0,
                            "plusMinus": p.get("plusMinus", 0),
                        }))
        return out

    grouped = {pid: [] for pid in targets}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for rows in ex.map(scan, sorted(game_ids)):
            for pid, row in rows:
                grouped[pid].append(row)

    manifest = []
    for pid, name in PAIRS:
        rows = sorted(grouped[pid], key=lambda r: (r["gameDate"], r["gameId"]))
        # schema check: every row must have all 14 game-log fields, no nulls
        need = ["gameId", "gameDate", "team", "opp", "homeRoad", "shots",
                "toi", "shifts", "pim", "goals", "assists", "ppGoals",
                "ppPoints", "plusMinus"]
        for r in rows:
            assert all(k in r and r[k] is not None for k in need), r
            assert r["gameDate"] < "2025-08-01", "holdout guard"
        fp = RAW / "game_logs" / str(SEASON) / f"{pid}.json"
        d = json.loads(fp.read_text())
        before = len(d["games"])
        d["games"] = sorted(d["games"] + rows,
                            key=lambda r: (r["gameDate"], r["gameId"]))
        # keep meta honest
        d["meta"]["backfilled_from_boxscore"] = len(rows)
        fp.write_text(json.dumps(d))
        manifest.append({"season": SEASON, "playerId": pid, "name": name,
                         "n_backfilled": len(rows),
                         "gameIds": [r["gameId"] for r in rows],
                         "before": before, "after": len(d["games"])})
        print(f"{name}: {before} -> {len(d['games'])} "
              f"(+{len(rows)} from boxscores)", flush=True)

    (RAW / "boxscore_backfills.json").write_text(
        json.dumps(manifest, indent=1))
    print("wrote boxscore_backfills.json", flush=True)


if __name__ == "__main__":
    main()
