"""Gate: raw ingest must reconcile against official boxscores.

For every regular-season game 2018-19..2024-25, every skater listed in the
official boxscore (forwards + defense) must have a raw game-log file for
that season containing that game_id.

Exit 0 iff zero missing player-games. Anything missing -> nonzero exit and
the pipeline must not proceed. Run after every ingest (full or resume).

This is the check that caught the ~2% silent data loss on 2026-09-27.
"""
import json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

WEB = "https://api-web.nhle.com/v1"
TEAMS = ["ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
         "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
         "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
         "VGK", "WPG"]
SEASONS = [20182019, 20192020, 20202021, 20212022, 20222023, 20232024, 20242025]
SEALED_HOLDOUT = 20252026

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def season_of_game(gid: int) -> int:
    """game_id 2018020001 -> seasonId 20182019."""
    y = gid // 1000000
    return y * 10000 + (y + 1)


def get(url, tries=6):
    for i in range(tries):
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "nhl-sog-research/1.0"})
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed: {url}")


def _toi_minutes(s) -> float:
    try:
        m, sec = str(s).split(":")
        return int(m) + int(sec) / 60
    except Exception:
        return 0.0


def main():
    # 1. index raw files: game_id -> set(player_id)
    game_index, n_files = {}, 0
    for season in SEASONS:
        sdir = RAW / "game_logs" / str(season)
        for fp in sdir.glob("*.json"):
            n_files += 1
            try:
                d = json.loads(fp.read_text())
            except Exception as e:
                print(f"FATAL: unreadable raw file {fp}: {e}", flush=True)
                sys.exit(1)
            pid = int(fp.stem)
            for g in d.get("games", []):
                if g.get("gameDate", "") >= "2025-08-01":
                    print(f"FATAL: holdout-window game {g['gameId']} in {fp}",
                          flush=True)
                    sys.exit(1)
                game_index.setdefault(g["gameId"], set()).add(pid)
    print(f"indexed {n_files} raw files -> {len(game_index)} games", flush=True)

    # 2. official regular-season game list from club schedules
    def sched(args):
        team, season = args
        d = get(f"{WEB}/club-schedule-season/{team}/{season}")
        return [g["id"] for g in d.get("games", []) if g.get("gameType") == 2]

    game_ids = set()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for ids in ex.map(sched, [(t, s) for s in SEASONS for t in TEAMS]):
            game_ids.update(ids)
    print(f"{len(game_ids)} official regular-season games", flush=True)

    extra = sorted(set(game_index) - game_ids)
    if extra:
        print(f"WARNING: {len(extra)} games in raw files but not in official "
              f"schedules (showing 10): {extra[:10]}", flush=True)

    # 3. per-game boxscore reconciliation
    def audit(gid):
        b = get(f"{WEB}/gamecenter/{gid}/boxscore")
        miss, ros = [], []
        season_s = str(season_of_game(gid))
        for side in ("homeTeam", "awayTeam"):
            t = b[side]
            have = game_index.get(gid, set())
            for k in ("forwards", "defense"):
                for p in b["playerByGameStats"][side].get(k, []):
                    pid = p["playerId"]
                    ros.append((season_s, pid, p["name"]["default"]))
                    # A boxscore-listed skater with 0:00 TOI dressed but took
                    # no shifts; the game-log API omits these and they are not
                    # real shot-rate observations. Not required to be in files.
                    if _toi_minutes(p.get("toi")) == 0:
                        continue
                    if pid not in have:
                        miss.append((gid, t["abbrev"], pid))
        return gid, miss, ros

    missing, failed, done = [], 0, 0
    official_rosters = {str(s): {} for s in SEASONS}  # season -> {pid: name}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for gid, m, ros in ex.map(audit, sorted(game_ids)):
            done += 1
            if m is None:
                failed += 1
            else:
                missing.extend(m)
            for season_s, pid, name in ros:
                official_rosters[season_s][str(pid)] = name
            if done % 1000 == 0:
                print(f"  {done}/{len(game_ids)} games, "
                      f"{len(missing)} missing player-games", flush=True)

    # ground-truth roster map (used by ingest to repair list gaps)
    (RAW / "official_rosters.json").write_text(
        json.dumps(official_rosters, indent=1))
    n_ros = sum(len(v) for v in official_rosters.values())
    print(f"wrote official_rosters.json ({n_ros} player-seasons)", flush=True)

    by_season = {}
    for gid, team, pid in missing:
        ssn = str(season_of_game(gid))
        by_season[ssn] = by_season.get(ssn, 0) + 1
    print(f"\nboxscore fetch failures: {failed}", flush=True)
    print(f"missing player-games: {len(missing)}", flush=True)
    for s in sorted(by_season):
        print(f"  {s}: {by_season[s]}", flush=True)

    if failed:
        print("FATAL: boxscore fetches failed; reconciliation inconclusive",
              flush=True)
        sys.exit(1)
    if missing:
        pairs = sorted({(str(season_of_game(g)), p) for g, t, p in missing})
        print(f"FATAL: {len(pairs)} (season, player) pairs missing "
              f"({len(missing)} player-games). Refetch, then re-run this gate.",
              flush=True)
        for s, p in pairs[:20]:
            print(f"  {s} player {p}", flush=True)
        sys.exit(1)
    print("GATE PASS: zero missing player-games across all seasons", flush=True)


if __name__ == "__main__":
    main()
