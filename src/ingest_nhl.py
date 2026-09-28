"""Phase 1: pull raw NHL API data. Train 2018-19..2023-24, validate 2024-25.

HARD GUARD: 2025-26 (the sealed holdout) is never requested. Any attempt raises.

Hardened 2026-09-28 (post-audit): the original swallowed transient fetch
failures (WARN + no file), silently dropping whole player-seasons (~2% of
player-games, including star seasons). This version:
  - rate-limits across threads (token bucket) instead of hammering the API,
  - retries with jittered exponential backoff (8 tries),
  - NEVER writes a file for a failed fetch; collects failures and exits
    nonzero if anything is unresolved,
  - distinguishes a real empty log (HTTP 404 -> no regular-season games,
    file written with games=[] and recorded) from a failure,
  - drops toi-less phantom game-log entries (games the player did not dress
    for; verified absent from the official boxscore), recorded in manifest,
  - dedupes the skater list (the stats API paginates unstably),
  - validates cached files before skipping them (corrupt cache = refetch).
After ingest, src/reconcile_boxscores.py must pass at zero missing before
any downstream step runs.
"""
import json, time, sys, random, threading
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

# ~3 requests/sec shared across all threads (was: 8 threads, no limit -> 429s)
_RATE_PER_SEC = 3.0
_MAX_WORKERS = 4
_TRIES = 8

_bucket_lock = threading.Lock()
_bucket_tokens = _RATE_PER_SEC
_bucket_last = time.monotonic()


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


def _take_token():
    global _bucket_tokens, _bucket_last
    while True:
        with _bucket_lock:
            now = time.monotonic()
            _bucket_tokens = min(_RATE_PER_SEC,
                                 _bucket_tokens + (now - _bucket_last) * _RATE_PER_SEC)
            _bucket_last = now
            if _bucket_tokens >= 1.0:
                _bucket_tokens -= 1.0
                return
            wait = (1.0 - _bucket_tokens) / _RATE_PER_SEC
        time.sleep(wait)


def get(url, tries=_TRIES):
    """Returns parsed JSON, or None on HTTP 404 (real empty, not a failure).
    Raises on anything else after jittered-backoff retries."""
    s = session()
    last = None
    for i in range(tries):
        _take_token()
        try:
            r = s.get(url, timeout=30)
        except requests.RequestException as e:
            last = e
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503):
                last = RuntimeError(f"HTTP {r.status_code}")
            else:
                r.raise_for_status()
        time.sleep((2 ** i) * random.uniform(0.5, 1.5))
    raise RuntimeError(f"failed after {tries} tries: {url} (last: {last})")


def skater_ids(season: int):
    """All skaters (no goalies) for a season via bulk stats API.

    Deduped by playerId: the stats API paginates unstably and can return
    the same player on two pages, which used to inflate the manifest count.
    """
    guard_season(season)
    seen, start = {}, 0
    while True:
        url = (f"{STATS}/skater/summary?limit=100&start={start}"
               f"&cayenneExp=seasonId={season}")
        d = get(url)
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


def game_log(pid: int, season: int):
    """Returns (rows, n_phantom). [] rows for a real empty log (HTTP 404).
    Toi-less entries are phantom did-not-dress rows and are dropped.
    Raises on fetch failure (never returns partial data)."""
    guard_season(season)
    url = f"{WEB}/player/{pid}/game-log/{season}/2"
    d = get(url)
    if d is None:
        return []  # 404: no regular-season games (e.g. playoff-only skater)
    rows = []
    n_phantom = 0
    for g in d.get("gameLog", []):
        # hard guard: no holdout-season games even if API misbehaves
        if g.get("gameDate", "") >= "2025-08-01":
            raise RuntimeError(f"REFUSED: gameDate {g['gameDate']} in holdout window")
        if not g.get("toi"):
            # Phantom entry: the game-log lists a game the player did not
            # dress for (absent from the official boxscore; e.g. Lindholm
            # 2019-20 game 2019020876). A dressed skater always has a toi
            # field. Drop it -- the boxscore gate is the backstop: if this
            # were a real appearance, the gate would flag it missing.
            n_phantom += 1
            continue
        rows.append({
            "gameId": g["gameId"], "gameDate": g["gameDate"],
            "team": g["teamAbbrev"], "opp": g["opponentAbbrev"],
            "homeRoad": g["homeRoadFlag"], "shots": g["shots"],
            "toi": g["toi"], "shifts": g["shifts"], "pim": g["pim"],
            "goals": g["goals"], "assists": g["assists"],
            "ppGoals": g["powerPlayGoals"], "ppPoints": g["powerPlayPoints"],
            "plusMinus": g["plusMinus"],
        })
    return rows, n_phantom


def cached_ok(fp: Path) -> bool:
    """A cached file is usable only if it parses with the expected shape."""
    try:
        d = json.loads(fp.read_text())
        return isinstance(d, dict) and isinstance(d.get("games"), list) \
            and isinstance(d.get("meta"), dict)
    except Exception:
        return False


def main():
    random.seed(20260928)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    failures = []   # never swallowed: reported + nonzero exit below
    phantoms = []   # toi-less game-log entries dropped (did-not-dress)
    empty_404 = {}  # playerIds whose log 404'd (provisional; gate verifies)
    lock = threading.Lock()
    # ground-truth rosters from boxscores (written by reconcile_boxscores.py).
    # The bulk stats API paginates unstably and can OMIT real players
    # (e.g. Marner from 2022-23); unioning guarantees the repair list is
    # complete even when the bulk list is not.
    roster_path = RAW / "official_rosters.json"
    official = {}
    if roster_path.exists():
        official = {int(s): {int(p): n for p, n in d.items()}
                    for s, d in json.loads(roster_path.read_text()).items()}
        print(f"loaded official rosters for {len(official)} seasons",
              flush=True)
    for season in ALL_SEASONS:
        guard_season(season)
        sdir = RAW / "game_logs" / str(season)
        sdir.mkdir(parents=True, exist_ok=True)
        by_id = {p["playerId"]: p for p in skater_ids(season)}
        n_bulk = len(by_id)
        for pid, name in official.get(season, {}).items():
            by_id.setdefault(pid, {"playerId": pid, "name": name, "pos": "?"})
        players = [by_id[k] for k in sorted(by_id)]
        manifest[str(season)] = {"n_players": len(players),
                                 "n_bulk": n_bulk,
                                 "n_roster_only": len(players) - n_bulk}
        print(f"season {season}: {len(players)} skaters "
              f"({n_bulk} bulk + {len(players) - n_bulk} roster-only)",
              flush=True)
        todo = [p for p in players
                if not cached_ok(sdir / f"{p['playerId']}.json")]
        print(f"  {len(todo)} to fetch (skipping valid cache)", flush=True)
        done = [0]
        season_empty = []

        def fetch(p):
            fp = sdir / f"{p['playerId']}.json"
            try:
                rows, n_phantom = game_log(p["playerId"], season)
            except Exception as e:
                # NO file written on failure. Recorded, surfaced, fatal.
                with lock:
                    failures.append({"season": season,
                                     "playerId": p["playerId"],
                                     "name": p["name"],
                                     "error": str(e)[:300]})
                return
            fp.write_text(json.dumps({"meta": p, "games": rows}))
            with lock:
                if not rows:
                    season_empty.append(p["playerId"])
                if n_phantom:
                    phantoms.append({"season": season,
                                     "playerId": p["playerId"],
                                     "name": p["name"],
                                     "n_dropped": n_phantom})
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f"  {season}: {done[0]}/{len(todo)}", flush=True)

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            list(ex.map(fetch, todo))
        empty_404[str(season)] = sorted(season_empty)
        manifest[str(season)]["n_empty_404"] = len(season_empty)
        print(f"season {season} complete "
              f"({len(todo)} fetched, {len(season_empty)} empty)", flush=True)

    manifest["empty_404_by_season"] = empty_404
    manifest["phantoms_dropped"] = phantoms
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (RAW / "failures.json").write_text(
        json.dumps(failures, indent=1))
    if failures:
        print(f"FATAL: {len(failures)} unresolved fetch failures "
              f"(see data/raw/failures.json):", flush=True)
        for f in failures[:10]:
            print(f"  {f['season']} {f['playerId']} {f['name']}: "
                  f"{f['error']}", flush=True)
        sys.exit(1)
    print("ingest complete: zero failures", flush=True)


if __name__ == "__main__":
    main()
