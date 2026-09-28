"""Holdout opening milestone -- authorized by Terry 2026-09-28.

A separate, deliberate run: pulls 2025-26, reconciles it against official
boxscores, builds features with the FROZEN features.build(), scores with
the FROZEN model artifacts (via src/predict.py), and evaluates strictly
per EVALUATION_CONTRACT.md.

Frozen scripts (ingest_nhl, reconcile_boxscores, features, model) are NOT
modified. This script reuses their unguarded primitives and mirrors their
guarded logic for the holdout season only. Nothing here retrains, retunes,
or edits the baseline, the contract, the buckets, or the hypotheses.

Phases: pull | reconcile | build | score | evaluate | all
"""
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest_nhl as ing
import reconcile_boxscores as rec
import features as feat
from model import evaluate, ece, nb_count_loglik, LINES  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
HOLDOUT_SEASON = 20252026
SEASON_DIR = ing.RAW / "game_logs" / str(HOLDOUT_SEASON)
HOLDOUT_PARQUET = REPO / "data" / "processed" / "holdout_2025_26.parquet"
HOLDOUT_DIR = REPO / "holdout"
PREDICTIONS_JSONL = HOLDOUT_DIR / "predictions_2025_26.jsonl"
REPORT_JSON = HOLDOUT_DIR / "report_2025_26.json"
ELIGIBLE_TOI = 13.0  # frozen (DIAGNOSTIC_BUCKETS.md)

# gameDate window for the 2025-26 regular season (safety net; the API is
# queried with gameType=2 so only regular-season games are returned)
SEASON_START = "2025-08-01"
SEASON_END = "2026-08-01"


# ---------------------------------------------------------------- pull ---
def holdout_bulk_skaters(season):
    """Mirror of ingest_nhl.skater_ids without the holdout guard."""
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


def holdout_game_log(pid, season):
    """Mirror of ingest_nhl.game_log with the date window flipped to the
    holdout season. Raises on fetch failure (never returns partial data)."""
    url = f"{ing.WEB}/player/{pid}/game-log/{season}/2"
    d = ing.get(url)
    if d is None:
        return [], 0
    rows, n_phantom = [], 0
    for g in d.get("gameLog", []):
        gd = g.get("gameDate", "")
        if not (SEASON_START <= gd < SEASON_END):
            raise RuntimeError(f"gameDate {gd} outside holdout window")
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


def phase_pull():
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    roster_path = ing.RAW / "official_rosters.json"
    official = {}
    if roster_path.exists():
        official = json.loads(roster_path.read_text()).get(str(HOLDOUT_SEASON), {})
        print(f"loaded {len(official)} official-roster players for repair union",
              flush=True)
    by_id = {p["playerId"]: p for p in holdout_bulk_skaters(HOLDOUT_SEASON)}
    n_bulk = len(by_id)
    for pid, name in official.items():
        by_id.setdefault(int(pid), {"playerId": int(pid), "name": name, "pos": "?"})
    players = [by_id[k] for k in sorted(by_id)]
    print(f"season {HOLDOUT_SEASON}: {len(players)} skaters "
          f"({n_bulk} bulk + {len(players) - n_bulk} roster-only)", flush=True)

    todo = [p for p in players
            if not ing.cached_ok(SEASON_DIR / f"{p['playerId']}.json")]
    print(f"{len(todo)} to fetch (skipping valid cache)", flush=True)
    failures, phantoms, empty = [], [], []
    lock = threading.Lock()
    done = [0]

    def fetch(p):
        fp = SEASON_DIR / f"{p['playerId']}.json"
        try:
            rows, n_phantom = holdout_game_log(p["playerId"], HOLDOUT_SEASON)
        except Exception as e:
            with lock:
                failures.append({"playerId": p["playerId"], "name": p["name"],
                                 "error": str(e)[:300]})
            return
        fp.write_text(json.dumps({"meta": p, "games": rows}))
        with lock:
            if not rows:
                empty.append(p["playerId"])
            if n_phantom:
                phantoms.append({"playerId": p["playerId"], "n_dropped": n_phantom})
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  {done[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=ing._MAX_WORKERS) as ex:
        list(ex.map(fetch, todo))
    manifest = {"n_players": len(players), "n_bulk": n_bulk,
                "n_fetched": len(todo), "n_empty_404": len(empty),
                "phantoms_dropped": phantoms}
    (ing.RAW / "holdout_manifest.json").write_text(json.dumps(manifest, indent=1))
    if failures:
        (ing.RAW / "holdout_failures.json").write_text(
            json.dumps(failures, indent=1))
        print(f"FATAL: {len(failures)} fetch failures", flush=True)
        sys.exit(1)
    print(f"pull complete: {len(players)} players, zero failures, "
          f"{len(empty)} empty, {len(phantoms)} with phantoms", flush=True)


# ----------------------------------------------------------- reconcile ---
def phase_reconcile():
    game_index, n_files = {}, 0
    for fp in SEASON_DIR.glob("*.json"):
        n_files += 1
        d = json.loads(fp.read_text())
        pid = int(fp.stem)
        for g in d.get("games", []):
            game_index.setdefault(g["gameId"], set()).add(pid)
    print(f"indexed {n_files} raw files -> {len(game_index)} games", flush=True)

    def sched(args):
        team, season = args
        d = rec.get(f"{rec.WEB}/club-schedule-season/{team}/{season}")
        return [g["id"] for g in d.get("games", []) if g.get("gameType") == 2]

    game_ids = set()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for ids in ex.map(sched, [(t, HOLDOUT_SEASON) for t in rec.TEAMS]):
            game_ids.update(ids)
    print(f"{len(game_ids)} official regular-season games", flush=True)
    extra = sorted(set(game_index) - game_ids)
    if extra:
        print(f"WARNING: {len(extra)} games in raw files but not official",
              flush=True)

    def audit(gid):
        b = rec.get(f"{rec.WEB}/gamecenter/{gid}/boxscore")
        miss, ros = [], []
        for side in ("homeTeam", "awayTeam"):
            t = b[side]
            have = game_index.get(gid, set())
            for k in ("forwards", "defense"):
                for p in b["playerByGameStats"][side].get(k, []):
                    pid = p["playerId"]
                    ros.append((pid, p["name"]["default"]))
                    if rec._toi_minutes(p.get("toi")) == 0:
                        continue
                    if pid not in have:
                        miss.append((gid, t["abbrev"], pid))
        return miss, ros

    missing, failed, ros_all, done = [], 0, [], 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for m, ros in ex.map(audit, sorted(game_ids)):
            done += 1
            if m is None:
                failed += 1
            else:
                missing.extend(m)
                ros_all.extend(ros)
            if done % 500 == 0:
                print(f"  {done}/{len(game_ids)} games, "
                      f"{len(missing)} missing", flush=True)
    print(f"boxscore fetch failures: {failed}", flush=True)
    print(f"missing player-games: {len(missing)}", flush=True)

    # merge holdout rosters into the ground-truth roster map (additive)
    rp = ing.RAW / "official_rosters.json"
    rosters = json.loads(rp.read_text()) if rp.exists() else {}
    ros_season = rosters.setdefault(str(HOLDOUT_SEASON), {})
    for pid, name in ros_all:
        ros_season[str(pid)] = name
    rp.write_text(json.dumps(rosters, indent=1))

    if failed:
        print("FATAL: boxscore fetches failed; reconciliation inconclusive",
              flush=True)
        sys.exit(1)
    if missing:
        pairs = sorted({p for _, _, p in missing})
        print(f"FATAL: {len(pairs)} players missing ({len(missing)} "
              f"player-games). Re-run pull (repair union), then reconcile.",
              flush=True)
        sys.exit(1)
    print("GATE PASS: zero missing player-games in 2025-26", flush=True)


# --------------------------------------------------------------- build ---
def load_all_seasons():
    """Mirror of features.load() over all 8 season dirs (holdout included).
    The frozen features.build() does the rest."""
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


def phase_build():
    fb_path = feat.OUT / "fallbacks.json"
    fb_before = fb_path.read_bytes()  # build() rewrites it; must be unchanged
    df = feat.build(load_all_seasons())
    assert (df["season"] <= HOLDOUT_SEASON).all()
    assert (df["game_date"] < "2026-08-01").all()
    if fb_path.read_bytes() != fb_before:
        fb_path.write_bytes(fb_before)
        raise RuntimeError("FATAL: build() changed frozen fallbacks.json")
    ho = df[df["season"] == HOLDOUT_SEASON].copy().reset_index(drop=True)
    assert len(ho) > 0
    HOLDOUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    ho.to_parquet(HOLDOUT_PARQUET, index=False)
    print(f"holdout frame: {len(ho):,} rows, "
          f"{ho['player_id'].nunique()} players, "
          f"{ho['game_id'].nunique()} games", flush=True)
    print(f"seasons in build: {sorted(df['season'].unique())}", flush=True)
    print(f"fallbacks.json unchanged: True", flush=True)


# --------------------------------------------------------------- score ---
def phase_score():
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_JSONL.exists():
        sys.exit(f"refusing to overwrite {PREDICTIONS_JSONL}")
    r = subprocess.run(
        [sys.executable, str(REPO / "src" / "predict.py"),
         "--input", str(HOLDOUT_PARQUET), "--out", str(PREDICTIONS_JSONL),
         "--run-id", "holdout-2025-26"],
        capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)
    print(r.stdout.strip(), flush=True)


# ------------------------------------------------------------ evaluate ---
def _pbucket(p):
    edges = [0, .40, .50, .60, .70, .80, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= p < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "[0.80,1.01)"


def phase_evaluate():
    cfg = json.load(open(REPO / "models" / "v1" / "config.json"))
    alpha_gbm = cfg["alpha"]["gbm_nb"]
    alpha_eb = cfg["alpha"]["eb"]
    ho = pd.read_parquet(HOLDOUT_PARQUET)
    preds = pd.read_json(PREDICTIONS_JSONL, lines=True)
    ev = ho.merge(preds[["game_id", "player_id", "mu_gbm", "mu_eb",
                         "p_over_1_5", "p_over_2_5", "p_over_3_5",
                         "p_over_4_5", "eligible", "run_id"]],
                  on=["game_id", "player_id"], how="inner")
    assert len(ev) == len(ho), "prediction/frame row mismatch"
    assert (ev["run_id"] == "holdout-2025-26").all()

    def bucket_rows(df):
        return {
            "mu_bucket": pd.cut(df["mu_gbm"],
                               [-np.inf, 1.0, 1.75, 2.5, np.inf],
                               labels=["low", "medium", "high", "elite"]),
            "toi_bucket": pd.cut(df["toi_l10"],
                                 [-np.inf, 14, 17, 20, np.inf],
                                 labels=["low_usage", "normal", "high",
                                         "very_high"]),
            "toi_trend": pd.cut(df["toi_trend"],
                                [-np.inf, -0.5, 0.5, np.inf],
                                labels=["declining", "stable", "rising"]),
            "shot_vol": pd.cut(df["shots_l10"],
                               [-np.inf, 1.0, 1.75, 2.5, np.inf],
                               labels=["cold", "cool", "warm", "hot"]),
            "form": pd.cut(df["shots_l5"] - df["shots_l10"],
                           [-np.inf, -0.4, 0.4, np.inf],
                           labels=["slumping", "neutral", "surging"]),
            "history": pd.cut(df["n_trailing"],
                              [-np.inf, 20, 150, np.inf],
                              labels=["low_sample", "developing",
                                      "established"]),
        }

    report = {"holdout_season": str(HOLDOUT_SEASON), "run_id": "holdout-2025-26",
              "n_rows": len(ev), "n_players": int(ev["player_id"].nunique()),
              "n_games": int(ev["game_id"].nunique()),
              "slices": {}, "gates": {}, "diagnostics": {},
              "calibration_miss_check": {}}
    mu = ev["mu_gbm"].values
    eb = ev["mu_eb"].values
    y = ev["shots"].values

    for sname, sdf, smu, seb in [
            ("all_skater", ev, mu, eb),
            ("production_eligible", ev[ev["eligible"]], None, None)]:
        if sname == "production_eligible":
            smu = sdf["mu_gbm"].values
            seb = sdf["mu_eb"].values
        sy = sdf["shots"].values
        r_gbm = evaluate(sdf, smu, alpha_gbm, "gbm_nb")
        r_eb = evaluate(sdf, seb, alpha_eb, "eb")
        mae = float(np.mean(np.abs(sy - smu)))
        rmse = float(np.sqrt(np.mean((sy - smu) ** 2)))
        mae_eb = float(np.mean(np.abs(sy - seb)))
        report["slices"][sname] = {
            "n": len(sdf),
            "gbm_nb": {"pooled": r_gbm["pooled"], "lines": r_gbm["lines"],
                       "mae_mu": mae, "rmse_mu": rmse},
            "eb": {"pooled": r_eb["pooled"], "lines": r_eb["lines"],
                   "mae_mu": mae_eb},
        }
        print(f"[{sname}] n={len(sdf):,} "
              f"brier gbm={r_gbm['pooled']['brier']:.4f} "
              f"eb={r_eb['pooled']['brier']:.4f} "
              f"logloss gbm={r_gbm['pooled']['logloss']:.4f} "
              f"ece={r_gbm['pooled']['ece10']:.4f} "
              f"auc={r_gbm['pooled']['auc']:.4f} "
              f"mae_mu={mae:.3f}", flush=True)

    # pre-registered gates on the production-eligible slice (contract §gates
    # are defined on the holdout; report both slices' gate status)
    for sname in ("all_skater", "production_eligible"):
        s = report["slices"][sname]
        g, b = s["gbm_nb"]["pooled"], s["eb"]["pooled"]
        cal_ok = True
        for lm in s["gbm_nb"]["lines"].values():
            for bk in lm["buckets"]:
                if bk["n"] >= 200 and abs(bk["pred"] - bk["hit"]) > 0.03:
                    cal_ok = False
        sdf = ev if sname == "all_skater" else ev[ev["eligible"]]
        sy = sdf["shots"].values
        smu = sdf["mu_gbm"].values
        seb = sdf["mu_eb"].values
        ll_gbm = sum(nb_count_loglik(sy, smu, alpha_gbm[str(L)]) for L in LINES)
        ll_eb = sum(nb_count_loglik(sy, seb, alpha_eb[str(L)]) for L in LINES)
        report["gates"][sname] = {
            "brier_beats_eb": bool(g["brier"] < b["brier"]),
            "logloss_beats_eb": bool(g["logloss"] < b["logloss"]),
            "calibration_3pp": bool(cal_ok),
            "nb_ll_beats_eb": bool(ll_gbm > ll_eb),
            "nb_count_loglik": {"gbm_nb": ll_gbm, "eb": ll_eb},
        }
    print("GATES:", json.dumps(
        {k: {gk: gv for gk, gv in v.items() if gk != "nb_count_loglik"}
         for k, v in report["gates"].items()}), flush=True)

    # calibration-miss persistence: the three v1.2 validation miss buckets
    miss_defs = [("1.5", 0.85, 0.90), ("2.5", 0.50, 0.55), ("2.5", 0.65, 0.70)]
    for L, lo, hi in miss_defs:
        key = f"over-{L}[{lo},{hi})"
        report["calibration_miss_check"][key] = {}
        for sname in ("all_skater", "production_eligible"):
            for bk in report["slices"][sname]["gbm_nb"]["lines"][L]["buckets"]:
                if abs(bk["lo"] - lo) < 1e-9 and abs(bk["hi"] - hi) < 1e-9:
                    report["calibration_miss_check"][key][sname] = bk
    # high-confidence diagnostic buckets per line (overconfidence check)
    hi_conf = {}
    for L in ["1.5", "2.5", "3.5", "4.5"]:
        col = f"p_over_{L.replace('.', '_')}"
        yy = (ev["shots"].values > float(L)).astype(int)
        pp = ev[col].values
        row = {}
        for lo, hi in [(0.70, 0.80), (0.80, 1.01)]:
            m = (pp >= lo) & (pp < hi)
            if m.sum():
                row[f"[{lo:.2f},{hi:.2f})"] = {
                    "n": int(m.sum()), "pred": round(float(pp[m].mean()), 4),
                    "hit": round(float(yy[m].mean()), 4),
                    "miss_pp": round(float((pp[m].mean() - yy[m].mean()) * 100), 2)}
        hi_conf[f"over_{L}"] = row
    report["diagnostics"]["high_confidence_p_over"] = hi_conf

    # per-bucket pooled Brier for the remaining diagnostic dims
    P = np.concatenate([ev[f"p_over_{L.replace('.', '_')}"].values
                        for L in ["1.5", "2.5", "3.5", "4.5"]])
    Y = np.concatenate([(ev["shots"].values > float(L)).astype(int)
                        for L in ["1.5", "2.5", "3.5", "4.5"]])
    bdf = pd.concat([ev] * 4, ignore_index=True)
    bmap = bucket_rows(ev)
    diag = {}
    dims = {"mu_bucket": bmap["mu_bucket"], "toi_bucket": bmap["toi_bucket"],
            "toi_trend": bmap["toi_trend"], "shot_vol": bmap["shot_vol"],
            "form": bmap["form"], "history": bmap["history"]}
    for dname, cats in dims.items():
        rep = pd.concat([cats] * 4, ignore_index=True)
        tab = {}
        for cat in rep.cat.categories:
            m = (rep == cat).values
            if m.sum() >= 200:
                from sklearn.metrics import brier_score_loss, log_loss
                tab[str(cat)] = {
                    "n": int(m.sum()),
                    "brier": round(float(brier_score_loss(Y[m], P[m])), 4),
                    "logloss": round(float(log_loss(Y[m], P[m])), 4)}
        diag[dname] = tab
    pos_tab = {}
    for pos, lab in [("D", "D"), ("C", "F-C"), ("L", "F-L"), ("R", "F-R")]:
        m = np.concatenate([(ev["pos"].values == pos)] * 4)
        if m.sum() >= 200:
            from sklearn.metrics import brier_score_loss
            pos_tab[lab] = {"n": int(m.sum()),
                            "brier": round(float(brier_score_loss(Y[m], P[m])), 4)}
    diag["position"] = pos_tab
    for h, lab in [(1, "home"), (0, "road")]:
        m = np.concatenate([(ev["home"].values == h)] * 4)
        from sklearn.metrics import brier_score_loss
        diag[lab] = {"n": int(m.sum()),
                     "brier": round(float(brier_score_loss(Y[m], P[m])), 4)}
    report["diagnostics"]["buckets"] = diag

    REPORT_JSON.write_text(json.dumps(report, indent=1))
    print(f"report -> {REPORT_JSON}", flush=True)


PHASES = {"pull": phase_pull, "reconcile": phase_reconcile,
          "build": phase_build, "score": phase_score,
          "evaluate": phase_evaluate}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["pull", "reconcile", "build", "score", "evaluate"]
    if which == "all":
        for p in order:
            print(f"===== phase: {p} =====", flush=True)
            PHASES[p]()
    elif which in PHASES:
        PHASES[which]()
    else:
        sys.exit(f"unknown phase: {which} (pull|reconcile|build|score|evaluate|all)")


if __name__ == "__main__":
    main()
