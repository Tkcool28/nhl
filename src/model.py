"""Phase 3: train on <=2023-24, validate on 2024-25. 2025-26 never loaded.

Models: naive L10, empirical-Bayes (baselines), two-stage ridge (TOI -> shots),
GBM-poisson for mu + negative-binomial head for P(over/under).
Pre-registered gates evaluated on 2024-25 only.
"""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
ART = Path(__file__).resolve().parent.parent / "models" / "v1"
SEALED = 20252026
LINES = [1.5, 2.5, 3.5, 4.5]

FEATURES = ["shots_l5", "shots_l10", "shots_l20", "shots_per60_l10",
            "toi_l10", "toi_trend", "pp_share_l10",
            "shots_home_l10", "shots_road_l10", "n_trailing",
            "opp_shots_against_l10", "opp_pim_l10",
            "team_shots_for_l10", "team_pace_l10",
            "rest_days", "b2b", "home", "month", "prior_pg"]
TOI_FEATURES = ["toi_l10", "toi_trend", "rest_days", "b2b", "home", "month",
                "opp_pim_l10", "team_pace_l10", "n_trailing"]

def load():
    tr = pd.read_parquet(PROC / "train.parquet")
    va = pd.read_parquet(PROC / "valid.parquet")
    assert (tr["season"] != SEALED).all() and (va["season"] != SEALED).all()
    assert (va["season"] == 20242025).all()
    return tr, va

def prep(tr, va):
    med = tr[FEATURES + TOI_FEATURES].median()
    Xtr = tr[FEATURES].fillna(med[FEATURES])
    Xva = va[FEATURES].fillna(med[FEATURES])
    Ttr = tr[TOI_FEATURES].fillna(med[TOI_FEATURES])
    Tva = va[TOI_FEATURES].fillna(med[TOI_FEATURES])
    return (Xtr, va["shots"].values, va["toi"].fillna(med["toi_l10"]).values,
            Xva, tr["shots"].values, Ttr, Tva, med)

def estimate_alpha(y, mu):
    mu = np.clip(mu, 0.05, None)
    num = np.mean(np.maximum((y - mu) ** 2 - mu, 0))
    den = np.mean(mu ** 2)
    return max(num / max(den, 1e-9), 1e-6)

def p_over(mu, alpha, line):
    n, p = 1.0 / alpha, 1.0 / (1.0 + alpha * np.clip(mu, 0.05, None))
    return nbinom.sf(int(np.floor(line)), n, p)

def line_metrics(y, p):
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "base_rate": float(y.mean())}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(y) >= 50 else None
    out["brier"] = float(brier_score_loss(y, p))
    out["logloss"] = float(log_loss(y, p))
    return out

def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum():
            e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(e)

def evaluate(df_valid, mu, alpha, tag):
    """Pooled + per-line metrics for P(over) at standard lines."""
    res = {"tag": tag, "alpha": float(alpha), "lines": {}}
    P, Y = [], []
    for L in LINES:
        p = p_over(mu, alpha, L)
        y = (df_valid["shots"].values > L).astype(int)
        m = res["lines"][str(L)] = line_metrics(y, p)
        # preset calibration buckets (5pp bins) - apples-to-apples record
        buckets = []
        edges = [0, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (p >= lo) & (p < hi)
            if mm.sum():
                buckets.append({"lo": lo, "hi": round(hi, 2), "n": int(mm.sum()),
                                "pred": round(float(p[mm].mean()), 4),
                                "hit": round(float(y[mm].mean()), 4)})
        m["buckets"] = buckets
        P.append(p); Y.append(y)
    P = np.concatenate(P); Y = np.concatenate(Y)
    res["pooled"] = line_metrics(Y, P)
    res["pooled"]["ece10"] = ece(Y, P)
    return res

def main():
    ART.mkdir(parents=True, exist_ok=True)
    tr, va = load()
    Xtr, yva, toi_va, Xva, ytr, Ttr, Tva, med = prep(tr, va)

    # ---- stage A: model selection sanity on 2023-24 (within train) ----
    sel_tr = tr[tr["season"] <= 20222023]
    sel_va = tr[tr["season"] == 20232024]
    med_s = sel_tr[FEATURES + TOI_FEATURES].median()
    Xs_tr = sel_tr[FEATURES].fillna(med_s[FEATURES]); ys_tr = sel_tr["shots"].values
    Xs_va = sel_va[FEATURES].fillna(med_s[FEATURES]); ys_va = sel_va["shots"].values
    gbm_s = HistGradientBoostingRegressor(loss="poisson", max_iter=300,
                                          learning_rate=0.05, early_stopping=True,
                                          random_state=7).fit(Xs_tr, ys_tr)
    mu_s = np.clip(gbm_s.predict(Xs_va), 0.05, None)
    a_s = estimate_alpha(ys_tr, np.clip(gbm_s.predict(Xs_tr), 0.05, None))
    eb_s = sel_va["eb_pred"].values
    a_eb = estimate_alpha(ys_tr, np.clip(sel_tr["eb_pred"].values, 0.05, None))
    p_gbm = np.concatenate([p_over(mu_s, a_s, L) for L in LINES])
    p_eb = np.concatenate([p_over(np.clip(eb_s, 0.05, None), a_eb, L) for L in LINES])
    y_bin = np.concatenate([((sel_va["shots"].values > L).astype(int)) for L in LINES])
    print(f"[selection 23-24] GBM brier={brier_score_loss(y_bin, p_gbm):.4f} "
          f"EB brier={brier_score_loss(y_bin, p_eb):.4f}", flush=True)

    # ---- stage B: final fit on all train, validate on 24-25 ----
    ridge_toi = Ridge(alpha=1.0).fit(Ttr, tr["toi"].fillna(med["toi_l10"]).values)
    pred_toi = np.clip(ridge_toi.predict(Tva), 1, None)
    s60 = Xva["shots_per60_l10"].fillna(tr["shots"].sum() / tr["toi"].sum() * 60)
    mu_ridge2 = pred_toi / 60 * s60.values

    gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=400,
                                        learning_rate=0.05, early_stopping=True,
                                        random_state=7).fit(Xtr, ytr)
    mu_gbm = np.clip(gbm.predict(Xva), 0.05, None)
    mu_tr_gbm = np.clip(gbm.predict(Xtr), 0.05, None)

    models = {
        "naive_l10": np.clip(va["naive_l10"].fillna(va["prior_pg"]).values, 0.05, None),
        "eb": np.clip(va["eb_pred"].values, 0.05, None),
        "ridge2stage": np.clip(mu_ridge2, 0.05, None),
        "gbm_nb": mu_gbm,
    }
    # alpha per model from train residuals
    train_mu = {"naive_l10": np.clip(tr["naive_l10"].fillna(tr["prior_pg"]).values, 0.05, None),
                "eb": np.clip(tr["eb_pred"].values, 0.05, None),
                "ridge2stage": np.clip(
                    Ridge(alpha=1.0).fit(Ttr, tr["toi"].fillna(med["toi_l10"]).values)
                    .predict(Ttr).clip(1) / 60 *
                    Xtr["shots_per60_l10"].fillna(tr["shots"].sum() / tr["toi"].sum() * 60).values,
                    0.05, None),
                "gbm_nb": mu_tr_gbm}
    report = {"n_train": len(tr), "n_valid": len(va),
              "valid_mean_shots": float(va["shots"].mean()),
              "valid_players": int(va["player_id"].nunique())}
    results = {}
    for tag, mu in models.items():
        alpha = estimate_alpha(ytr, train_mu[tag])
        results[tag] = evaluate(va, mu, alpha, tag)
        r = results[tag]
        print(f"[{tag}] pooled brier={r['pooled']['brier']:.4f} "
              f"logloss={r['pooled']['logloss']:.4f} ece={r['pooled']['ece10']:.4f} "
              f"auc={r['pooled']['auc']:.4f}", flush=True)

    # ---- pre-registered gates ----
    g = results["gbm_nb"]["pooled"]; b = results["eb"]["pooled"]
    gates = {
        "brier_beats_eb": g["brier"] < b["brier"],
        "logloss_beats_eb": g["logloss"] < b["logloss"],
    }
    cal_ok = True
    for L, lm in results["gbm_nb"]["lines"].items():
        for bk in lm["buckets"]:
            if bk["n"] >= 200 and abs(bk["pred"] - bk["hit"]) > 0.03:
                cal_ok = False
    gates["calibration_3pp"] = cal_ok
    report["gates"] = {k: bool(v) for k, v in gates.items()}
    report["results"] = results
    print("GATES:", json.dumps(report["gates"]), flush=True)

    (ART / "metrics.json").write_text(json.dumps(report, indent=1))
    with open(ART / "gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)
    with open(ART / "ridge_toi.pkl", "wb") as f:
        pickle.dump(ridge_toi, f)
    (ART / "config.json").write_text(json.dumps({
        "features": FEATURES, "toi_features": TOI_FEATURES,
        "medians": med.to_dict(), "lines": LINES,
        "alpha": {t: results[t]["alpha"] for t in results},
        "version": "v1", "frozen": True}, indent=1))
    print("model v1 written to", ART, flush=True)

if __name__ == "__main__":
    main()
