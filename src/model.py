"""Phase 3: train on <=2023-24, validate on 2024-25. 2025-26 never loaded.

Main model (v1): residual-target GBM. Learns deviations from the shrunk EB
prior: mu = eb_pred + GBM_resid(features). This forces shrinkage structure and
fixes the top-end mu bias diagnosed in round 1 (GBM-poisson overpredicted
stars: mean mu 3.33 vs actual 3.15 in the top bucket).
Count distribution: negative binomial head -> P(over/under) per line.
Adjustment rounds used: 3 of <=3 (round1: per-line alpha; round2: residual
target; round3: mu-split alpha). 2026-09-27: bug-fix rerun (prior_pg fallback
leak) logged in FREEZE_PROTOCOL.md -- not a tuning round.
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
ALPHAS = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
# frozen training-only fallback (written by features.py; never recomputed here)
TRAIN_MEAN = float(json.load(open(PROC / "fallbacks.json"))["train_mean_shots"])

FEATURES = ["shots_l5", "shots_l10", "shots_l20", "shots_per60_l10",
            "toi_l10", "toi_trend", "pp_share_l10",
            "shots_home_l10", "shots_road_l10", "n_trailing",
            "opp_shots_against_l10", "opp_pim_l10",
            "team_shots_for_l10", "team_pace_l10",
            "rest_days", "b2b", "home", "month", "prior_pg"]
TOI_FEATURES = ["toi_l10", "toi_trend", "rest_days", "b2b", "home", "month",
                "opp_pim_l10", "team_pace_l10", "n_trailing"]
ALL_COLS = list(dict.fromkeys(FEATURES + TOI_FEATURES))

def stage_a_medians(tr):
    """Imputation medians for Stage A (fit <=2022-23, select on 2023-24).

    2026-09-28 correctness fix: computed from the Stage A fit population
    ONLY. The old code used medians over the full train frame, so the
    2023-24 selection frame contributed its own imputation statistics."""
    return tr[tr["season"] <= 20222023][ALL_COLS].median()

def stage_b_medians(tr):
    """Imputation medians for Stage B (final fit <=2023-24, validate 2024-25):
    the full allowed training population."""
    return tr[ALL_COLS].median()

def load():
    tr = pd.read_parquet(PROC / "train.parquet")
    va = pd.read_parquet(PROC / "valid.parquet")
    assert (tr["season"] != SEALED).all() and (va["season"] != SEALED).all()
    assert (va["season"] == 20242025).all()
    return tr, va

def clean_mu(mu, df):
    # safety-net fill uses the FROZEN training mean only -- never a mean
    # computed over the frame being scored (bugfix 2026-09-27).
    # 2026-09-28: s is indexed like df so the prior_pg fill aligns by row.
    # (The old RangeIndex silently misaligned whenever df was a slice,
    # leaving NaNs -- masked before because leakage kept NaN eb_pred rare
    # and clustered at the frame start where the indices coincided.)
    s = pd.Series(np.asarray(mu, dtype=float), index=df.index)
    return np.clip(s.fillna(df["prior_pg"].fillna(TRAIN_MEAN)).values, 0.05, None)

ALPHA_SPLIT_MU = 3.0  # mu threshold for dispersion split (round 3)

def p_over(mu, alpha, line):
    mu = np.clip(mu, 0.05, None)
    if isinstance(alpha, dict) and "alpha_lo" in alpha:
        out = np.empty_like(mu)
        for a, m in ((alpha["alpha_lo"], mu <= ALPHA_SPLIT_MU),
                     (alpha["alpha_hi"], mu > ALPHA_SPLIT_MU)):
            n, p = 1.0 / a, 1.0 / (1.0 + a * mu[m])
            out[m] = nbinom.sf(int(np.floor(line)), n, p)
        return out
    n, p = 1.0 / alpha, 1.0 / (1.0 + alpha * mu)
    return nbinom.sf(int(np.floor(line)), n, p)

def alpha_for_mu(mu, alpha):
    """Per-row alpha: split dict -> alpha_lo/hi at ALPHA_SPLIT_MU, else scalar."""
    mu = np.asarray(mu, dtype=float)
    if isinstance(alpha, dict) and "alpha_lo" in alpha:
        return np.where(mu <= ALPHA_SPLIT_MU, alpha["alpha_lo"], alpha["alpha_hi"])
    return np.full_like(mu, float(alpha))

def nb_count_loglik(y, mu, alpha):
    """Sum of NB log-pmf at observed counts (pre-registered gate 3)."""
    mu = np.clip(np.asarray(mu, dtype=float), 0.05, None)
    a = alpha_for_mu(mu, alpha)
    n, p = 1.0 / a, 1.0 / (1.0 + a * mu)
    return float(np.sum(nbinom.logpmf(np.asarray(y, dtype=int),
                                      np.maximum(n, 1e-9),
                                      np.clip(p, 1e-9, 1 - 1e-9))))

def tune_alpha_per_line(df, mu):
    y = df["shots"].values
    out = {}
    for L in LINES:
        Y = (y > L).astype(int)
        best = (None, 1e9)
        for a in ALPHAS:
            b = brier_score_loss(Y, np.clip(p_over(mu, a, L), 1e-6, 1 - 1e-6))
            if b < best[1]:
                best = (a, b)
        out[str(L)] = best[0]
    return out

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
    res = {"tag": tag, "lines": {}}
    res["alpha"] = alpha if isinstance(alpha, dict) else float(alpha)
    P, Y = [], []
    for L in LINES:
        a = alpha[str(L)] if isinstance(alpha, dict) else alpha
        p = p_over(mu, a, L)
        y = (df_valid["shots"].values > L).astype(int)
        m = res["lines"][str(L)] = line_metrics(y, p)
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

def fit_residual_gbm(Xf, yf, eb_f):
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=300, learning_rate=0.05,
        early_stopping=True, random_state=7).fit(Xf, yf - eb_f)

def main():
    ART.mkdir(parents=True, exist_ok=True)
    tr, va = load()
    med_a = stage_a_medians(tr)
    med_b = stage_b_medians(tr)

    def frame(df, med):
        X = df[FEATURES].fillna(med[FEATURES])
        eb = clean_mu(df["eb_pred"].values, df)
        return X, df["shots"].values, eb

    # ---- stage A: tune per-line alpha on 2023-24 (train window) ----
    # (split-alpha values come from src/tune_alpha_split.py, run beforehand)
    d_fit = tr[tr["season"] <= 20222023]
    d_23 = tr[tr["season"] == 20232024]
    Xf, yf, eb_f = frame(d_fit, med_a)
    X23, y23, eb_23 = frame(d_23, med_a)
    gbm_a = fit_residual_gbm(Xf, yf, eb_f)
    mu_23 = clean_mu(eb_23 + gbm_a.predict(X23), d_23)
    tune = json.load(open(ART / "alpha_tune.json"))
    alpha_gbm = tune["alpha_split"]
    alpha_eb = tune["alpha_split_eb"]
    P = np.concatenate([np.clip(p_over(mu_23, alpha_gbm[str(L)], L), 1e-6, 1 - 1e-6) for L in LINES])
    Pe = np.concatenate([np.clip(p_over(eb_23, alpha_eb[str(L)], L), 1e-6, 1 - 1e-6) for L in LINES])
    Y = np.concatenate([((d_23["shots"].values > L).astype(int)) for L in LINES])
    print(f"[selection 23-24] resid-GBM brier={brier_score_loss(Y, P):.4f} "
          f"EB brier={brier_score_loss(Y, Pe):.4f}", flush=True)

    # ---- stage B: final fit on all train, validate on 24-25 ----
    Xtr, ytr, eb_tr = frame(tr, med_b)
    Xva, yva, eb_va = frame(va, med_b)
    gbm = fit_residual_gbm(Xtr, ytr, eb_tr)
    mu_main = clean_mu(eb_va + gbm.predict(Xva), va)

    # two-stage ridge reference (unchanged)
    Ttr = tr[TOI_FEATURES].fillna(med_b[TOI_FEATURES])
    Tva = va[TOI_FEATURES].fillna(med_b[TOI_FEATURES])
    ridge_toi = Ridge(alpha=1.0).fit(Ttr, tr["toi"].fillna(med_b["toi_l10"]).values)
    pred_toi = np.clip(ridge_toi.predict(Tva), 1, None)
    s60 = Xva["shots_per60_l10"].fillna(tr["shots"].sum() / tr["toi"].sum() * 60)
    mu_ridge2 = clean_mu(pred_toi / 60 * s60.values, va)

    models = {
        "naive_l10": clean_mu(va["naive_l10"].values, va),
        "eb": eb_va,
        "ridge2stage": mu_ridge2,
        "gbm_nb": mu_main,
    }
    alphas = {"gbm_nb": alpha_gbm, "eb": alpha_eb}
    ytr_s = tr["shots"].values
    for tag in ("naive_l10", "ridge2stage"):
        mu_tr = clean_mu(tr["naive_l10"].values if tag == "naive_l10"
                         else np.clip(Ridge(alpha=1.0).fit(Ttr, tr["toi"].fillna(med_b["toi_l10"]).values)
                                      .predict(Ttr).clip(1) / 60 *
                                      Xtr["shots_per60_l10"].fillna(
                                          tr["shots"].sum() / tr["toi"].sum() * 60).values, 0.05, None), tr)
        num = np.mean(np.maximum((ytr_s - mu_tr) ** 2 - mu_tr, 0))
        alphas[tag] = max(num / max(np.mean(mu_tr ** 2), 1e-9), 1e-6)

    report = {"n_train": len(tr), "n_valid": len(va),
              "valid_mean_shots": float(va["shots"].mean()),
              "valid_players": int(va["player_id"].nunique()),
              "adjustment_rounds_used": 3,
              "adjustment_log": [
                  "round1: per-line NB alpha tuned on 2023-24 (moment-matched alpha was overdispersed)",
                  "round2: residual-target GBM (mu = eb_pred + GBM(shots - eb_pred)) to fix top-end mu bias",
                  "round3: mu-dependent NB dispersion, alpha_lo (mu<=3) / alpha_hi (mu>3) per line, tuned on 2023-24"]}
    results = {}
    for tag, mu in models.items():
        results[tag] = evaluate(va, mu, alphas[tag], tag)
        r = results[tag]
        print(f"[{tag}] pooled brier={r['pooled']['brier']:.4f} "
              f"logloss={r['pooled']['logloss']:.4f} ece={r['pooled']['ece10']:.4f} "
              f"auc={r['pooled']['auc']:.4f}", flush=True)

    g = results["gbm_nb"]["pooled"]; b = results["eb"]["pooled"]
    gates = {"brier_beats_eb": g["brier"] < b["brier"],
             "logloss_beats_eb": g["logloss"] < b["logloss"]}
    cal_ok = True
    for L, lm in results["gbm_nb"]["lines"].items():
        for bk in lm["buckets"]:
            if bk["n"] >= 200 and abs(bk["pred"] - bk["hit"]) > 0.03:
                cal_ok = False
    gates["calibration_3pp"] = cal_ok
    # pre-registered gate 3: NB count log-likelihood beats baseline.
    # v1 note: alphas are per-line, so LL is summed over the four line-heads
    # (each row scored once per head); both models get identical treatment.
    yva_s = va["shots"].values
    ll_gbm = sum(nb_count_loglik(yva_s, mu_main, alpha_gbm[str(L)]) for L in LINES)
    ll_eb = sum(nb_count_loglik(yva_s, eb_va, alpha_eb[str(L)]) for L in LINES)
    report["nb_count_loglik"] = {"gbm_nb": ll_gbm, "eb": ll_eb}
    gates["nb_ll_beats_eb"] = ll_gbm > ll_eb
    print(f"NB count loglik: gbm_nb={ll_gbm:.1f} eb={ll_eb:.1f}", flush=True)
    report["gates"] = {k: bool(v) for k, v in gates.items()}
    report["results"] = results
    print("GATES:", json.dumps(report["gates"]), flush=True)

    # row-level validation predictions (audit trail; hash at freeze)
    pred_rows = pd.DataFrame({
        "player_id": va["player_id"].values, "game_id": va["game_id"].values,
        "game_date": va["game_date"].dt.strftime("%Y-%m-%d"),
        "season": va["season"].values, "team": va["team"].values,
        "opp": va["opp"].values, "home": va["home"].values, "pos": va["pos"].values,
        "shots": yva_s, "n_trailing": va["n_trailing"].values,
        "mu_gbm": mu_main, "mu_eb": np.asarray(eb_va, dtype=float)})
    for L in LINES:
        pred_rows[f"p_over_{L}"] = p_over(mu_main, alpha_gbm[str(L)], L)
        pred_rows[f"p_over_eb_{L}"] = p_over(np.asarray(eb_va, dtype=float),
                                            alpha_eb[str(L)], L)
    pred_rows.to_parquet(ART / "valid_predictions.parquet", index=False)
    print(f"row-level predictions: {len(pred_rows):,} rows -> valid_predictions.parquet",
          flush=True)

    (ART / "metrics.json").write_text(json.dumps(report, indent=1))
    with open(ART / "gbm.pkl", "wb") as f:
        pickle.dump(gbm, f)
    with open(ART / "ridge_toi.pkl", "wb") as f:
        pickle.dump(ridge_toi, f)
    (ART / "config.json").write_text(json.dumps({
        "features": FEATURES, "toi_features": TOI_FEATURES,
        "medians": med_b.to_dict(), "lines": LINES,
        "alpha": {t: (alphas[t] if isinstance(alphas[t], dict) else float(alphas[t]))
                  for t in alphas},
        "model": "residual-target GBM: mu = eb_pred + GBM(shots - eb_pred)",
        "version": "v1", "frozen": True}, indent=1))
    print("model v1 written to", ART, flush=True)

if __name__ == "__main__":
    main()
