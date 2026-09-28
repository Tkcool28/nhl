"""Round 2 experiment: fix top-end mu bias. Candidates:
A) blend: mu = w*mu_gbm + (1-w)*mu_eb
B) residual-target GBM: fit on (shots - eb_pred), mu = eb_pred + resid_pred
Scored on 2023-24: max bucket violation (n>=200) + pooled brier."""
import json
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss
import pickle

LINES = [1.5, 2.5, 3.5, 4.5]
ALPHAS = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]

def p_over(mu, alpha, line):
    n, p = 1.0 / alpha, 1.0 / (1.0 + alpha * np.clip(mu, 0.05, None))
    return nbinom.sf(int(np.floor(line)), n, p)

def clean_mu(mu, df):
    s = pd.Series(np.asarray(mu, dtype=float))
    return np.clip(s.fillna(df["prior_pg"].fillna(df["prior_pg"].mean())).values, 0.05, None)

def score(df, mu, tag):
    y = df["shots"].values
    maxviol, nb = 0, 0
    P_all, Y_all = [], []
    for L in LINES:
        Y = (y > L).astype(int)
        ba, bb = None, 1e9
        for a in ALPHAS:
            p = np.clip(p_over(mu, a, L), 1e-6, 1 - 1e-6)
            b = brier_score_loss(Y, p)
            if b < bb:
                bb, ba = b, a
        p = np.clip(p_over(mu, ba, L), 1e-6, 1 - 1e-6)
        P_all.append(p); Y_all.append(Y)
        for lo in [0, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]:
            hi = lo + .05 if lo >= .5 else .5
            m = (p >= lo) & (p < hi + 1e-9) if lo == 0 else (p >= lo) & (p < hi)
            if m.sum() >= 200:
                nb += 1
                maxviol = max(maxviol, abs(p[m].mean() - Y[m].mean()))
    P_all = np.concatenate(P_all); Y_all = np.concatenate(Y_all)
    return {"tag": tag, "pooled_brier": round(float(brier_score_loss(Y_all, P_all)), 4),
            "max_bucket_viol": round(float(maxviol), 4), "n_buckets": nb}

tr = pd.read_parquet("data/processed/train.parquet")
d_fit = tr[tr["season"] <= 20222023].copy()
d23 = tr[tr["season"] == 20232024].copy()
cfg = json.load(open("models/v1/config.json"))
med = pd.Series(cfg["medians"])
F = cfg["features"]
Xf, yf = d_fit[F].fillna(med[F]), d_fit["shots"].values
X23 = d23[F].fillna(med[F])
eb_fit = clean_mu(d_fit["eb_pred"].values, d_fit)
eb_23 = clean_mu(d23["eb_pred"].values, d23)

gbm = pickle.load(open("models/v1/gbm.pkl", "rb"))  # refit below on d_fit for fairness
gbm_f = HistGradientBoostingRegressor(loss="poisson", max_iter=300,
                                      learning_rate=0.05, early_stopping=True,
                                      random_state=7).fit(Xf, yf)
mu_g = clean_mu(gbm_f.predict(X23), d23)

out = [score(d23, mu_g, "A blend w=1.0")]
for w in (0.9, 0.8, 0.7):
    out.append(score(d23, w * mu_g + (1 - w) * eb_23, f"A blend w={w}"))
# candidate B: residual target
res_f = HistGradientBoostingRegressor(loss="squared_error", max_iter=300,
                                      learning_rate=0.05, early_stopping=True,
                                      random_state=7).fit(Xf, yf - eb_fit)
mu_r = clean_mu(eb_23 + res_f.predict(X23), d23)
out.append(score(d23, mu_r, "B residual-target"))
for r in out:
    print(r)
json.dump(out, open("models/v1/round2.json", "w"), indent=1)
