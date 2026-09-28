"""Adjustment round 3 (of 3, final): mu-dependent NB dispersion.
alpha(mu) = alpha_lo if mu<=3 else alpha_hi, tuned per line on 2023-24."""
import json
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.metrics import brier_score_loss
import pickle

LINES = [1.5, 2.5, 3.5, 4.5]
GRID = [0.02, 0.05, 0.1, 0.2, 0.3]
SPLIT = 3.0

def p_over_split(mu, a_lo, a_hi, line):
    mu = np.clip(mu, 0.05, None)
    out = np.empty_like(mu)
    for a, m in ((a_lo, mu <= SPLIT), (a_hi, mu > SPLIT)):
        n, p = 1.0 / a, 1.0 / (1.0 + a * mu[m])
        out[m] = nbinom.sf(int(np.floor(line)), n, p)
    return out

def clean_mu(mu, df):
    s = pd.Series(np.asarray(mu, dtype=float))
    return np.clip(s.fillna(df["prior_pg"].fillna(df["prior_pg"].mean())).values, 0.05, None)

tr = pd.read_parquet("data/processed/train.parquet")
d_fit = tr[tr["season"] <= 20222023].copy()
d_23 = tr[tr["season"] == 20232024].copy()
cfg = json.load(open("models/v1/config.json"))
med = pd.Series(cfg["medians"])
F = cfg["features"]

from sklearn.ensemble import HistGradientBoostingRegressor
Xf = d_fit[F].fillna(med[F]); yf = d_fit["shots"].values
eb_f = clean_mu(d_fit["eb_pred"].values, d_fit)
X23 = d_23[F].fillna(med[F]); y23 = d_23["shots"].values
eb_23 = clean_mu(d_23["eb_pred"].values, d_23)
gbm = HistGradientBoostingRegressor(loss="squared_error", max_iter=300,
                                    learning_rate=0.05, early_stopping=True,
                                    random_state=7).fit(Xf, yf - eb_f)
mu23 = clean_mu(eb_23 + gbm.predict(X23), d_23)

tuned = {}
eb_tuned = {}
for L in LINES:
    Y = (y23 > L).astype(int)
    best = (None, None, 1e9)
    beste = (None, None, 1e9)
    for a_lo in GRID:
        for a_hi in GRID:
            p = np.clip(p_over_split(mu23, a_lo, a_hi, L), 1e-6, 1 - 1e-6)
            b = brier_score_loss(Y, p)
            if b < best[2]:
                best = (a_lo, a_hi, b)
            pe = np.clip(p_over_split(eb_23, a_lo, a_hi, L), 1e-6, 1 - 1e-6)
            be = brier_score_loss(Y, pe)
            if be < beste[2]:
                beste = (a_lo, a_hi, be)
    tuned[str(L)] = {"alpha_lo": best[0], "alpha_hi": best[1]}
    eb_tuned[str(L)] = {"alpha_lo": beste[0], "alpha_hi": beste[1]}
    print(f"line {L}: gbm lo={best[0]} hi={best[1]} brier={best[2]:.4f} | "
          f"eb lo={beste[0]} hi={beste[1]} brier={beste[2]:.4f}")

t = json.load(open("models/v1/alpha_tune.json"))
t["alpha_split"] = tuned
t["alpha_split_eb"] = eb_tuned
json.dump(t, open("models/v1/alpha_tune.json", "w"), indent=1)
print("updated models/v1/alpha_tune.json")
