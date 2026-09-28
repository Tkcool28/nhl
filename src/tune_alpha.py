"""Adjustment round 1 (of <=3): tune the NB dispersion mapping on 2023-24 only.
Final validation on 2024-25 stays untouched until the end."""
import json
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import pickle

LINES = [1.5, 2.5, 3.5, 4.5]

def p_over(mu, alpha, line):
    n, p = 1.0 / alpha, 1.0 / (1.0 + alpha * np.clip(mu, 0.05, None))
    return nbinom.sf(int(np.floor(line)), n, p)

def pooled_brier(df, mu, alpha):
    P = np.concatenate([p_over(mu, alpha, L) for L in LINES])
    Y = np.concatenate([(df["shots"].values > L).astype(int) for L in LINES])
    return brier_score_loss(Y, np.clip(P, 1e-6, 1 - 1e-6))

tr = pd.read_parquet("data/processed/train.parquet")
d23 = tr[tr["season"] == 20232024].copy()
gbm = pickle.load(open("models/v1/gbm.pkl", "rb"))
cfg = json.load(open("models/v1/config.json"))
med = pd.Series(cfg["medians"])
X23 = d23[cfg["features"]].fillna(med[cfg["features"]])

def clean_mu(mu, df):
    s = pd.Series(np.asarray(mu, dtype=float))
    return np.clip(s.fillna(df["prior_pg"].fillna(df["prior_pg"].mean())).values, 0.05, None)

mus = {"gbm_nb": clean_mu(gbm.predict(X23), d23),
       "eb": clean_mu(d23["eb_pred"].values, d23)}
y23 = d23["shots"].values
out = {}
for tag, mu in mus.items():
    per_line = {}
    for L in LINES:
        Y = (y23 > L).astype(int)
        bl = (None, 1e9)
        for a in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
            p = np.clip(p_over(mu, a, L), 1e-6, 1 - 1e-6)
            b = brier_score_loss(Y, p)
            if b < bl[1]:
                bl = (a, b)
        per_line[str(L)] = bl[0]
    out[tag] = per_line
    print(tag, per_line)

json.dump(out, open("models/v1/alpha_tune.json", "w"), indent=1)
print("written models/v1/alpha_tune.json")
