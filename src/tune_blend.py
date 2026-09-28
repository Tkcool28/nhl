"""Adjustment round 2 (of <=3): shrink extreme GBM mu toward the EB prior.
mu_final = w*mu_gbm + (1-w)*mu_eb. Tune w + per-line alpha on 2023-24 only."""
import json
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.metrics import brier_score_loss
import pickle

LINES = [1.5, 2.5, 3.5, 4.5]

def p_over(mu, alpha, line):
    n, p = 1.0 / alpha, 1.0 / (1.0 + alpha * np.clip(mu, 0.05, None))
    return nbinom.sf(int(np.floor(line)), n, p)

def clean_mu(mu, df):
    s = pd.Series(np.asarray(mu, dtype=float))
    return np.clip(s.fillna(df["prior_pg"].fillna(df["prior_pg"].mean())).values, 0.05, None)

tr = pd.read_parquet("data/processed/train.parquet")
d23 = tr[tr["season"] == 20232024].copy()
gbm = pickle.load(open("models/v1/gbm.pkl", "rb"))
cfg = json.load(open("models/v1/config.json"))
med = pd.Series(cfg["medians"])
X23 = d23[cfg["features"]].fillna(med[cfg["features"]])
mu_g = clean_mu(gbm.predict(X23), d23)
mu_e = clean_mu(d23["eb_pred"].values, d23)
y23 = d23["shots"].values

ALPHAS = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
results = {}
for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    mu = w * mu_g + (1 - w) * mu_e
    per_line, briers = {}, []
    for L in LINES:
        Y = (y23 > L).astype(int)
        bl = (None, 1e9)
        for a in ALPHAS:
            p = np.clip(p_over(mu, a, L), 1e-6, 1 - 1e-6)
            b = brier_score_loss(Y, p)
            if b < bl[1]:
                bl = (a, b)
        per_line[str(L)] = bl[0]
        briers.append(bl[1])
    # pooled brier with chosen per-line alphas
    P = np.concatenate([np.clip(p_over(mu, per_line[str(L)], L), 1e-6, 1 - 1e-6) for L in LINES])
    Y = np.concatenate([(y23 > L).astype(int) for L in LINES])
    pb = brier_score_loss(Y, P)
    results[w] = (pb, per_line)
    print(f"w={w}: pooled brier={pb:.4f} alphas={per_line}")

best_w = min(results, key=lambda w: results[w][0])
print("best w:", best_w)
tune = json.load(open("models/v1/alpha_tune.json"))
tune["gbm_nb"] = results[best_w][1]
tune["blend_w"] = best_w
json.dump(tune, open("models/v1/alpha_tune.json", "w"), indent=1)
print("updated models/v1/alpha_tune.json")
