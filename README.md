# NHL SOG Projection Model

Predicts per-player shots on goal for each game as a full count distribution
(negative binomial) → P(over/under) per standard line → fair American odds →
EV / edge-vs-book / reliability in the dashboard.

- `PLAN.md` — the full agreed plan (data split, gates, logging protocol).
- `src/ingest_nhl.py` — Phase 1: raw pull from the official NHL API.
- `src/features.py` — Phase 2: prior-games-only features, shrinkage priors.
- `src/model.py` — Phase 3: train ≤2023-24, validate 2024-25. **2025-26 is the
  sealed holdout and is never touched by any script (hard guards).**
- `tests/test_leakage.py` — leakage + holdout guards.

## Reproduce

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/ingest_nhl.py   # ~15-20 min, cached/resumable
.venv/bin/python src/features.py
.venv/bin/python src/model.py
.venv/bin/python tests/test_leakage.py
```

Raw data is gitignored (too large); `data/raw/manifest.json` describes it.
Model artifacts land in `models/v1/` (also gitignored except config).
