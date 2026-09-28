# NHL SOG — Challenger Hypothesis Registry

Frozen entries H1–H6 written 2026-09-28, before the 2025-26 holdout was
opened. Purpose: distinguish "we suspected this before seeing new data"
from "we invented this explanation afterward."

New hypotheses may be appended only as explicit, dated additions. Never
rewrite an existing entry to fit observed results.

---

**H1 — Count-distribution coherence**
The per-line NB dispersion approach may calibrate individual thresholds
reasonably while lacking one coherent player-game shot distribution.
Future challenger: a model producing one internally consistent SOG count
distribution (single NB / hurdle / zero-aware head shared across lines).

**H2 — TOI modeling**
A projected-TOI × shots-per-60 architecture may outperform the direct
residual SOG model when player usage changes materially.
Future challenger: true two-stage TOI / shot-rate model.

**H3 — Power-play deployment**
`pp_share_l10` (trailing PP-points share) may not capture real PP
opportunity well enough.
Future challenger: stronger pregame PP-role/deployment representation.

**H4 — Role-change / usage-change games**
The baseline may be weaker when recent TOI materially differs from a
player's established usage (see `toi_trend` and short-vs-long form
buckets in `DIAGNOSTIC_BUCKETS.md`).
Future challenger: better role-change sensitivity.

**H5 — High-confidence calibration**
The baseline may remain overconfident in some high-probability Over
buckets (2024-25 validation: over-1.5 [0.85,0.90) missed by 4.3pp).
Future challenger: separately preregistered calibration treatment or
alternate distributional model.

**H6 — Market efficiency varies by player type / line**
The model's independent hockey signal may have different value versus the
market for low vs high lines, regular vs elite shooters, forwards vs
defensemen, and stable-role vs changing-role players.
To be tested from logged market data (`MARKET_LOGGING.md`), not assumed.

---

*Challenger protocol (standing): a future challenger originates from (1) a
pre-existing hypothesis here or a clearly documented new observation,
(2) a frozen proposed design, (3) chronological unseen evaluation, and
(4) comparison against both v1.2 and EB. The baseline itself is not
repeatedly modified.*

---

## Dated additions (append-only)

**2026-09-28 — H7 (from 2025-26 holdout evidence):**
Calibration-slope / over-shrinkage. On the holdout the model's P(over)
was systematically *under*-confident through the mid-high range
(over-1.5 [0.55,0.80): −3.9 to −7.5pp, n=708–3,480; over-2.5 [0.50,0.55):
−8.3pp, n=681) while *over*-confident at the low end (over-1.5 [0,0.50):
+4.8pp, n=32,258) — probabilities too compressed toward the middle.
League scoring did not shift (holdout mean shots 1.546 vs validation
1.571), so this is not a scoring-environment change. The high-end
overconfidence seen on validation (over-1.5 [0.85,0.90) +4.3pp) did NOT
persist on holdout (n=62, −0.7pp). Future challenger: preregistered
recalibration (e.g. Platt/isotonic fit on validation only) or reduced
shrinkage strength, evaluated on unseen data — not fit to the holdout.

**2026-09-28 — evidence for H4 (from 2025-26 holdout):**
Form buckets support the role-change weakness hypothesis: pooled Brier
was worse for `surging` (0.1434) and `slumping` (0.1342) than `neutral`
(0.1145) player-games. Noted as evidence; H4 text above unchanged.
