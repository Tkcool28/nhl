# Audit response — 2026-09-27 (MDT), second Claude audit

Second independent audit received and worked through item by item. One item
turned into a real finding: the ingest completeness check (which had never
been run) surfaced a genuine data defect. Everything else is answered below.

No model code was changed. No holdout data was touched. The `v1-frozen` tag
and all frozen artifacts are untouched.

## 1. NEW FINDING: ingest is missing ~2% of player-games (verified)

The completeness check against official boxscore team/individual SOG had never
been run. I ran it tonight as a read-only audit: every regular-season game
2018-19 through 2024-25 (~8,500 games), official boxscore skaters vs our
per-game rosters.

**Result: 5,916 player-games missing (1.98% of 298,889), across 128
(season, player) pairs.** 113 pairs are entire missing player-season files;
15 are files present but missing 1–4 specific games.

Missing full seasons include genuine stars, not fringe call-ups:

| Season | Missing (games) |
|---|---|
| 2024-25 (validation) | Pastrnak 82, Teravainen 82, Schmaltz 82, Luostarinen 80, Aston-Reese 79, Sherwood 78, Cates 78, Zamula 63, Greenway 34, Bellows 19 (+4 small) = **687** |
| 2023-24 (tuning) | Forsberg 82, Evangelista 80, McCann 80, Veleno 80, Clifton 79, Power 76, Greenway 67, Scandella 65 (+3 small) = **674** |
| 2022-23 | Suter 82, Zadorov 82, Hague 81, Marner 80, Skinner 79, Raddysh 78, Soucy 78, Guentzel 78, Peterka 77, Girard 76, Chiarot 76, Gudbranson 70, Eichel 67, Marino 64, DeBrusk 64, Makar 60, Cirelli 58, Gauthier 57, Forbort 54, OEL 54, Foote 50 (+18 more) = **1,786** |
| 2021-22 | 27 pairs = **1,418** |
| 2020-21 | Marner 55, Skinner 53, DeBrincat 52, Tatar 50, DeBrusk 41, Bjornfot 33, Brannstrom 30, Pettersson 26, Bunting 21, Richardson 17, Pederson 15, Gerbe 9, Chmelevski 5, York 3, Addison 3 (+6 more) = **638** |
| 2018-19 | 13 pairs = **656** (Dubois 82, McNabb 81, Paquette 80…) |
| 2019-20 | 3 pairs = **57** |

**Root cause** (`src/ingest_nhl.py::fetch`): on any exception (e.g. transient
429/500 under 8-thread load, 4 retries exhausted) it prints `WARN` and returns
**without writing the file**. The player-season is then silently absent. There
is no failure manifest and nothing re-verifies. (The `todo` skip-cached logic
means a re-run resumes exactly the missing files — the repair path exists.)

Two adjacent quirks found while diagnosing (not defects in the sweep):
- The bulk stats API's historical season skater lists are unstable over time
  (2022-23 list returned 997 at ingest, 932 today; Marner dropped out). The
  boxscore sweep is the ground truth, not the list.
- Its `gamesPlayed` includes playoffs, so per-season file counts can't be
  reconciled from it naively.

**Impact on frozen v1.** Both the model and the EB baseline were scored on
the same incomplete rows, so the *comparison* (the gates) is like-for-like.
But: the validation population is missing 687 player-games including three
full 82-game star seasons; team features (`shots_for_l10` etc.) are biased
down ~0.6 shots/game wherever skaters are missing (consistent train→valid,
so learned around, but a defect regardless); and we would knowingly ship a
model trained on data missing Pastrnak's entire validation season.

**Recommendation: data-only refreeze before the holdout opens** (decision
needed — see §5). This is the "better now than after the holdout" case.

## 2. Flag-by-flag answers

**Dates (protocol header 2026-09-27 vs baseline "frozen 2026-09-28").**
Labeling inconsistency, not a protocol violation — UTC vs MDT. The earlier
session dated things in UTC; the protocol was written in MDT. Evidence the
commitment predates the fix, both orderings:
- Local mtimes: `FREEZE_PROTOCOL.md` 21:45:26 < `src/features.py` edit 21:45:34 (MDT).
- Git commits (UTC): protocol `9fcf9af1` 03:47:14Z < fix `820b7cd9` 03:47:18Z.
The protocol (and the metrics copy) were written before any fix code was
touched, and committed before the fix commit. Standardizing on MDT going
forward.

**blend_w: 1.0.** `tune_blend.py` searched a GBM/EB blend weight
w ∈ {0.5,…,1.0} on 2023-24 (poisson-GBM era) and landed on w=1.0 — no blend,
the spec'd model. Full disclosure: its docstring calls itself "round 2" while
the adjustment log's round 2 is the residual GBM, so the log undercounts —
**four** tuning experiments happened, not three (α grid, blend search,
residual target, μ-split α). The blend experiment selected no-change, so it
manufactured nothing, but the bookkeeping was sloppy and this corrects it.
`blend_w` is a dead key: nothing in `model.py` or `config.json` reads it, and
the per-line entries it refreshed are fossils superseded by the split dicts.
Left untouched in the frozen artifacts; v2 cleans it up.

**Alpha grid floor (0.02).** Confirmed: 6 of the 16 live split entries sit on
the floor (gbm 4.5-lo; eb 1.5-hi, 2.5-lo/hi, 3.5-lo/hi), plus 5 fossil entries.
The data want near-Poisson dispersion. Not a bug; v2 fits α continuously by NB
likelihood (already on the v2 list).

**Which α did the NB likelihood gate use?** Summed over the four line-heads:
`sum(nb_count_loglik(y, mu, alpha[line]) for line in LINES)` — each validation
row scored 4×, both models identically (`src/model.py:229-230`). The absolute
value (−288,224.5) is not interpretable as a single model's fit; the
*comparison* is fair because treatment is identical. v2's single coherent
head makes this clean.

**"Not fully strictly prior" (2018-19 fallback).** Conceded on language.
For 2018-19 the fallback is the training-wide mean, which includes later
seasons — so "strictly prior" was overstrong phrasing. On substance: the
fallback is a single constant per season applied to training rows only; no
row-specific future information can pass through a constant, and it cannot
reach validation. "Leak gone" holds where it matters.

**Hashes / off-machine copies / tag.** Tag `v1-frozen` exists (points at
`2df3fd65`). All artifacts are in the GitHub repo itself — `gbm.pkl`
(532 KB), `valid_predictions.parquet` (3.3 MB), `SHA256SUMS`, JSONs — so
off-machine copies exist today, not just hashes. A GitHub release asset for
the binaries would be nicer hardening; optional.

**Brier argument correction.** Conceded fully — and it's the same
conclusion-first pattern as before, so noted as a standing correction, not
just an instance. Comparing absolute Brier across seasons (0.1213 vs 0.1276)
was invalid; base rates differ. The fair measure is edge over EB:
selection 0.1294−0.1276 = **0.0018** → validation 0.1228−0.1212 = **0.0016**.
Small shrink, consistent with mild selection optimism from the tuning rounds,
nothing alarming. Conclusion stands; the reasoning didn't.

## 3. "Ask Muse" answers

**Poison test / merge row-count assertions:** the row-count assertions ran
and passed (3 merge asserts in `features.py`, exit 0; leakage test passed).
The poison test was **not** built — deferred to v2. Offer stands to build it
now as read-only verification (inject a bogus future value, confirm no
earlier-game feature moves).

**Ingest completeness check:** had never been run — ran tonight, found the
§1 defect. No refreeze has been performed; awaiting the §5 decision.

## 4. What was NOT changed

- No model code touched. No metrics recomputed. `v1-frozen` tag untouched.
- This file is new; `FREEZE_PROTOCOL.md` and all frozen artifacts are
  byte-identical.

## 5. Decision required: data-only refreeze

**Proposed:** (1) harden `ingest_nhl.py` (failure manifest, retry-until-clean —
mechanical, not modeling); (2) re-fetch the 128 missing pairs via the
existing resume logic; (3) re-execute the deterministic pipeline **as coded**
(features → tune scripts → train → validate), no knob changes, no new
experiments; (4) record current frozen numbers as the pre-refreeze baseline;
(5) freeze the output under a new tag (`v1-frozen` stays as the historical
record; suggest `v1.1-frozen`).

**Why:** the defect is mechanical, the repair is deterministic, the holdout
is still sealed — this is exactly the window for it. Shipping v1 (and
building the frontend against it) while knowing Pastrnak's validation season
is absent would be knowingly shipping a damaged validation.

**Why not:** it reopens a freeze, which costs something in discipline. The
validation *verdict* (beats EB / fails calibration) is unlikely to flip —
missingness is ~random w.r.t. skill (rate-limit failures don't select on
talent) and both models were scored on identical rows.

My view: do the refreeze. The precedent that matters is "fix real defects
before the holdout opens," not "never touch anything." But it's your call —
say the word and I run it, or we hold v1 as-is and log the defect.

## Appendix: full missing-pair list (128 pairs, 5,916 player-games)

status = file_missing (113) → whole player-season file absent;
file_exists_games_missing (15) → file present, specific games absent.

| season | player_id | name | n_games | teams | status |
|---|---|---|---|---|---|
| 20182019 | 8479400 | Pierre-Luc Dubois | 82 | CBJ | file_missing |
| 20182019 | 8475188 | Brayden McNabb | 81 | VGK | file_missing |
| 20182019 | 8476975 | Cedric Paquette | 80 | TBL | file_missing |
| 20182019 | 8476927 | Teddy Blueger | 28 | PIT | file_missing |
| 20182019 | 8474736 | Zac Rinaldo | 23 | NSH | file_missing |
| 20182019 | 8474774 | Dalton Prout | 20 | CGY | file_missing |
| 20182019 | 8479365 | Trent Frederic | 15 | BOS | file_missing |
| 20182019 | 8478873 | Troy Terry | 1 | ANA | file_exists_games_missing |
| (+5 more small 2018-19 pairs) |||||
| 20192020 | 8476854 | Hampus Lindholm | 55 | ANA | file_missing |
| 20192020 | 8478870 | Rudolfs Balcers | 1 | OTT | file_exists_games_missing |
| 20192020 | 8474000 | Steven Kampfer | 1 | BOS | file_exists_games_missing |
| 20202021 | 8478483 | Mitch Marner | 55 | TOR | file_missing |
| 20202021 | 8475784 | Jeff Skinner | 53 | BUF | file_missing |
| 20202021 | 8479337 | Alex DeBrincat | 52 | CHI | file_missing |
| 20202021 | 8475193 | Tomas Tatar | 50 | MTL | file_missing |
| 20202021 | 8478498 | Jake DeBrusk | 41 | BOS | file_missing |
| 20202021 | 8481600 | Tobias Bjornfot | 33 | LAK | file_missing |
| 20202021 | 8480073 | Erik Brannstrom | 30 | OTT | file_missing |
| 20202021 | 8480012 | Elias Pettersson | 26 | VAN | file_missing |
| 20202021 | 8478047 | Michael Bunting | 21 | ARI | file_missing |
| 20202021 | 8470755 | Brad Richardson | 17 | NSH | file_missing |
| 20202021 | 8478967 | Lane Pederson | 15 | ARI | file_missing |
| 20202021 | 8471804 | Nathan Gerbe | 9 | CBJ | file_missing |
| 20202021 | 8480053 | Sasha Chmelevski | 5 | SJS | file_missing |
| 20202021 | 8481546 | Cam York | 3 | PHI | file_missing |
| 20202021 | 8480884 | Calen Addison | 3 | MIN | file_missing |
| 20202021 | 8474685 | Matt Calvert | 1 | COL | file_exists_games_missing |
| 20202021 | 8477845 | Trevor van Riemsdyk | 1 | WSH | file_exists_games_missing |
| (+5 more small 2020-21 pairs) |||||
| 20212022 | 8479977 | Kailer Yamamoto | 81 | EDM | file_missing |
| 20212022 | 8477938 | Haydn Fleury | 36 | SEA | file_missing |
| 20212022 | 8466138 | Joe Thornton | 34 | FLA | file_missing |
| 20212022 | 8478099 | Kevin Labanc | 21 | SJS | file_missing |
| 20212022 | 8481059 | Scott Perunovich | 19 | STL | file_missing |
| 20212022 | 8478512 | Austin Czarnik | 17 | NYI,SEA | file_missing |
| 20212022 | 8478491 | Jacob Larsson | 6 | ANA | file_missing |
| 20212022 | 8478055 | Gustav Forsling | 1 | FLA | file_exists_games_missing |
| 20212022 | 8479639 | Dylan Coghlan | 1 | VGK | file_exists_games_missing |
| 20212022 | 8480823 | Alexander Alexeyev | 1 | WSH | file_missing |
| (+17 more 2021-22 pairs) |||||
| 20222023 | 8470600 | Ryan Suter | 82 | DAL | file_missing |
| 20222023 | 8477507 | Nikita Zadorov | 82 | CGY | file_missing |
| 20222023 | 8479980 | Nicolas Hague | 81 | VGK | file_missing |
| 20222023 | 8478483 | Mitch Marner | 80 | TOR | file_missing |
| 20222023 | 8475784 | Jeff Skinner | 79 | BUF | file_missing |
| 20222023 | 8479390 | Taylor Raddysh | 78 | CHI | file_missing |
| 20222023 | 8477369 | Carson Soucy | 78 | SEA | file_missing |
| 20222023 | 8477404 | Jake Guentzel | 78 | PIT | file_missing |
| 20222023 | 8482175 | JJ Peterka | 77 | BUF | file_missing |
| 20222023 | 8479398 | Samuel Girard | 76 | COL | file_missing |
| 20222023 | 8475279 | Ben Chiarot | 76 | DET | file_missing |
| 20222023 | 8475790 | Erik Gudbranson | 70 | CBJ | file_missing |
| 20222023 | 8478403 | Jack Eichel | 67 | VGK | file_missing |
| 20222023 | 8478507 | John Marino | 64 | NJD | file_missing |
| 20222023 | 8478498 | Jake DeBrusk | 64 | BOS | file_missing |
| 20222023 | 8480069 | Cale Makar | 60 | COL | file_missing |
| 20222023 | 8478519 | Anthony Cirelli | 58 | TBL | file_missing |
| 20222023 | 8479328 | Julien Gauthier | 57 | NYR,OTT | file_missing |
| 20222023 | 8475762 | Derek Forbort | 54 | BOS | file_missing |
| 20222023 | 8475171 | Oliver Ekman-Larsson | 54 | VAN | file_missing |
| 20222023 | 8479984 | Cal Foote | 50 | NSH,TBL | file_missing |
| 20222023 | 8482671 | Owen Power | 1 | BUF | file_exists_games_missing |
| (+16 more 2022-23 pairs) |||||
| 20232024 | 8476887 | Filip Forsberg | 82 | NSH | file_missing |
| 20232024 | 8482146 | Luke Evangelista | 80 | NSH | file_missing |
| 20232024 | 8477955 | Jared McCann | 80 | SEA | file_missing |
| 20232024 | 8480813 | Joseph Veleno | 80 | DET | file_missing |
| 20232024 | 8477365 | Connor Clifton | 79 | BUF | file_missing |
| 20232024 | 8482671 | Owen Power | 76 | BUF | file_missing |
| 20232024 | 8478413 | Jordan Greenway | 67 | BUF | file_missing |
| 20232024 | 8474618 | Marco Scandella | 65 | STL | file_missing |
| 20232024 | 8481019 | David Gustafsson | 39 | WPG | file_missing |
| 20232024 | 8480007 | Jonas Rondbjerg | 20 | VGK | file_missing |
| 20232024 | 8479522 | Hardy Haman Aktell | 6 | WSH | file_missing |
| 20242025 | 8477956 | David Pastrnak | 82 | BOS | file_missing |
| 20242025 | 8476882 | Teuvo Teravainen | 82 | CHI | file_missing |
| 20242025 | 8477951 | Nick Schmaltz | 82 | UTA | file_missing |
| 20242025 | 8480185 | Eetu Luostarinen | 80 | FLA | file_missing |
| 20242025 | 8479944 | Zachary Aston-Reese | 79 | CBJ | file_missing |
| 20242025 | 8480748 | Kiefer Sherwood | 78 | VAN | file_missing |
| 20242025 | 8480220 | Noah Cates | 78 | PHI | file_missing |
| 20242025 | 8481178 | Egor Zamula | 63 | PHI | file_missing |
| 20242025 | 8478413 | Jordan Greenway | 34 | BUF | file_missing |
| 20242025 | 8479356 | Kieffer Bellows | 19 | NSH | file_missing |
| 20242025 | 8482641 | Matt Kiersted | 2 | FLA | file_missing |
| 20242025 | 8481848 | Jacob Gaucher | 4 | PHI | file_exists_games_missing |
| 20242025 | 8478400 | Colin White | 3 | SJS | file_exists_games_missing |
| 20242025 | 8481546 | Cam York | 1 | PHI | file_exists_games_missing |

Full machine-readable list: 128 pairs (audit working files, not committed).
Method: per-game boxscore skater reconciliation, ~8,500 regular-season games,
2018-19–2024-25. Note n_games counts regular-season games only; some GP
figures from the stats API run higher because they include playoffs.
