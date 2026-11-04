# Pre-registration (OSF)

Not posted. This is the Phase 2 draft, built from the plan's checklist in Step 2.5. Every blank is filled only from `results/` or a committed decision.

- OSF URL: TODO
- OSF timestamp: TODO
- RHUH-GBM download date: TODO (must be later than the OSF timestamp)

## Checklist (plan Step 2.5) and where each item comes from

| Item | Source in this repository | State |
|---|---|---|
| Tournament leaderboard and the named challenger, with the commit hash | `results/tournament/leaderboard.json` + its `run.manifest.json` (`scripts/02_tournament.py`) | not run |
| SHA-256 of every refit model | `results/release/model_hashes.json` (`scripts/03_refit_release.py`) | not run |
| Feature definitions and the label mapping for each cohort | `DECISIONS.md` D6 (LUMIERE); RHUH-GBM mapping from its dataset paper (plan Step 2.2) | LUMIERE drafted |
| Cavity rule and label-verification rule | plan Step 2.2, copied here once worded | TODO |
| Registration recipe, screen thresholds, exclusion rules | plan Step 3.3; threshold fixed against the tuned recipe's LUMIERE round-trip distribution | TODO |
| Power table | `results/power/power_table.json` (`scripts/04_power.py`) | not run |
| Primary, verdict rule, secondaries, sensitivity and exploratory lists | plan Step 2.3, copied here verbatim | TODO |
| Result sentences | below (plan Step 2.4) | drafted |
| What was seen before posting | all LUMIERE results, both predecessor studies, the RHUH-GBM dataset paper, no RHUH-GBM image | TODO at posting |

## Result sentences (plan Step 2.4, verbatim, blanks unfilled)

- *Challenger wins:* "On an independent cohort, [challenger] adds [Δ] PR-AUC (95% CI [...]) over distance from the treatment target, and at the volume of the 15 mm guideline margin captures [x]% of recurrence against [y]% for distance."
- *Distance holds:* "On an independent cohort, the best challenger from an eight-model tournament adds less than [upper limit] PR-AUC over distance, and at the volume of the 15 mm guideline margin captures [x]% of recurrence against [y]% for distance; distance remains the best available predictor."
- *The two metrics disagree:* "The verdict follows the pre-registered primary, PR-AUC, which [does / does not] favour [challenger]. At the guideline margin's volume alone, coverage [does / does not] differ ([x]% vs [y]%, 95% CI [...]); the difference is reported as a secondary result."

"Eight-model" must match the number of candidates actually run: seven if C5 is dropped at Step 0.4.

## Title bands (plan Step 6.1, written in week 2)

- *Challenger champion:* "Where does glioblastoma come back? [Champion] finds more of it than distance, on patients it has never seen."
- *Distance holds:* "Where does glioblastoma come back? Eight models, one sealed test: distance still wins."
