# PROBAST+AI self-audit (plan Step 6.2)

Filled in after the confirmatory run, as the report's appendix. TODO: verify the domain wording and signalling questions against the published PROBAST+AI tool before rating. Each domain is rated for risk of bias and for applicability. Every judgement cites the repository file it rests on.

| Domain | Development (LUMIERE) | Evaluation (RHUH-GBM) | Evidence in this repository |
|---|---|---|---|
| 1. Participants and data sources | TODO | TODO | gbm-4d cohort lock; `results/rhuh_lock/consort.md`; deviation V1 |
| 2. Predictors | TODO | TODO | D6 item 2; D7; features computed from the baseline only |
| 3. Outcome | TODO | TODO | D6 item 1; label transport and screens; automated vs corrected labels |
| 4. Analysis | TODO | TODO | D6; `results/power/`; OSF record; single-run marker in `results/confirm/` |

## Points to weigh
- The development labels are automated segmentations transported by registration; registration was screened, not measured against landmarks.
- The evaluation cohort was used once, after pre-registration, with hash-checked models.
- Undefined patients are excluded from means, never scored 0.
- The small evaluation cohort is reported with its achieved MDE.
