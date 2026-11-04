"""Tournament candidates (plan Step 1.4), folds (Step 1.5) and the selection rule (Step 1.6).

Candidates in complexity order, which breaks ties and sets the forest plot's x-axis:

====  ==============================  =============  =============================
ID    Candidate                       Inputs         Learner
====  ==============================  =============  =============================
C0    distance                        f1             none: -f1
C0T   trimmed distance                c0t            none: -c0t
C4    FLAIR edge                      f3             none: -f3
C1    distance + FLAIR, linear        f1-f4          L2 logistic, patient-balanced
C3    + anatomy, linear               f1-f5          L2 logistic, patient-balanced
C2    distance + FLAIR, non-linear    f1-f4          LightGBM, gbm-4d's parameters
C5    3D CNN (gated, Step 0.4)        images         not implemented here
C6    rank-average ensemble           C1, C3, C2     mean within-patient percentile rank
====  ==============================  =============  =============================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from gbmc.features import COLUMNS

ORDER = ("C0", "C0T", "C4", "C1", "C3", "C2", "C5", "C6")
CHALLENGER_POOL = ("C4", "C1", "C3", "C2", "C5", "C6")


def _col(name: str) -> int:
    return COLUMNS.index(name)


@dataclass(frozen=True)
class RuleCandidate:
    """A candidate with nothing fitted: score = -column."""

    name: str
    column: str

    def score(self, matrix: np.ndarray) -> np.ndarray:
        return -matrix[:, _col(self.column)].astype(np.float64)


RULES = {
    "C0": RuleCandidate("C0", "f1"),
    "C0T": RuleCandidate("C0T", "c0t"),
    "C4": RuleCandidate("C4", "f3"),
}


#: Plan Step 1.4 fixes these values; they are the ones gbm-4d used.
LIGHTGBM_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                   "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
                   "colsample_bytree": 0.8, "reg_lambda": 1.0, "verbose": -1}
LOGISTIC_PARAMS = {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"}

@dataclass(frozen=True)
class LearnedCandidate:
    name: str
    columns: tuple[str, ...]
    learner: str               # "logistic_l2" or "lightgbm"
    patient_balanced: bool

    @property
    def column_index(self) -> list[int]:
        return [_col(c) for c in self.columns]

    def build(self, seed: int):
        """A fresh estimator with an explicit random_state.

        LightGBM uses :data:`LIGHTGBM_PARAMS`. The logistic model standardises first,
        because distances run to 300 mm and L2 would otherwise be scale-dependent.
        """
        if self.learner == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(random_state=seed, **LIGHTGBM_PARAMS)
        if self.learner == "logistic_l2":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(random_state=seed, **LOGISTIC_PARAMS))
        raise KeyError(f"unknown learner {self.learner!r}")


LEARNED = {
    "C1": LearnedCandidate("C1", ("f1", "f2", "f3", "f4"), "logistic_l2", True),
    "C3": LearnedCandidate("C3", ("f1", "f2", "f3", "f4", "f5"), "logistic_l2", True),
    "C2": LearnedCandidate("C2", ("f1", "f2", "f3", "f4"), "lightgbm", False),
}

#: Fixed now, never by how the members score (plan Step 1.4, C6 rule 1).
#: C5 joins only if the Step 0.4 gate passes.
ENSEMBLE_MEMBERS = ("C1", "C3", "C2")


def fit(candidate: LearnedCandidate, matrices: Sequence[np.ndarray],
        truths: Sequence[np.ndarray], seed: int):
    """Fit on already-subsampled training rows, one block per patient.

    With ``patient_balanced`` every training patient carries the same total
    weight, so a patient with a large recurrence (and so many sampled rows)
    does not dominate the fit.
    """
    x = np.concatenate([m[:, candidate.column_index] for m in matrices])
    y = np.concatenate(truths)
    model = candidate.build(seed)
    if candidate.patient_balanced:
        weights = np.concatenate([np.full(len(t), 1.0 / len(t)) for t in truths])
        weights *= len(y) / weights.sum()
        # Pipeline(StandardScaler, LogisticRegression): route the weights to the model step.
        step = model.steps[-1][0]
        model.fit(x, y, **{f"{step}__sample_weight": weights})
    else:
        model.fit(x, y)
    return model


def predict(candidate: LearnedCandidate, model, matrix: np.ndarray) -> np.ndarray:
    return model.predict_proba(matrix[:, candidate.column_index])[:, 1]


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Within-patient ranks scaled to [0, 1]; ties share the average rank."""
    n = scores.size
    if n < 2:
        return np.zeros(n)
    return (rankdata(scores, method="average") - 1.0) / (n - 1.0)


def rank_average(member_scores: Sequence[np.ndarray]) -> np.ndarray:
    """C6 for one patient: unweighted mean of the members' percentile ranks."""
    return np.mean([percentile_ranks(s) for s in member_scores], axis=0)


def patient_folds(patients: Sequence[str], n_folds: int, seed: int) -> list[list[str]]:
    """One seeded, patient-grouped split shared by every candidate.

    Patients are sorted first, so the split depends only on the ID set and the
    seed, then permuted and dealt round-robin (63 patients, 7 folds -> 9 each).
    """
    ordered = sorted(patients)
    order = np.random.default_rng(seed).permutation(len(ordered))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, idx in enumerate(order):
        folds[i % n_folds].append(ordered[idx])
    return [sorted(f) for f in folds]


def select_challenger(mean_pr_auc: dict[str, float], tie_margin: float,
                      order: Sequence[str] = ORDER,
                      pool: Sequence[str] = CHALLENGER_POOL) -> str:
    """The Step 1.6 rule.

    Take the pool candidate with the highest mean held-out PR-AUC. If any
    simpler candidate is within ``tie_margin`` of it, the simplest such one is
    chosen instead. C6 is last in ``order``, so it wins only by beating every
    simpler candidate by more than the margin.
    """
    scored = [c for c in pool if c in mean_pr_auc and np.isfinite(mean_pr_auc[c])]
    if not scored:
        raise ValueError("no challenger-pool candidate has a finite mean PR-AUC")
    best = max(scored, key=lambda c: mean_pr_auc[c])
    close = [c for c in scored if mean_pr_auc[best] - mean_pr_auc[c] <= tie_margin]
    return min(close, key=order.index)
