"""The champion verdict and its pre-written sentences (plan Steps 2.3, 2.4, 4.4, 4.5).

Pure functions, so the rules can be tested now and applied unchanged once.
"""

from __future__ import annotations

import numpy as np

OUTCOMES = ("Challenger champion", "Bounded null", "Inconclusive", "Challenger worse")


def classify(ci_low: float, ci_high: float, mde: float) -> str:
    """Step 4.4's four outcomes from the primary Δ PR-AUC CI and the achieved MDE."""
    if ci_low > 0:
        return "Challenger champion"
    if ci_high < 0:
        return "Challenger worse"
    return "Bounded null" if ci_high < mde else "Inconclusive"


def champion(outcome: str, challenger: str) -> str:
    """Step 2.3: the challenger is champion only if the CI's lower bound is above 0."""
    return challenger if outcome == "Challenger champion" else "C0"


def bootstrap_p(boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a mean difference of zero."""
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan")
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))


def benjamini_hochberg(pvalues: dict[str, float], q: float) -> dict[str, dict]:
    """BH-FDR over one family (the secondaries only). Returns adjusted p and rejection."""
    names = [k for k, p in pvalues.items() if np.isfinite(p)]
    p = np.array([pvalues[k] for k in names])
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adjusted[i] = running
    return {k: {"p": float(p[i]), "p_adjusted": float(adjusted[i]),
                "rejected": bool(adjusted[i] <= q)} for i, k in enumerate(names)}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def sentence(outcome: str, challenger: str, delta: float, ci: tuple[float, float],
             coverage_challenger: float, coverage_c0: float, n_candidates: int,
             coverage_ci: tuple[float, float] | None = None,
             coverage_differs: bool | None = None) -> str:
    """Fill the Step 2.4 sentence that matches the result.

    The "metrics disagree" sentence is used when exactly one of the PR-AUC and
    matched-volume coverage CIs excludes zero; the caller says whether coverage
    does (``coverage_differs``).
    """
    pr_differs = ci[0] > 0 or ci[1] < 0
    if coverage_differs is not None and coverage_differs != pr_differs:
        return ("The verdict follows the pre-registered primary, PR-AUC, which "
                f"{'does' if ci[0] > 0 else 'does not'} favour {challenger}. At the guideline "
                f"margin's volume alone, coverage {'does' if coverage_differs else 'does not'} "
                f"differ ({_pct(coverage_challenger)} vs {_pct(coverage_c0)}, 95% CI "
                f"[{coverage_ci[0]:+.3f}, {coverage_ci[1]:+.3f}]); the difference is reported "
                "as a secondary result.")
    if outcome == "Challenger champion":
        return (f"On an independent cohort, {challenger} adds {delta:+.4f} PR-AUC (95% CI "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]) over distance from the treatment target, and at "
                f"the volume of the 15 mm guideline margin captures {_pct(coverage_challenger)} of "
                f"recurrence against {_pct(coverage_c0)} for distance.")
    number = {7: "seven", 8: "eight"}.get(n_candidates, str(n_candidates))
    return (f"On an independent cohort, the best challenger from an {number}-model tournament "
            f"adds less than {ci[1]:.4f} PR-AUC over distance, and at the volume of the 15 mm "
            f"guideline margin captures {_pct(coverage_challenger)} of recurrence against "
            f"{_pct(coverage_c0)} for distance; distance remains the best available predictor.")
