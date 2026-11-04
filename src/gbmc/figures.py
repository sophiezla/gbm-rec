"""Figures of plan Step 4.7, as functions of plain data (no file reading here).

``scripts/50_figures.py`` feeds them from ``results/``; the tests feed them
synthetic numbers. Every figure is written as PNG (300 dpi) and SVG.

Colour roles (plan Step 5.6) are fixed here and shared with the app
(``app/src/roles.ts``). Light-surface palette, validated with the dataviz
skill's ``validate_palette.js --mode light --pairs all``: all checks pass. The
distance/challenger pair sits in the 6-8 CVD band, which is legal only with
secondary encoding, so both are always direct-labelled. Three roles are below
3:1 contrast on white, so every mark carrying them is labelled in text.
A dark-surface palette is still open (HANDOFF).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")   # files only; no GUI toolkit (Tk failed under pytest on Windows)

ROLE = {
    "target": "#e69f00",
    "flair_region": "#56b4e9",
    "guideline_margin": "#0072b2",
    "distance": "#cc79a7",
    "challenger": "#009e73",
    "recurrence": "#d55e00",
}
INK, MUTED, GRID, SURFACE = "#1c1c1c", "#6b6b6b", "#e4e4e0", "#fcfcfb"
FIXED_LABEL = ("Research visualization of a pre-registered method comparison on public data. "
               "Not a treatment prediction.")
ORDER = ("C0", "C0T", "C4", "C1", "C3", "C2", "C5", "C6")
NAMES = {"C0": "Distance", "C0T": "Trimmed\ndistance", "C4": "FLAIR\nedge",
         "C1": "Dist+FLAIR\nlinear", "C3": "+anatomy\nlinear", "C2": "Dist+FLAIR\nboosted",
         "C5": "3D CNN", "C6": "Ensemble"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    out = []
    for ext in ("png", "svg"):
        p = path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        out.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def demo_patients(deltas: dict[str, float]) -> dict[str, str]:
    """Plan rule: the demo patient has the median per-patient Δ; failure = lowest, best = highest.

    With an even count the lower of the two middle patients is taken, so the
    choice is one patient, fixed by rule.
    """
    ranked = sorted((v, k) for k, v in deltas.items() if np.isfinite(v))
    return {"demo": ranked[(len(ranked) - 1) // 2][1], "failure": ranked[0][1],
            "best": ranked[-1][1]}


def complexity_forest(validation: dict[str, dict] | None, development: dict[str, dict],
                      challenger: str | None, path: Path, champion: str | None = None
                      ) -> list[Path]:
    """Δ PR-AUC vs distance per candidate, simplest to most complex; zero line = distance.

    Each dict maps candidate -> {"mean", "ci_low", "ci_high"} (C0 is the zero
    line and may be absent). ``validation`` is RHUH-GBM (None before Phase 4);
    ``development`` is LUMIERE, drawn faint beside it.
    """
    import matplotlib.pyplot as plt

    names = [c for c in ORDER if c in development or c == "C0"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _style(ax)
    x = np.arange(len(names))
    ax.axhline(0, color=ROLE["distance"], linewidth=1.5)
    ax.text(len(names) - 0.5, 0, "  distance", color=INK, va="center", fontsize=9)

    def draw(series, offset, colour, alpha, label):
        for i, c in enumerate(names):
            if c == "C0" or c not in series:
                continue
            d = series[c]
            highlight = c in (champion, challenger) and series is validation
            col = ROLE["challenger"] if highlight else colour
            ax.plot([x[i] + offset] * 2, [d["ci_low"], d["ci_high"]], color=col, lw=2,
                    alpha=alpha, solid_capstyle="round")
            ax.plot(x[i] + offset, d["mean"], "o", ms=8 if series is validation else 6,
                    color=col, alpha=alpha, mec=SURFACE, mew=2)
        ax.plot([], [], "o", color=colour, alpha=alpha, label=label)

    draw(development, -0.15 if validation else 0.0, MUTED, 0.45 if validation else 0.9,
         "LUMIERE (development)")
    if validation:
        draw(validation, 0.15, INK, 1.0, "RHUH-GBM (validation)")
    tag = "champion" if champion and champion != "C0" else "challenger"
    ticks = [NAMES[c] + (f"\n({tag})" if c == challenger else "") for c in names]
    ax.set_xticks(x, ticks, fontsize=8.5)
    if challenger:
        ax.get_xticklabels()[names.index(challenger)].set_fontweight("bold")
    ax.set_ylabel("Δ PR-AUC vs distance (95% CI)", color=INK, fontsize=10)
    ax.set_xlabel("simplest  →  most complex", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncols=2)
    ax.set_title("Does more complexity help?", loc="left", color=INK, fontsize=12)
    return _save(fig, path)


def hero_coverage(bars: dict[str, dict], denominator: str, path: Path) -> list[Path]:
    """Matched-volume coverage: guideline margin, distance, challenger/champion, with CIs.

    ``bars`` maps a label to {"role", "mean", "ci_low", "ci_high"}; the plan's
    order is guideline margin, distance, champion. Coverage always carries its
    denominator (language rule), printed under the axis.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _style(ax)
    labels = list(bars)
    for i, label in enumerate(labels):
        b = bars[label]
        ax.bar(i, 100 * b["mean"], width=0.62, color=ROLE[b["role"]], edgecolor=SURFACE,
               linewidth=2)
        ax.plot([i, i], [100 * b["ci_low"], 100 * b["ci_high"]], color=INK, lw=1.4)
        ax.text(i, 100 * b["ci_high"] + 1.5, f"{100 * b['mean']:.1f}%", ha="center",
                va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("recurrence covered (%)", color=INK, fontsize=10)
    ax.set_title("Same treated volume: which catches more?", loc="left", color=INK, fontsize=12)
    fig.text(0.01, -0.04, f"Denominator: {denominator}. Volume = each patient's own 15 mm "
             "guideline margin.", fontsize=8, color=MUTED)
    return _save(fig, path)


def coverage_curve(curves: dict[str, dict[float, float]], roles: dict[str, str],
                   denominator: str, path: Path, margin_ml: float | None = None) -> list[Path]:
    """Mean coverage against treated volume (ml), one line per candidate."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    _style(ax)
    for name, curve in curves.items():
        v = sorted(curve)
        ax.plot(v, [100 * curve[k] for k in v], color=ROLE[roles[name]], lw=2)
        ax.text(v[-1], 100 * curve[v[-1]], f"  {name}", va="center", fontsize=9, color=INK)
    if margin_ml:
        ax.axvline(margin_ml, color=ROLE["guideline_margin"], lw=1)
        ax.text(margin_ml, 2, " median 15 mm margin volume", color=INK, fontsize=8)
    ax.set_xlabel("treated volume outside the target (ml)", color=INK, fontsize=10)
    ax.set_ylabel("recurrence covered (%)", color=INK, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Coverage at every volume", loc="left", color=INK, fontsize=12)
    fig.text(0.01, -0.04, f"Denominator: {denominator}.", fontsize=8, color=MUTED)
    return _save(fig, path)


def per_patient_strip(deltas: dict[str, float], path: Path, label: str) -> list[Path]:
    """Each patient's Δ, ordered; demo (median), failure and best marked."""
    import matplotlib.pyplot as plt

    picks = demo_patients(deltas)
    ranked = sorted((v, k) for k, v in deltas.items() if np.isfinite(v))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    _style(ax)
    ax.axhline(0, color=ROLE["distance"], lw=1.2)
    for i, (v, k) in enumerate(ranked):
        tag = next((t for t, p in picks.items() if p == k), None)
        ax.plot(i, v, "o", ms=8 if tag else 5, color=ROLE["challenger"] if tag else MUTED,
                mec=SURFACE, mew=1.5)
        if tag:
            ax.annotate(f"{tag}\n{k}", (i, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7.5, color=INK)
    ax.set_xticks([])
    ax.set_xlabel(f"patients, ordered by Δ (n = {len(ranked)})", color=MUTED, fontsize=9)
    ax.set_ylabel(f"Δ PR-AUC\n{label}", color=INK, fontsize=9)
    ax.set_title("Does one patient drive the mean?", loc="left", color=INK, fontsize=12)
    return _save(fig, path)


def escape_panel(rows: dict[str, dict], candidates_shown: list[str], roles: dict[str, str],
                 path: Path) -> list[Path]:
    """Descriptive: share of recurrence beyond the 15/20 mm margins, and what each model catches."""
    import matplotlib.pyplot as plt

    patients = sorted(rows, key=lambda p: -rows[p]["share_beyond_15mm"])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    _style(ax)
    x = np.arange(len(patients))
    ax.bar(x, [100 * rows[p]["share_beyond_15mm"] for p in patients], width=0.7,
           color=ROLE["recurrence"], alpha=0.35, label="beyond 15 mm")
    ax.bar(x, [100 * rows[p]["share_beyond_20mm"] for p in patients], width=0.7,
           color=ROLE["recurrence"], label="beyond 20 mm")
    for name in candidates_shown:
        ys = [100 * rows[p]["share_beyond_15mm"] * rows[p][name].get("catches_beyond_15mm", 0)
              for p in patients]
        ax.plot(x, ys, "_", ms=10, mew=2, color=ROLE[roles[name]], label=f"caught by {name}")
    ax.set_xticks([])
    ax.set_xlabel("patients (descriptive only; no test)", color=MUTED, fontsize=9)
    ax.set_ylabel("% of recurrence\noutside the target", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncols=2 + len(candidates_shown), loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Where distance fails", loc="left", color=INK, fontsize=12)
    return _save(fig, path)


def later_progression(summary: dict[str, dict], path: Path) -> list[Path]:
    """Exploratory: mean PR-AUC of each map on the new tumour at progressions 2 and 3."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    _style(ax)
    names = [c for c in ORDER if any(c in summary[k] for k in summary)]
    for j, (prog, marker) in enumerate((("2", "o"), ("3", "s"))):
        if prog not in summary:
            continue
        ys = [summary[prog].get(c, {}).get("mean_pr_auc") for c in names]
        ax.plot([i + (j - 0.5) * 0.2 for i in range(len(names))],
                [np.nan if y is None else y for y in ys], marker, ms=7, ls="none",
                color=INK if prog == "2" else MUTED,
                label=f"progression {prog} (n = {summary[prog].get('n', 0)})")
    ax.set_xticks(range(len(names)), [NAMES[c] for c in names], fontsize=8.5)
    ax.set_ylabel("PR-AUC on new tumour", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The fourth dimension (exploratory, LUMIERE development data)", loc="left",
                 color=INK, fontsize=12)
    return _save(fig, path)


def protocol_strip(steps: list[tuple[str, str]], path: Path) -> list[Path]:
    """The truth test: boxes left to right, each with a caption (e.g. a date or n)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.set_axis_off()
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    for i, (title, caption) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.35), 0.8, 0.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=SURFACE, ec=INK, lw=1))
        ax.text(i + 0.46, 0.6, title, ha="center", va="center", fontsize=9, color=INK)
        ax.text(i + 0.46, 0.15, caption, ha="center", va="center", fontsize=7.5, color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", (i + 1.06, 0.6), (i + 0.86, 0.6),
                        arrowprops={"arrowstyle": "->", "color": MUTED})
    return _save(fig, path)


def prediction_vs_reality(t1ce: np.ndarray, target: np.ndarray, margin: np.ndarray,
                          distance_top: np.ndarray, champion_top: np.ndarray,
                          recurrence: np.ndarray, coverages: dict[str, float],
                          champion_name: str, path: Path) -> list[Path]:
    """The centrepiece: one patient, one slice, four panels (plan Step 4.7).

    Arrays are 2-D slices already chosen and oriented. Panels: 15 mm guideline
    margin; distance's prediction; the champion's prediction (both at the
    margin's volume, coverage printed under each); actual recurrence.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    panels = [("15 mm guideline margin", margin, "guideline_margin", coverages.get("margin")),
              ("Distance predicts", distance_top, "distance", coverages.get("distance")),
              (f"{champion_name} predicts", champion_top, "challenger",
               coverages.get("champion")),
              ("Actual recurrence", recurrence, "recurrence", None)]
    for ax, (title, mask, role, cov) in zip(axes, panels, strict=True):
        ax.imshow(t1ce, cmap="gray", interpolation="nearest")
        ax.contour(target, levels=[0.5], colors=[ROLE["target"]], linewidths=1.2)
        overlay = np.ma.masked_where(~mask, mask)
        from matplotlib.colors import ListedColormap
        ax.imshow(overlay, cmap=ListedColormap([ROLE[role]]), alpha=0.55,
                  interpolation="nearest")
        if role != "recurrence":
            ax.contour(recurrence, levels=[0.5], colors=[ROLE["recurrence"]], linewidths=1.0)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        if cov is not None:
            ax.set_xlabel(f"covers {100 * cov:.1f}% of recurrence\noutside the target",
                          fontsize=9, color=INK)
    fig.text(0.01, -0.02, FIXED_LABEL, fontsize=8, color=MUTED)
    return _save(fig, path)
