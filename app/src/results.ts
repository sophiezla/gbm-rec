// Results (plan Step 5.2, screen 4): the complexity forest plot and the
// matched-volume coverage bars, drawn as SVG from results.json. Each mark has
// a hover tooltip (<title>) and the numbers are also given as a table.

import type { Interval, ResultsDoc } from "./case";

const NAMES: Record<string, string> = {
  C0: "Distance", C0T: "Trimmed distance", C4: "FLAIR edge", C1: "Distance + FLAIR, linear",
  C3: "+ anatomy, linear", C2: "Distance + FLAIR, boosted", C5: "3D CNN", C6: "Ensemble",
};
const INK = "var(--fg)", MUTED = "var(--muted)", GRID = "var(--border)";
const DISTANCE = "#cc79a7", CHALLENGER = "#009e73";

const fmt = (v: number) => (v >= 0 ? "+" : "") + v.toFixed(4);

/** Δ PR-AUC vs distance, simplest to most complex, validation over faint development points. */
export function forestSvg(results: ResultsDoc): string {
  const dev = results.development?.delta_pr_auc_vs_C0 ?? {};
  const val = results.validation?.delta_pr_auc_vs_C0 ?? {};
  const names = results.order.filter((c) => c === "C0" || dev[c] || val[c]);
  if (!results.development) {
    return "<p class='pending'>The forest plot appears here once the tournament has run.</p>";
  }
  const all = [...Object.values(dev), ...Object.values(val)].filter(Boolean) as Interval[];
  const lo = Math.min(0, ...all.map((d) => d.ci_low));
  const hi = Math.max(0, ...all.map((d) => d.ci_high));
  const W = 640, H = 300, L = 70, R = 20, T = 20, B = 70;
  const x = (i: number) => L + ((i + 0.5) * (W - L - R)) / names.length;
  const y = (v: number) => T + ((hi - v) * (H - T - B)) / (hi - lo || 1);
  const lead = results.validation?.challenger ?? results.development.challenger;
  const marks: string[] = [];
  const mark = (d: Interval | null | undefined, cx: number, colour: string, r: number,
                label: string) => {
    if (!d) return;
    marks.push(`<g><title>${label}: Δ ${fmt(d.mean)} (95% CI ${fmt(d.ci_low)} to ${fmt(d.ci_high)}), n = ${d.n}</title>
      <line x1="${cx}" x2="${cx}" y1="${y(d.ci_low)}" y2="${y(d.ci_high)}" stroke="${colour}" stroke-width="2" stroke-linecap="round"/>
      <circle cx="${cx}" cy="${y(d.mean)}" r="${r}" fill="${colour}" stroke="var(--panel)" stroke-width="2"/>
      <rect x="${cx - 12}" y="${y(d.ci_high) - 4}" width="24" height="${y(d.ci_low) - y(d.ci_high) + 8}" fill="transparent"/></g>`);
  };
  names.forEach((c, i) => {
    const hasVal = Boolean(results.validation);
    mark(dev[c], x(i) - (hasVal ? 8 : 0), MUTED, 4, `${NAMES[c]}, LUMIERE (development)`);
    mark(val[c], x(i) + 8, c === lead ? CHALLENGER : INK, 5, `${NAMES[c]}, RHUH-GBM (validation)`);
  });
  const ticks = names.map((c, i) => `<text x="${x(i)}" y="${H - B + 18}" text-anchor="middle"
      font-size="11" fill="${c === lead ? INK : MUTED}" font-weight="${c === lead ? 700 : 400}">${c}</text>`);
  const rows = names.filter((c) => c !== "C0").map((c) => `<tr><th scope="row">${NAMES[c]}</th>
      <td>${dev[c] ? fmt(dev[c]!.mean) : "–"}</td><td>${val[c] ? fmt(val[c]!.mean) : "–"}</td></tr>`);
  return `<figure class="chart">
    <figcaption>Does more complexity help? Δ PR-AUC vs distance, 95% CI. Grey: LUMIERE
      (development)${results.validation ? "; dark: RHUH-GBM (validation)" : ""}.</figcaption>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Forest plot of each model's difference from distance">
      <line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="${DISTANCE}" stroke-width="2"/>
      <text x="${x(0)}" y="${y(0) - 6}" text-anchor="middle" font-size="11" fill="${INK}">distance = 0</text>
      <line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="${GRID}"/>
      <text x="${L - 8}" y="${y(hi)}" text-anchor="end" font-size="11" fill="${MUTED}">${fmt(hi)}</text>
      <text x="${L - 8}" y="${y(lo)}" text-anchor="end" font-size="11" fill="${MUTED}">${fmt(lo)}</text>
      ${marks.join("")}${ticks.join("")}
      <text x="${(W + L) / 2}" y="${H - 20}" text-anchor="middle" font-size="11" fill="${MUTED}">simplest → most complex</text>
    </svg>
    <details><summary>Table</summary><table><tr><th>Model</th><th>LUMIERE Δ</th><th>RHUH-GBM Δ</th></tr>${rows.join("")}</table></details>
  </figure>`;
}

/** Matched-volume coverage bars (validation only): guideline margin, distance, lead model. */
export function heroSvg(results: ResultsDoc): string {
  const v = results.validation;
  if (!v) return "<p class='pending'>The coverage bars appear here after the confirmatory run.</p>";
  const bars: [string, string, string][] = [["Guideline margin", "C0T", "#0072b2"],
    ["Distance", "C0", DISTANCE], [v.challenger, v.challenger, CHALLENGER]];
  const W = 420, H = 240, base = 200, scale = 180;
  const g = bars.map(([label, c, colour], i) => {
    const value = v.hero_coverage_margin15[c];
    const x0 = 40 + i * 125;
    return `<g><title>${label}: ${(100 * value).toFixed(1)}% of ${v.coverage_denominator}</title>
      <rect x="${x0}" y="${base - scale * value}" width="90" height="${scale * value}" rx="4" fill="${colour}"/>
      <text x="${x0 + 45}" y="${base - scale * value - 6}" text-anchor="middle" font-size="12" fill="${INK}">${(100 * value).toFixed(1)}%</text>
      <text x="${x0 + 45}" y="${base + 16}" text-anchor="middle" font-size="11" fill="${MUTED}">${label}</text></g>`;
  });
  return `<figure class="chart"><figcaption>Same treated volume, which catches more?
    Denominator: ${v.coverage_denominator}.</figcaption>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Coverage at the guideline margin's volume">${g.join("")}</svg></figure>`;
}

export function renderResults(results: ResultsDoc): void {
  const el = document.getElementById("results-body")!;
  const verdict = results.validation
    ? `<p class="verdict"><strong>${results.validation.outcome}.</strong> ${results.validation.sentence}</p>`
    : "<p class='pending'>The verdict appears here after the one confirmatory run on RHUH-GBM.</p>";
  el.innerHTML = verdict + forestSvg(results) + heroSvg(results);
}
