// Story (plan Step 5.2, screen 1): the guided walkthrough of one rule-chosen
// patient. The model commits before the truth is shown. Step text lives in
// content/story.html so the audit script checks it; this file only decides
// which layers each step shows and where the numbers come from.

import storyHtml from "./content/story.html?raw";
import { leadModel, loadCase, type CaseDoc, type ResultsDoc } from "./case";
import { modelStyle, ROLES } from "./roles";
import { forestSvg } from "./results";
import { Viewer3D } from "./viewer3d";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class Story {
  private viewer: Viewer3D;
  private steps: HTMLElement[] = [];
  private index = 0;
  private doc: CaseDoc | null = null;

  constructor(private results: ResultsDoc) {
    this.viewer = new Viewer3D($("story-3d"));
    const holder = document.createElement("div");
    const n = results.order.filter((c) => c !== "C5" || results.development?.delta_pr_auc_vs_C0?.C5)
      .length;
    holder.innerHTML = storyHtml.replace("{{n_models}}", n ? String(n) : "Eight");
    this.steps = [...holder.querySelectorAll<HTMLElement>("section")];
    $("story-prev").addEventListener("click", () => this.go(this.index - 1));
    $("story-next").addEventListener("click", () => this.go(this.index + 1));
    document.addEventListener("keydown", (e) => {
      if (document.body.dataset.screen !== "story") return;
      if (e.key === "ArrowRight" || e.key === "PageDown") this.go(this.index + 1);
      if (e.key === "ArrowLeft" || e.key === "PageUp") this.go(this.index - 1);
    });
  }

  /** The case: RHUH-GBM's median-Δ patient after the verdict; a LUMIERE hero case before. */
  async load(caseId: string): Promise<void> {
    const doc = await loadCase(caseId);
    this.doc = doc;
    this.viewer.clear();
    await Promise.all(doc.layers.map((l) => {
      const style = l.role === "model_prediction" ? modelStyle(l.layer) : ROLES[l.role];
      return style ? this.viewer.add(l.layer, `./cases/${caseId}/${l.file}`, style, false) : null;
    }));
    this.viewer.frame("brain");
    $("story-case").textContent = this.results.status === "validation"
      ? `${doc.cohort} case ${doc.case}, chosen by rule: the median patient.`
      : `${doc.cohort} case ${doc.case} (development data). After the confirmatory run, ` +
        "the RHUH-GBM patient with the median result replaces it.";
    this.go(0);
  }

  private layersFor(step: HTMLElement): Set<string> {
    const doc = this.doc!;
    const lead = leadModel(this.results);
    const keys = new Set<string>();
    for (const token of (step.dataset.show ?? "").split(/\s+/)) {
      if (token.startsWith("model:")) {
        const name = token.slice(6) === "lead" ? lead : token.slice(6);
        if (name) keys.add(`model_${name}`);
      } else if (token === "recurrence") {
        const first = doc.timepoints[1];
        if (first) keys.add(`recurrence_${first.id}`);
      } else {
        for (const l of doc.layers) if (l.role === token) keys.add(l.layer);
      }
    }
    return keys;
  }

  private go(i: number): void {
    if (!this.doc) return;
    this.index = Math.max(0, Math.min(this.steps.length - 1, i));
    const step = this.steps[this.index];
    const keys = this.layersFor(step);
    this.viewer.showOnly(keys);
    const text = $("story-text");
    text.innerHTML = step.innerHTML;
    const missing = [...keys].filter((k) => !this.viewer.has(k));
    if (missing.some((k) => k.startsWith("model_"))) {
      text.insertAdjacentHTML("beforeend",
        "<p class='pending'>The model predictions appear here once the tournament has run.</p>");
    }
    if (step.dataset.scores) text.insertAdjacentHTML("beforeend", this.scoreTable());
    if (step.dataset.results) text.insertAdjacentHTML("beforeend", forestSvg(this.results));
    $("story-progress").textContent = `Step ${this.index + 1} of ${this.steps.length}`;
    ($("story-prev") as HTMLButtonElement).disabled = this.index === 0;
    ($("story-next") as HTMLButtonElement).disabled = this.index === this.steps.length - 1;
  }

  private scoreTable(): string {
    const s = this.doc?.scores;
    const lead = leadModel(this.results);
    if (!s || !lead) return "<p class='pending'>Scores appear here once the tournament has run.</p>";
    const row = (label: string, c: string) => s.candidates[c]
      ? `<tr><th scope="row">${label}</th><td>${(100 * s.candidates[c].coverage_margin15).toFixed(1)}%</td></tr>`
      : "";
    return `<table class="scores"><caption>Coverage at the guideline margin's volume
      (denominator: ${s.coverage_denominator}; ${s.status})</caption>
      ${row("Guideline margin", "C0T")}${row("Distance", "C0")}${row(lead, lead)}</table>`;
  }
}
