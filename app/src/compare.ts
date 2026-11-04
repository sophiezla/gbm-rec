// Compare (plan Step 5.2, screen 3): two models side by side at the same
// treated volume, cameras linked, each with its coverage number.

import { loadCase, type CaseDoc } from "./case";
import { modelStyle, ROLES } from "./roles";
import { Viewer3D } from "./viewer3d";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class Compare {
  private views: [Viewer3D, Viewer3D];
  private doc: CaseDoc | null = null;
  private linking = false;

  constructor() {
    this.views = [new Viewer3D($("compare-a")), new Viewer3D($("compare-b"))];
    const [a, b] = this.views;
    a.onCameraChange(() => this.link(b, a));
    b.onCameraChange(() => this.link(a, b));
    for (const side of ["a", "b"] as const) {
      $(`compare-select-${side}`).addEventListener("change", () => this.refresh());
    }
    $("compare-reveal").addEventListener("change", () => this.refresh());
  }

  private link(follower: Viewer3D, leader: Viewer3D): void {
    if (this.linking) return;
    this.linking = true;
    follower.followCamera(leader);
    this.linking = false;
  }

  async load(caseId: string): Promise<void> {
    const doc = await loadCase(caseId);
    this.doc = doc;
    const models = doc.layers.filter((l) => l.role === "model_prediction");
    for (const [i, view] of this.views.entries()) {
      view.clear();
      await Promise.all(doc.layers.map((l) => {
        const style = l.role === "model_prediction" ? modelStyle(l.layer) : ROLES[l.role];
        return style ? view.add(l.layer, `./cases/${caseId}/${l.file}`, style, false) : null;
      }));
      view.frame("brain");
      const select = $(`compare-select-${i === 0 ? "a" : "b"}`) as HTMLSelectElement;
      select.innerHTML = "";
      for (const l of models) select.add(new Option(modelStyle(l.layer).label, l.layer));
      select.disabled = models.length === 0;
      if (models.length) select.selectedIndex = Math.min(i, models.length - 1);
    }
    $("compare-note").textContent = models.length ? ""
      : "Model predictions appear here once the tournament has run.";
    this.refresh();
  }

  private refresh(): void {
    const doc = this.doc;
    if (!doc) return;
    const reveal = ($("compare-reveal") as HTMLInputElement).checked;
    const first = doc.timepoints[1] ? `recurrence_${doc.timepoints[1].id}` : "";
    for (const [i, view] of this.views.entries()) {
      const side = i === 0 ? "a" : "b";
      const model = ($(`compare-select-${side}`) as HTMLSelectElement).value;
      const keys = new Set(["brain", "target", model]);
      if (reveal && first) keys.add(first);
      view.showOnly(keys);
      const name = model.replace("model_", "");
      const cov = doc.scores?.candidates[name]?.coverage_margin15;
      $(`compare-score-${side}`).textContent = cov === undefined ? "" :
        `${name}: ${(100 * cov).toFixed(1)}% of recurrence outside the target ` +
        `(${doc.scores!.status})`;
    }
  }
}
