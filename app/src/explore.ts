// Explore (plan Step 5.2, screen 2): any case; rotate, slice, toggle layers,
// time slider over real scans, model selector, reveal toggle, and PDE mode.

import { Niivue, SHOW_RENDER } from "@niivue/niivue";
import { loadCase, recurrenceLayerFor, type CaseDoc } from "./case";
import { modelStyle, ROLES, type RoleStyle } from "./roles";
import { Viewer3D } from "./viewer3d";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export class Explore {
  private viewer: Viewer3D;
  private nv = new Niivue({ backColor: [0.07, 0.07, 0.09, 1], multiplanarShowRender: SHOW_RENDER.NEVER });
  private visible = new Map<string, boolean>();
  private model: string | null = null;
  private doc: CaseDoc | null = null;
  private pdeTimer: number | null = null;

  constructor() {
    this.viewer = new Viewer3D($("viewer-3d"));
    void this.nv.attachTo("slices");
    ($("time") as HTMLInputElement).addEventListener("input", (e) => {
      if (this.doc) this.applyTime(Number((e.target as HTMLInputElement).value));
    });
    ($("model-select") as HTMLSelectElement).addEventListener("change", (e) => {
      this.model = (e.target as HTMLSelectElement).value || null;
      this.refreshModels();
    });
    $("pde-play").addEventListener("click", () => this.togglePde());
  }

  styleFor(role: string, layer: string): RoleStyle | undefined {
    return role === "model_prediction" ? modelStyle(layer) : ROLES[role];
  }

  async show(id: string): Promise<void> {
    this.stopPde();
    $("summary").textContent = "Loading case…";
    const doc = await loadCase(id);
    this.doc = doc;
    $("credit").textContent = `Data: ${doc.credit}.`;
    this.viewer.clear();
    this.buildToggles(doc);
    this.buildModelSelect(doc);
    await Promise.all(doc.layers.map((l) => {
      const style = this.styleFor(l.role, l.layer);
      return style ? this.viewer.add(l.layer, `./cases/${id}/${l.file}`, style, false) : null;
    }));
    if (doc.pde) {
      await Promise.all(doc.pde.frames.map((f) =>
        this.viewer.add(`pde:${f.day}`, `./cases/${id}/${f.file}`, ROLES.pde, false)));
    }
    $("pde-play").hidden = !doc.pde;
    this.viewer.frame("brain");
    const slider = $("time") as HTMLInputElement;
    slider.max = String(doc.timepoints.length - 1);
    slider.value = "0";
    await this.nv.loadVolumes([{ url: `./cases/${id}/${doc.slices.t1ce}` }]);
    this.nv.setSliceType(this.nv.sliceTypeMultiplanar);
    this.applyTime(0);
  }

  private buildToggles(doc: CaseDoc): void {
    const box = $("layers");
    box.querySelectorAll("label").forEach((n) => n.remove());
    for (const role of [...new Set(doc.layers.map((l) => l.role))]) {
      const style = ROLES[role];
      if (!style || role === "model_prediction") continue;
      this.visible.set(role, this.visible.get(role) ?? style.defaultOn);
      const label = document.createElement("label");
      label.className = "layer";
      label.innerHTML =
        `<input type="checkbox" ${this.visible.get(role) ? "checked" : ""}/>` +
        `<span class="swatch" style="background:${style.color}"></span>${style.label}`;
      label.querySelector("input")!.addEventListener("change", (e) => {
        this.visible.set(role, (e.target as HTMLInputElement).checked);
        this.applyTime(Number(($("time") as HTMLInputElement).value));
      });
      box.appendChild(label);
    }
  }

  private buildModelSelect(doc: CaseDoc): void {
    const select = $("model-select") as HTMLSelectElement;
    select.innerHTML = "";
    const models = doc.layers.filter((l) => l.role === "model_prediction");
    select.add(new Option(models.length ? "none" : "none yet (tournament not run)", ""));
    for (const l of models) select.add(new Option(modelStyle(l.layer).label, l.layer));
    select.disabled = models.length === 0;
    if (this.model && !models.some((l) => l.layer === this.model)) this.model = null;
    select.value = this.model ?? "";
  }

  private refreshModels(): void {
    if (!this.doc) return;
    for (const l of this.doc.layers) {
      if (l.role === "model_prediction") this.viewer.setVisible(l.layer, l.layer === this.model);
    }
  }

  private applyTime(step: number): void {
    const doc = this.doc!;
    const shown = recurrenceLayerFor(doc, step);
    for (const l of doc.layers) {
      if (l.role === "recurrence") {
        this.viewer.setVisible(l.layer, l.layer === shown && (this.visible.get("recurrence") ?? false));
      } else if (l.role !== "model_prediction") {
        this.viewer.setVisible(l.layer, this.visible.get(l.role) ?? false);
      }
    }
    this.refreshModels();
    const tp = doc.timepoints[step];
    $("time-label").textContent =
      `${tp.role} (${tp.id}), ${tp.weeks_from_baseline} weeks after the post-op baseline`;
    void this.sliceOverlay(step);
    const on = [...this.visible.entries()].filter(([, v]) => v).map(([k]) => ROLES[k]?.label ?? k);
    $("summary").textContent =
      `${doc.cohort} case ${doc.case}, ${tp.role}. Showing: ${on.join(", ")}` +
      (this.model ? `, ${modelStyle(this.model).label}.` : ".");
  }

  /** Recurrence overlay on the slices, same timepoint and toggle as in 3D. */
  private async sliceOverlay(step: number): Promise<void> {
    const doc = this.doc!;
    const file = doc.slices.recurrence_overlays[doc.timepoints[step].id];
    const want = step > 0 && (this.visible.get("recurrence") ?? false) && file ? file : null;
    while (this.nv.volumes.length > 1) this.nv.removeVolume(this.nv.volumes[1]);
    if (want) {
      await this.nv.addVolumeFromUrl({ url: `./cases/${doc.case}/${want}`, colormap: "red",
                                       opacity: 0.9, cal_min: 0.5, cal_max: 1 });
      const c = doc.timepoints[step].recurrence_centroid_ras_mm;
      if (c) this.nv.scene.crosshairPos = this.nv.mm2frac(c);
    }
    this.nv.updateGLVolume();
  }

  /** PDE mode: step through the Fisher-KPP frames, day 0 to the horizon. */
  private togglePde(): void {
    if (this.pdeTimer !== null) return this.stopPde();
    const frames = this.doc?.pde?.frames ?? [];
    if (!frames.length) return;
    let i = 0;
    $("pde-play").textContent = "Stop";
    const tick = () => {
      frames.forEach((f, j) => this.viewer.setVisible(`pde:${f.day}`, j === i));
      $("pde-label").textContent =
        `${this.doc!.pde!.label}: day ${Math.round(frames[i].day)} ` +
        `(Fisher-KPP, D ${this.doc!.pde!.d_white_mm2_per_day} mm²/day, ` +
        `ρ ${this.doc!.pde!.rho_per_day}/day)`;
      i = (i + 1) % frames.length;
    };
    tick();
    this.pdeTimer = window.setInterval(tick, 900);
  }

  private stopPde(): void {
    if (this.pdeTimer !== null) window.clearInterval(this.pdeTimer);
    this.pdeTimer = null;
    this.doc?.pde?.frames.forEach((f) => this.viewer.setVisible(`pde:${f.day}`, false));
    $("pde-play").textContent = "Play diffusion model";
    $("pde-label").textContent = "";
  }
}
