// case.json schema gbmc/case/1 and results.json, both written by
// scripts/40_export_case.py. The app reads every number from these files;
// none are typed into the page.

export interface Timepoint {
  id: string;
  role: string;
  weeks_from_baseline: number;
  recurrence_centroid_ras_mm: [number, number, number] | null;
}

export interface Layer {
  layer: string;
  role: string;
  file: string;
  faces: number;
}

export interface CaseDoc {
  schema: string;
  case: string;
  cohort: string;
  licence: string;
  credit: string;
  label: string;
  timepoints: Timepoint[];
  layers: Layer[];
  slices: {
    t1ce: string;
    flair: string;
    spacing_mm: number;
    recurrence_overlays: Record<string, string>;
  };
  pde: {
    label: string;
    d_white_mm2_per_day: number;
    rho_per_day: number;
    surface: string;
    frames: { day: number; file: string }[];
  } | null;
  scores: {
    status: string;
    coverage_denominator: string;
    candidates: Record<string, { pr_auc: number; lift: number; coverage_margin15: number }>;
  } | null;
}

export interface Interval {
  mean: number;
  ci_low: number;
  ci_high: number;
  n: number;
}

export interface ResultsDoc {
  label: string;
  status: "pending" | "development" | "validation";
  order: string[];
  development?: {
    cohort: string;
    challenger: string;
    delta_pr_auc_vs_C0: Record<string, Interval | null>;
  };
  validation?: {
    cohort: string;
    n_evaluable: number;
    challenger: string;
    champion: string;
    outcome: string;
    sentence: string;
    delta_pr_auc_vs_C0: Record<string, Interval | null>;
    hero_coverage_margin15: Record<string, number>;
    coverage_denominator: string;
  };
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return (await res.json()) as T;
}

export async function loadIndex(): Promise<string[]> {
  return (await getJson<{ cases: string[] }>("./cases/index.json")).cases;
}

export async function loadCase(id: string): Promise<CaseDoc> {
  const doc = await getJson<CaseDoc>(`./cases/${id}/case.json`);
  if (doc.schema !== "gbmc/case/1") throw new Error(`unsupported schema ${doc.schema}`);
  return doc;
}

export async function loadResults(): Promise<ResultsDoc> {
  try {
    return await getJson<ResultsDoc>("./results.json");
  } catch {
    return { label: "", status: "pending", order: [] };
  }
}

/** The recurrence layer belonging to a time step (none at the baseline). */
export function recurrenceLayerFor(doc: CaseDoc, step: number): string | null {
  const tp = doc.timepoints[step];
  if (!tp || step === 0) return null;
  const key = `recurrence_${tp.id}`;
  return doc.layers.some((l) => l.layer === key) ? key : null;
}

/** The lead model: the champion after the verdict, else the challenger, else none. */
export function leadModel(results: ResultsDoc): string | null {
  const v = results.validation;
  if (v) return v.champion !== "C0" ? v.champion : v.challenger;
  return results.development?.challenger ?? null;
}
