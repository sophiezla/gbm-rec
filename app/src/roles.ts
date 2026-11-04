// Colour roles, fixed across app, poster and figures (plan Step 5.6).
// Same hexes as src/gbmc/figures.py:ROLE, validated with the dataviz skill's
// validate_palette.js (--mode light --pairs all: all checks pass). The
// distance/challenger pair sits in the 6-8 CVD band, so both are always
// labelled in text as well. A dark-surface palette is still open (HANDOFF).

export interface RoleStyle {
  label: string;
  color: string;
  opacity: number;
  defaultOn: boolean;
}

export const ROLES: Record<string, RoleStyle> = {
  brain: { label: "Brain", color: "#bbbbbb", opacity: 0.12, defaultOn: true },
  target: { label: "Target (tumour + cavity)", color: "#e69f00", opacity: 0.9, defaultOn: true },
  flair_region: { label: "FLAIR-abnormal region", color: "#56b4e9", opacity: 0.35, defaultOn: false },
  guideline_margin: { label: "15 mm guideline margin", color: "#0072b2", opacity: 0.25, defaultOn: true },
  guideline_margin_20: { label: "20 mm margin", color: "#0072b2", opacity: 0.12, defaultOn: false },
  model_prediction: { label: "Model prediction (margin volume)", color: "#009e73", opacity: 0.45, defaultOn: false },
  recurrence: { label: "Actual recurrence", color: "#d55e00", opacity: 0.95, defaultOn: false },
  pde: { label: "What a diffusion model draws", color: "#cc79a7", opacity: 0.4, defaultOn: false },
};

/** Distance's prediction uses the distance colour; every other model the challenger colour. */
export function modelStyle(layer: string): RoleStyle {
  const base = ROLES.model_prediction;
  return layer === "model_C0"
    ? { ...base, label: "Distance prediction", color: "#cc79a7" }
    : { ...base, label: `${layer.replace("model_", "")} prediction` };
}
