// App shell: five screens (plan Step 5.2) behind a hash router, one fixed label
// on every screen, and every number read from case.json / results.json.

import learnHtml from "./content/learn.html?raw";
import { loadIndex, loadResults } from "./case";
import { Compare } from "./compare";
import { Explore } from "./explore";
import { renderResults } from "./results";
import { Story } from "./story";

const FIXED_LABEL =
  "Research visualization of a pre-registered method comparison on public data. " +
  "Not a treatment prediction.";
const SCREENS = ["story", "explore", "compare", "results", "learn"] as const;
type Screen = (typeof SCREENS)[number];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function start(): Promise<void> {
  $("fixed-label").textContent = FIXED_LABEL;
  $("learn-body").innerHTML = learnHtml;
  const results = await loadResults();
  renderResults(results);

  let ids: string[] = [];
  try {
    ids = await loadIndex();
  } catch {
    $("summary").textContent = "No exported cases: run scripts/40_export_case.py.";
  }
  const explore = new Explore();
  const story = new Story(results);
  const compare = new Compare();
  const loaded = new Set<Screen>();

  const select = $("case-select") as HTMLSelectElement;
  for (const id of ids) select.add(new Option(id, id));
  select.addEventListener("change", () => void explore.show(select.value));
  const compareSelect = $("compare-case") as HTMLSelectElement;
  for (const id of ids) compareSelect.add(new Option(id, id));
  compareSelect.addEventListener("change", () => void compare.load(compareSelect.value));

  // The Story case: the rule-chosen patient once it exists; the first hero case until then.
  const storyCase = ids[0];

  async function route(): Promise<void> {
    const want = (location.hash.slice(1) || "story") as Screen;
    const screen: Screen = SCREENS.includes(want) ? want : "story";
    document.body.dataset.screen = screen;
    for (const s of SCREENS) {
      $(`screen-${s}`).hidden = s !== screen;
      $(`tab-${s}`).setAttribute("aria-current", s === screen ? "page" : "false");
    }
    if (loaded.has(screen) || !ids.length) return;
    loaded.add(screen);
    if (screen === "story") await story.load(storyCase);
    if (screen === "explore") await explore.show(ids[0]);
    if (screen === "compare") await compare.load(ids[0]);
  }
  window.addEventListener("hashchange", () => void route());
  await route();
}

void start();
