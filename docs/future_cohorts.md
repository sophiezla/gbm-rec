# Candidate cohorts for a later external test (explored 2026-11-16)

**Decided 2026-11-16 (`DECISIONS.md` D18): MU-Glioma-Post is the sealed test of a later follow-up study.** Models are tweaked only on LUMIERE and, after its verdict, RHUH-GBM. MU's images stay undownloaded until the follow-up is pre-registered, and `tests/test_config.py::test_mu_glioma_post_stays_sealed` enforces this. The roadmap is at the end.

Not part of the pre-registered study. This is a feasibility look for a possible follow-up test after the RHUH-GBM verdict, at the user's request. **Only documentation and clinical metadata were read: no image and no segmentation was downloaded or opened**, so both cohorts could still serve as a sealed test later. Any use needs SRC approval and its own pre-registration.

## MU-Glioma-Post (University of Missouri, TCIA)

- Source: https://www.cancerimagingarchive.net/collection/mu-glioma-post/ (DOI 10.7937/7k9k-3c83), CC BY 4.0, public download.
  - Images: 11 GB of skull-stripped NIfTI, SRI24-registered (same atlas as RHUH-GBM), via IBM Aspera.
  - Labels (NETC, SNFH, ET, resection cavity): nnU-Net, then "manually validated and refined by neuroradiologists" (TCIA page).
- Downloaded into `data/raw/mu_glioma_post/` (not in git):
  - `MU-Glioma-Post_ClinicalData-July2025.xlsx`, SHA-256 `4341badf954511d67fc52b82b3b488ed7323f7e46d249a354ff9b3a62269c79b`;
  - `MU-Glioma-Post_Segmentation_Volumes.xlsx`, SHA-256 `4ba60e04279e91d920974f5641b313fd04fc2d0319f782f37567b48bc7795fa6`.
- Eligibility funnel from the clinical table (203 patients):
  1. GBM diagnosis: 158.
  2. IDH-wildtype (IDH1 = 0, IDH2 not mutated): 148. Another 5 GBM have IDH1 unknown and 5 are IDH-mutant.
  3. No previous brain tumour: 143.
  4. First progression seen on imaging (type 1 or 3): 99. Another 14 have clinical-only progression, and 29 none.
  5. **No early post-operative scan.** The first MRI is "the first follow up (post-treatment) MRI" (data dictionary), a median 93.5 days after surgery (quartiles 72–107). None is within 7 days; 8 are within 30 days. This contradicts the TCIA page's "typically include the immediate post-operative scan". <!-- audit: ignore (working note: metadata counts and a journal name) -->
  6. If the baseline is instead the first post-treatment MRI taken at least 56 days before progression, and a later MRI falls between 14 days before and 60 days after the first progression: **44 patients** (46 at 28 days). The first progression comes a median 179 days after surgery.
- Inconsistencies noticed:
  - The volume table lists 286 patient IDs, against 203 released, and has no timepoint column, so it cannot show which scan carries enhancing tumour.
  - The TCIA page's 3 T counts disagree with each other.
- Fit: the cohort is large, but it needs a different baseline (post-chemoradiation, not early post-op). That changes the target definition and brings treatment effects, such as pseudoprogression, into the baseline. Whether enhancing recurrence lies outside the target is unknown until the labels are opened.

## UCSF-ALPTDG (UCSF)

- Source: https://imagingdatasets.ucsf.edu/dataset/2 (DOI 10.58078/C21592); paper: Fields et al., Radiology: AI 2024, 10.1148/ryai.230182 (paywalled). <!-- audit: ignore (working note: metadata counts and a journal name) -->
- 298 patients with diffuse glioma, at two consecutive post-treatment follow-ups (596 scans).
  - Expert voxelwise segmentations at each timepoint: ET, SNFH, necrotic core, resection cavity.
  - Longitudinal ET/SNFH changes are annotated on subtraction images.
  - Clinical data: demographics, treatment history, extent of resection, genetics.
- Access: "Request Data", non-commercial use under a data use agreement (RSNA Atlas). The clinical table is behind it too.
- Not verified: the GBM count, how the scan pairs were chosen, whether the second scan is a progression, and whether there is an early post-op scan. The scan dates (surveillance MRIs, 2018–2019) suggest routine follow-up pairs, not baseline-to-recurrence pairs.

## Summary (before the decision)

MU-Glioma-Post is the more usable candidate on present evidence: it is public, has progression dates, and gives about 44 IDH-wildtype GBM pairs if the baseline is redefined. UCSF-ALPTDG has better labels, but its fit is unknown until access is granted.

## Follow-up roadmap (D18)

The current study comes first; none of this starts before step 1 is complete.

1. **Finish the current study, unchanged.**
   - Partner trains the C5 release model.
   - `03_` (refit and hash), `04_` (power), `07_ --summary` (round-trip threshold).
   - Post the OSF pre-registration, download RHUH-GBM, lock the cohort (`20_`–`24_`).
   - The C5 RHUH-GBM maps (`14_`), then `30_` once and `31_`.
   - The verdict stands as pre-registered, whatever the follow-up later finds.
2. **SRC approval** for MU-Glioma-Post as a new data source, before any MU image is downloaded.
3. **Develop ("tweak") on LUMIERE + RHUH-GBM only.** Every change is logged in the follow-up's own records and labelled exploratory. Ideas the current results point to, none tested:
   - The C1 + ADC arm. RHUH-GBM has ADC; MU does not list ADC among its four sequences, so a model needing ADC could not be tested on MU. Check before choosing.
   - Features that capture where distance fails (the "escape" region).
   - Fewer candidates, fixed in advance, rather than a new tournament.
4. **Pre-register the follow-up on OSF** before any MU image is downloaded:
   - the one frozen model (with its SHA-256) and the comparator, C0 distance;
   - MU eligibility: IDH-wildtype GBM, no earlier brain tumour, imaging-confirmed first progression, baseline = first post-treatment MRI at least 56 days before progression, plus an MRI from 14 days before to 60 days after progression;
   - the target definition on a post-treatment scan, and the label mapping (NETC/SNFH/ET/resection cavity);
   - the primary (Δ PR-AUC vs C0, BCa CI), the verdict rule and the power at about 44 patients.
5. **Then download MU** (11 GB, Aspera), adapt the pipeline (SRI24 grid as for RHUH-GBM; MU's label names), lock the cohort, and score the frozen model **once**, with a run-once marker as in `30_`.

Expected limits:
- About 44 patients gives wide confidence intervals, similar to LUMIERE's.
- A post-treatment baseline means the target includes treatment effects, so the MU result answers a slightly different question than the RHUH-GBM one, and must be reported that way.
