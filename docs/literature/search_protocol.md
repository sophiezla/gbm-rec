# Literature scan: search protocol (plan Step 0.6)

Status: **DRAFT, not yet committed.** The plan requires this file and `extraction_form.csv` to be committed *before* any paper is read. Nothing has been searched or read yet.

## Question
Which published studies predict *where* glioblastoma recurs (voxel-level or region-level), and how do they compare with this project on cohort size, external validation, a distance-to-tumour baseline, metric, and pre-registration?

## Database
PubMed (https://pubmed.ncbi.nlm.nih.gov/), searched once. The search date and the hit count are recorded below when the search is run.

## Search string
```
("Glioblastoma"[MeSH Terms] OR glioblastoma[tiab] OR GBM[tiab] OR "high-grade glioma"[tiab])
AND ("Neoplasm Recurrence, Local"[MeSH Terms] OR recurrence[tiab] OR recurrent[tiab] OR relapse[tiab] OR progression[tiab])
AND (location[tiab] OR spatial[tiab] OR voxel*[tiab] OR "site of recurrence"[tiab] OR "recurrence pattern*"[tiab] OR "pattern of recurrence"[tiab] OR infiltrat*[tiab])
AND (predict*[tiab] OR model*[tiab] OR "machine learning"[tiab] OR "deep learning"[tiab] OR radiomic*[tiab])
AND ("Magnetic Resonance Imaging"[MeSH Terms] OR MRI[tiab] OR "magnetic resonance"[tiab])
AND english[la]
```

## Eligibility (decided on title and abstract, then full text)
Include: original research that predicts the location of glioblastoma (or high-grade glioma including glioblastoma) recurrence or progression from MRI, at voxel level or region level, and reports a quantitative performance measure.

Exclude: reviews and meta-analyses (their primary studies may be screened instead); studies that predict *whether* or *when* a tumour recurs but not *where*; pseudoprogression-versus-progression classification; segmentation-only studies; case reports; conference abstracts without a full paper.

## Cap of 25 papers
PROPOSED, needs the user's approval: sort the hits by PubMed "Best Match", screen them in that order, and stop after the 25th included study. The number of records screened when the cap is reached is reported. If fewer than 25 qualify, all of them are included.

## Recording
One row per included study in `extraction_form.csv`. Counts are reported exactly as found (plan Step 0.6). `NR` means not reported; nothing is inferred.

## Search log
- Search date: TODO (after this file is committed)
- Hits: TODO
- Records screened: TODO
- Included: TODO
