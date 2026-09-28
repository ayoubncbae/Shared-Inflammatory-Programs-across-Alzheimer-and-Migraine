# AD--migraine PBMC transcriptomics analysis

This folder is the reproducible Python package derived from
`notebookfa40ae7b07.ipynb` and `transcriptomics-add-migrain.ipynb`. It contains
the original figure workflow plus the reviewer-requested donor-level
pseudobulk differential-expression (DE) and overlap analysis.

## Data

Place these unmodified GEO archives in a directory named `datasets` beside
`Final_code`:

```text
ADD_migraine_Traanscriptomics/
|-- datasets/
|   |-- GSE181279_RAW.tar
|   `-- GSE269117_RAW.tar
`-- Final_code/
```

The archives are intentionally excluded from this GitHub folder because each
is larger than GitHub's normal file limit. The scripts read the archives
directly; no manual extraction is needed.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Fast reviewer analysis (about 30--60 seconds on the tested workstation):

```bash
python pseudobulk_overlap.py
```

Regenerate the four requested manuscript figures and reviewer analysis:

```bash
python run_all.py
```

The two figure scripts reproduce the notebook-reported manuscript panels from
the finalized values. The exact supplied reference images are also retained as
`Result_Figure_1.png` through `Result_Figure_4.png`.

Optional full single-cell workflow (slower and memory intensive):

```bash
python analysis_pipeline.py
```

To use a different data location:

```bash
python pseudobulk_overlap.py --data-dir /path/to/datasets
```

For the full pipeline, set `AD_MIGRAINE_DATA_DIR` to the archive directory.

## Reviewer analysis specification

1. Cells are filtered using 200--7,000 detected genes and <=25% mitochondrial
   reads, matching the base workflow.
2. Raw counts are summed by donor separately in GSE181279 and GSE269117.
3. HC20 is excluded automatically because zero cells pass the pre-specified QC
   filters; this reproduces the notebook's 10 migraine-study donors.
4. Donor log2(CPM+1) values are compared using Welch's t-test, followed by
   Benjamini--Hochberg correction within each dataset.
5. DEGs require FDR < 0.05 and absolute log2FC >= 0.25 in both datasets.
6. Significant-list overlap is tested with a one-sided hypergeometric test;
   concordant direction is reported separately. Top-100/300/500 independent
   ranked-list overlap is included as sensitivity analysis.

## Outputs

- `results/tables/AD_donor_pseudobulk_DE.csv`
- `results/tables/Migraine_donor_pseudobulk_DE.csv`
- `results/tables/Reviewer_DEG_overlap_summary.csv`
- `results/tables/Reviewer_topN_overlap_robustness.csv`
- `results/tables/Reviewer_significant_overlap_genes.csv`
- `results/figures/Figure5_Pseudobulk_DEG_Overlap.png` and PDF
- Four requested manuscript figures in `results/figures/`
- Full interpretation in `RESULTS_REPORT.md`

All inference is donor-level. Cells are not treated as independent replicates.
