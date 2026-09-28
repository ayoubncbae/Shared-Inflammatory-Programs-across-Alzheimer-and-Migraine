# AD--migraine PBMC transcriptomics analysis

Reproducible Python implementation of the analyses originally developed in
`notebookfa40ae7b07.ipynb` and `transcriptomics-add-migrain.ipynb`.

The workflow processes the GSE181279 and GSE269117 single-cell RNA-seq datasets,
performs quality control, normalization, HVG selection, PCA/UMAP, Leiden
clustering, cluster-marker analysis, donor-level pseudobulk differential
expression, shared-gene/pathway analysis, and manuscript figure/table
generation.

## Repository contents

- `analysis_pipeline.py` — main validated, start-to-finish analysis.
- `generate_additional_results.py` — generates the additional abundance table
  and Figure 3 after the main analysis.
- `01_notebookfa40ae7b07.py` — cell-by-cell Python export of the first notebook.
- `02_transcriptomics_add_migraine.py` — cell-by-cell Python export of the
  second notebook.
- `requirements.txt` — Python dependencies.
- `reference_results/` — the four supplied reference-result images.

The numbered exports retain the notebooks' full development history, including
earlier exploratory cells. Use `analysis_pipeline.py`, not the numbered exports,
for a clean reproducible run.

## System requirements

- Python 3.11 recommended
- Approximately 4 GB of available RAM or more
- Approximately 3 GB of free disk space
- Internet access if the raw GEO archives are not supplied manually

The complete workflow was validated on Windows using Python 3.11 and Scanpy
1.11.5. The main analysis took approximately 15 minutes on the validation
workstation.

## Installation

Clone the repository and enter its directory:

```bash
git clone <YOUR-REPOSITORY-URL>
cd AD_migraine_transcriptomics_GitHub
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Input data

Create a directory named `data` in the repository root:

```text
AD_migraine_transcriptomics_GitHub/
├── data/
│   ├── GSE181279_RAW.tar
│   └── GSE269117_RAW.tar
├── analysis_pipeline.py
└── requirements.txt
```

Download the unmodified GEO supplementary archives for accessions GSE181279
and GSE269117 and retain the exact filenames shown above. Alternatively, run
the main pipeline with an empty `data` directory; it will attempt to download
both archives from NCBI automatically.

Expected archive checksums:

| Archive | SHA-256 |
|---|---|
| `GSE181279_RAW.tar` | `58c82995fe1bf261e5f8a549ed31d482cb6355983b92dd71c5d3161e02ab59d3` |
| `GSE269117_RAW.tar` | `77e8341a188995c2aa287197b959c24f9f2178a620d43c9312077bf239625b97` |

The raw archives and extracted matrices are excluded from Git through
`.gitignore` because they exceed normal GitHub file-size limits.

## Run the analysis

Run all commands from the repository root.

First run the complete main analysis:

```bash
python analysis_pipeline.py
```

A successful run finishes with `DONE.` and creates:

```text
AD_Migraine_Final_Analysis/
├── figures/
├── objects/
├── tables/
├── AD_Migraine_Final_Result_Tables.xlsx
└── Auto_Result_Section_Interpretation.txt
```

It also creates `AD_Migraine_Final_Analysis_Package.zip`.

Next generate the additional abundance result:

```bash
python generate_additional_results.py
```

This adds Table IV and Figure 3 to the existing output directories.

## Validated checkpoint

The tested workflow completed with exit code 0 for both commands and produced:

- 36,772 retained GSE181279 cells
- 27,158 retained GSE269117 cells
- 63,930 retained cells in total
- 29,603 retained genes
- 5 AD-study donor profiles
- 10 migraine-study donor profiles
- 21 data-driven clusters

Key output files include:

- `figures/Picture1_Atlas_and_Shared_Gene_Convergence.png`
- `figures/Picture2_Shared_Pathway_and_Abundance_Contrast.png`
- `figures/Figure3_Cluster_Abundance_Contrast_600dpi.png`
- `tables/Table1_dataset_and_analysis_summary.csv`
- `tables/Table2_shared_AD_Migraine_genes.csv`
- `tables/Table3_shared_immune_pathway_enrichment.csv`
- `tables/Table4_top_cluster_abundance_shifts.csv`

Small differences in UMAP coordinates or figure rendering may occur across
operating systems, BLAS implementations, or major dependency changes. The
analysis random seed and validated dependency versions are defined in the code and
`requirements.txt`.

## Implementation notes

The standalone export makes only portability/runtime changes to the notebook
workflow:

- Kaggle-specific paths were replaced with repository-relative paths.
- Notebook package-install commands were moved to `requirements.txt`.
- The original cluster-mean assignment was vectorized without changing its
  assignment rule.
- Matplotlib 3.10 and Windows UTF-8 compatibility were added.

All biological thresholds and analysis steps remain those used by the final
notebook pipeline.
