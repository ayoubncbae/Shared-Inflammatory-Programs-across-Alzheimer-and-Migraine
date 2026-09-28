# AD--migraine transcriptomics: reproducible Python export

This directory contains Python conversions of the two supplied notebooks. The
analysis logic and cell order were retained. Notebook-only package commands
were removed, Kaggle/Colab absolute paths were changed to repository-relative
paths, and one per-gene cluster-mapping loop was vectorized without changing its
cluster-mean assignment rule so the standalone workflow finishes practically.

## Files

- `01_notebookfa40ae7b07.py`: cell-by-cell export of
  `notebookfa40ae7b07.ipynb` (the earlier analysis history).
- `02_transcriptomics_add_migraine.py`: cell-by-cell export of
  `transcriptomics-add-migrain.ipynb` (the complete notebook history).
- `analysis_pipeline.py`: the final self-contained, corrected low-memory
  pipeline cell from the second notebook. **Run this file to reproduce the
  complete analysis.**
- `generate_additional_results.py`: direct export of the notebook's final
  additional immune-state abundance figure cell.
- `convert_notebooks.py`: regenerates all Python files from the original
  notebooks, which should remain one directory above this folder.
- `reference_results/`: copies of the four supplied target result images for
  visual comparison.

The two numbered exports include the notebook's exploratory attempts and cells
that originally raised errors before corrected cells were added. They are kept
for exact provenance and are not the recommended entry point. The clean
`analysis_pipeline.py` is directly extracted from the notebook's final
self-contained pipeline cell, rather than being a rewritten analysis.

## 1. Clone and create an environment

Python 3.11 is recommended. From this directory:

```bash
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

or:

```bash
# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Provide the raw GEO archives

Create `data` inside this directory and place the two unmodified archives in
it:

```text
final_code_2/
|-- data/
|   |-- GSE181279_RAW.tar
|   `-- GSE269117_RAW.tar
|-- analysis_pipeline.py
`-- requirements.txt
```

The archives are available from GEO accessions GSE181279 and GSE269117. If the
files are absent, `analysis_pipeline.py` attempts to download them from NCBI.
They are excluded by `.gitignore` because they are too large for ordinary
GitHub storage.

For the supplied local project, copy them with:

```powershell
New-Item -ItemType Directory -Path .\data -Force
Copy-Item ..\datasets\GSE181279_RAW.tar .\data\
Copy-Item ..\datasets\GSE269117_RAW.tar .\data\
```

## 3. Run the analysis

Always run from the `final_code_2` directory:

```bash
python analysis_pipeline.py
```

The pipeline performs raw-data extraction, QC, normalization, HVG selection,
PCA/UMAP, Leiden clustering, cluster-marker analysis, donor-level pseudobulk
DE, shared-gene/pathway analysis, and manuscript figure/table generation.

Main outputs are written to:

```text
AD_Migraine_Final_Analysis/
|-- figures/
|-- tables/
`-- objects/
```

The pipeline also produces `AD_Migraine_Final_Analysis_Package.zip`.

After the main pipeline completes, generate the notebook's additional result:

```bash
python generate_additional_results.py
```

## 4. Validated result checkpoint

The complete README workflow was run successfully on Windows with Python 3.11
and Scanpy 1.11.5 on 2026-09-27. The main pipeline finished in approximately 15
minutes on the validation workstation and the additional-results script took
under 10 seconds. The run retained 36,772 GSE181279 cells and 27,158 GSE269117
cells (63,930 total), retained 29,603 genes, used 5 AD-study donor profiles and
10 migraine-study donor profiles, and generated 21 data-driven clusters.

Successful completion prints `DONE.` and creates:

- `AD_Migraine_Final_Analysis/AD_Migraine_Final_Result_Tables.xlsx`
- `AD_Migraine_Final_Analysis/Auto_Result_Section_Interpretation.txt`
- `AD_Migraine_Final_Analysis/figures/Picture1_Atlas_and_Shared_Gene_Convergence.*`
- `AD_Migraine_Final_Analysis/figures/Picture2_Shared_Pathway_and_Abundance_Contrast.*`
- `AD_Migraine_Final_Analysis/figures/Figure3_Cluster_Abundance_Contrast_600dpi.*`
- `AD_Migraine_Final_Analysis/tables/Table1_dataset_and_analysis_summary.csv`
- `AD_Migraine_Final_Analysis/tables/Table2_shared_AD_Migraine_genes.csv`
- `AD_Migraine_Final_Analysis/tables/Table3_shared_immune_pathway_enrichment.csv`
- `AD_Migraine_Final_Analysis/tables/Table4_top_cluster_abundance_shifts.csv`
- `AD_Migraine_Final_Analysis_Package.zip`

Small numeric or UMAP-layout differences can occur if major dependency versions,
BLAS implementations, or operating systems differ; the random seed and
dependency ranges are specified by the exported code and `requirements.txt`.

## Regenerate the exports

If the original notebooks are present one directory above `final_code_2`:

```bash
python convert_notebooks.py
```

Then confirm syntax with:

```bash
python -m py_compile 01_notebookfa40ae7b07.py 02_transcriptomics_add_migraine.py analysis_pipeline.py generate_additional_results.py
```
