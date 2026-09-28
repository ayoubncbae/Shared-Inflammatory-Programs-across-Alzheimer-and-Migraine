# Single-Cell Transcriptomics Identifies Candidate Shared Peripheral Inflammatory Programs in Alzheimer's Disease and Migraine" Accepted as Short Paper at IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2026).

 

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
  

Small differences in UMAP coordinates or figure rendering may occur across
operating systems, BLAS implementations, or major dependency changes. The
analysis random seed and validated dependency versions are defined in the code and
`requirements.txt`.

## Citation

If you use this code or results in your work, please cite the following paper:

**Single-Cell Transcriptomics Identifies Candidate Shared Peripheral Inflammatory Programs in Alzheimer's Disease and Migraine**

Muhammad Ayoub, Hai Zhao, and Hongming Shan  
Accepted as a short paper at the 2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2026).


