#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct Python export of notebookfa40ae7b07.ipynb."""

# %% [notebook cell 1]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (./) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %% [notebook cell 2]
# ============================================================
# CELL 1 — Install packages
# ============================================================
# Kaggle: turn Internet ON in Notebook settings before running.
# Notebook dependency command removed; use requirements.txt

# %% [notebook cell 3]
# ============================================================
# CELL 2 — Imports and global settings
# ============================================================
import os
import re
import gzip
import json
import tarfile
import shutil
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import scipy.io as sio
from scipy import sparse, stats
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

import scanpy as sc
import anndata as ad

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Publication-style defaults
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=120, facecolor="white", fontsize=10)

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sns.set_theme(style="whitegrid", context="paper")

DATA_DIR = Path("data")
OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
OBJ_DIR = OUT_DIR / "objects"

for d in [DATA_DIR, OUT_DIR, FIG_DIR, TAB_DIR, OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Scanpy:", sc.__version__)
print("Working directory:", OUT_DIR)

# %% [notebook cell 4]
# ============================================================
# CELL 3 — Dataset configuration
# ============================================================
# Direct GEO supplementary downloads.
# These are the RAW supplementary archives shown on GEO.
GEO_URLS = {
    "GSE181279_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE181279&format=file&file=GSE181279_RAW.tar",
    "GSE269117_RAW.tar": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE269117&format=file&file=GSE269117_RAW.tar",
}

# Analysis thresholds. You can make them stricter later.
MIN_GENES_PER_CELL = 200
MAX_MT_PERCENT = 20
MIN_CELLS_PER_GENE = 3
MIN_CELLS_PER_SAMPLE_CELLTYPE = 20
MIN_SAMPLES_PER_GROUP = 2
LOGFC_CUTOFF = 0.25
PADJ_CUTOFF = 0.20   # relaxed because AD has only 3 vs 2 donors

# Main comparison labels
AD_DISEASE_LABEL = "AD"
AD_CONTROL_LABEL = "Control"
MI_DISEASE_LABEL = "Migraine"
MI_CONTROL_LABEL = "Control"

print("Configured datasets:")
for k, v in GEO_URLS.items():
    print("-", k, "<-", v)

# %% [notebook cell 5]
# ============================================================
# CELL 4 — Download and extract GEO supplementary files
# ============================================================
def download_file(url, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"Already exists: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
        return
    print(f"Downloading {dst.name} ...")
    cmd = ["wget", "-q", "--show-progress", "-O", str(dst), url]
    subprocess.run(cmd, check=True)
    if not dst.exists() or dst.stat().st_size < 1000:
        raise RuntimeError(f"Download failed or file is too small: {dst}")
    print(f"Downloaded: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")


def safe_extract_tar(tar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"Already extracted: {tar_path.name}")
        return
    print(f"Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r") as tar:
        def is_safe(member):
            target = out_dir / member.name
            return str(target.resolve()).startswith(str(out_dir.resolve()))
        members = tar.getmembers()
        unsafe = [m.name for m in members if not is_safe(m)]
        if unsafe:
            raise RuntimeError(f"Unsafe paths in tar: {unsafe[:5]}")
        tar.extractall(out_dir)
    marker.write_text("ok")
    print(f"Extracted to: {out_dir}")

for fname, url in GEO_URLS.items():
    tar_path = DATA_DIR / fname
    download_file(url, tar_path)
    safe_extract_tar(tar_path, DATA_DIR / fname.replace(".tar", ""))

print("\nExtracted file examples:")
for root in [DATA_DIR / "GSE181279_RAW", DATA_DIR / "GSE269117_RAW"]:
    files = sorted([p for p in root.rglob("*") if p.is_file()])[:10]
    print("\n", root.name)
    for p in files:
        print(" ", p.relative_to(root))

# %% [notebook cell 6]
from pathlib import Path
import requests
import tarfile
import time


def resolve_geo_tar_url(geo_id: str):
    """
    Try correct GEO FTP location (this is the REAL stable source)
    """
    base = geo_id.replace("GSE", "")
    
    # GEO FTP pattern
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{base[:3]}nnn/{geo_id}/suppl/{geo_id}_RAW.tar"
    return url

# %% [notebook cell 7]
def download_file(url, dst: Path, retries=3):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"Already exists: {dst.name}")
        return

    headers = {"User-Agent": "Mozilla/5.0"}

    for i in range(retries):
        try:
            print(f"Downloading {dst.name} (try {i+1})")

            r = requests.get(url, stream=True, timeout=60, headers=headers)
            
            if r.status_code == 404:
                raise FileNotFoundError(f"URL not found: {url}")

            r.raise_for_status()

            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            if dst.stat().st_size < 1000:
                raise RuntimeError("Downloaded file too small")

            print(f"Downloaded: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
            return

        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(2)

    raise RuntimeError(f"Failed: {url}")

# %% [notebook cell 8]
for fname in GEO_URLS.keys():

    geo_id = fname.replace(".tar", "")  # GSE181279_RAW.tar → GSE181279_RAW (fix below)
    
    # IMPORTANT FIX: correct GEO folder name
    geo_id = geo_id.replace("_RAW", "")  # GSE181279_RAW → GSE181279

    url = resolve_geo_tar_url(geo_id)

    tar_path = DATA_DIR / f"{geo_id}_RAW.tar"

    print("Using URL:", url)

    download_file(url, tar_path)

# %% [notebook cell 9]
# ============================================================
# CELL 1 — Install/import packages and define folders
# ============================================================

import os, re, sys, tarfile, shutil, gzip, warnings, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")

# If these are not already installed in Kaggle, install them.
def install_if_missing(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except Exception:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg, imp in [
    ("scanpy", "scanpy"),
    ("anndata", "anndata"),
    ("leidenalg", "leidenalg"),
    ("igraph", "igraph"),
    ("openpyxl", "openpyxl"),
    ("statsmodels", "statsmodels"),
    ("adjustText", "adjustText"),
]:
    install_if_missing(pkg, imp)

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import sparse, stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

# ----------------------------
# Folders
# ----------------------------
try:
    DATA_DIR
except NameError:
    DATA_DIR = Path("data")

DATA_DIR = Path(DATA_DIR)
EXTRACT_DIR = DATA_DIR / "extracted"
CACHE_DIR = DATA_DIR / "tenx_cache"
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

for d in [EXTRACT_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

AD_TAR = DATA_DIR / "GSE181279_RAW.tar"
MIG_TAR = DATA_DIR / "GSE269117_RAW.tar"

print("AD tar exists:", AD_TAR.exists(), AD_TAR)
print("Migraine tar exists:", MIG_TAR.exists(), MIG_TAR)

# ----------------------------
# Plot style
# ----------------------------
sc.settings.verbosity = 2
sc.settings.figdir = str(FIG_DIR)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# %% [notebook cell 10]
# ============================================================
# CELL 2 — Safe extraction of GEO tar files
# ============================================================

def safe_extract_tar(tar_path, out_dir):
    """
    Safely extract tar file only once.
    """
    tar_path = Path(tar_path)
    out_dir = Path(out_dir)
    marker = out_dir / f".extracted_{tar_path.stem}"

    if marker.exists():
        print(f"Already extracted: {tar_path.name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path.name} ...")

    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = os.path.join(out_dir, member.name)
            if not is_within_directory(out_dir, member_path):
                raise Exception("Unsafe tar path detected.")
        tar.extractall(out_dir)

    marker.write_text("done")
    print(f"Extracted to: {out_dir}")

safe_extract_tar(AD_TAR, EXTRACT_DIR / "GSE181279")
safe_extract_tar(MIG_TAR, EXTRACT_DIR / "GSE269117")

print("\nTop extracted AD files:")
for p in list((EXTRACT_DIR / "GSE181279").rglob("*"))[:20]:
    print(p)

print("\nTop extracted Migraine files:")
for p in list((EXTRACT_DIR / "GSE269117").rglob("*"))[:20]:
    print(p)

# %% [notebook cell 11]
# ============================================================
# CELL 3 — File detection helpers for 10x MTX and 10x H5 files
# ============================================================

def clean_sample_name(name):
    name = Path(str(name)).name
    name = re.sub(r"\.gz$", "", name)
    name = re.sub(r"\.h5$", "", name)
    name = re.sub(r"\.mtx$", "", name)
    name = re.sub(r"_matrix$", "", name)
    name = re.sub(r"_barcodes$", "", name)
    name = re.sub(r"_features$", "", name)
    name = re.sub(r"_genes$", "", name)
    name = re.sub(r"filtered_feature_bc_matrix", "", name, flags=re.I)
    name = re.sub(r"raw_feature_bc_matrix", "", name, flags=re.I)
    name = re.sub(r"[_\-.]+$", "", name)
    return name

def infer_ad_condition(sample):
    s = sample.upper()
    if re.search(r"(^|[_\-.])AD[0-9]+", s) or "_AD" in s:
        return "AD"
    if re.search(r"(^|[_\-.])(NC|CTRL|CONTROL|HC)[0-9]*", s) or "_NC" in s:
        return "Control"
    return None

def infer_migraine_condition(sample):
    s = sample.upper()
    # Keep only true migraine MI and healthy controls HC.
    # Exclude VM and MD.
    if re.search(r"(^|[_\-.])MI[0-9]+", s) or "_MI" in s or "MIGRAINE" in s:
        if "VM" not in s and "VESTIBULAR" not in s:
            return "Migraine"
    if re.search(r"(^|[_\-.])HC[0-9]+", s) or "_HC" in s or "HEALTHY" in s or "CONTROL" in s:
        return "Control"
    return None

def is_expression_file(path):
    s = str(path).upper()
    bad = ["ATAC", "PEAK", "FRAGMENT", "BCR", "TCR", "VDJ"]
    return not any(x in s for x in bad)

def discover_h5_files(root):
    root = Path(root)
    h5s = sorted([p for p in root.rglob("*.h5") if is_expression_file(p)])
    return h5s

def discover_mtx_triples(root):
    """
    Finds 10x matrix/barcodes/features triples.
    Handles both:
    1) folder/matrix.mtx.gz, folder/barcodes.tsv.gz, folder/features.tsv.gz
    2) folder/GSMxxx_sample_matrix.mtx.gz, folder/GSMxxx_sample_barcodes.tsv.gz, etc.
    """
    root = Path(root)
    matrices = sorted(list(root.rglob("*matrix.mtx*")))

    triples = []
    for matrix in matrices:
        parent = matrix.parent
        name = matrix.name

        # Case A: standard 10x names
        if name.startswith("matrix.mtx"):
            candidates_bar = list(parent.glob("barcodes.tsv*"))
            candidates_feat = list(parent.glob("features.tsv*")) + list(parent.glob("genes.tsv*"))
            if candidates_bar and candidates_feat:
                triples.append({
                    "matrix": matrix,
                    "barcodes": candidates_bar[0],
                    "features": candidates_feat[0],
                    "sample": clean_sample_name(parent.name),
                })
            continue

        # Case B: prefixed files
        prefix = re.sub(r"matrix\.mtx(\.gz)?$", "", name)
        bar = parent / f"{prefix}barcodes.tsv.gz"
        if not bar.exists():
            bar = parent / f"{prefix}barcodes.tsv"

        feat = parent / f"{prefix}features.tsv.gz"
        if not feat.exists():
            feat = parent / f"{prefix}features.tsv"
        if not feat.exists():
            feat = parent / f"{prefix}genes.tsv.gz"
        if not feat.exists():
            feat = parent / f"{prefix}genes.tsv"

        if bar.exists() and feat.exists():
            triples.append({
                "matrix": matrix,
                "barcodes": bar,
                "features": feat,
                "sample": clean_sample_name(prefix),
            })

    return triples

def make_standard_10x_dir(triple, cache_root):
    """
    Create a standard 10x directory with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz.
    Uses symlinks when possible.
    """
    sample = clean_sample_name(triple["sample"])
    out = Path(cache_root) / sample
    out.mkdir(parents=True, exist_ok=True)

    links = {
        "matrix.mtx.gz": triple["matrix"],
        "barcodes.tsv.gz": triple["barcodes"],
        "features.tsv.gz": triple["features"],
    }

    for dst_name, src in links.items():
        dst = out / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    return out

def read_10x_h5_sample(path, sample_id, dataset, condition):
    print(f"Reading H5: {sample_id} | {dataset} | {condition}")
    x = sc.read_10x_h5(str(path), gex_only=True)
    x.var_names_make_unique()
    x.obs_names = [f"{sample_id}_{bc}" for bc in x.obs_names]
    x.obs["sample_id"] = sample_id
    x.obs["dataset"] = dataset
    x.obs["condition"] = condition
    return x

def read_10x_mtx_sample(triple, sample_id, dataset, condition):
    print(f"Reading MTX: {sample_id} | {dataset} | {condition}")
    tenx_dir = make_standard_10x_dir(triple, CACHE_DIR / dataset)
    x = sc.read_10x_mtx(str(tenx_dir), var_names="gene_symbols", cache=False)
    x.var_names_make_unique()
    x.obs_names = [f"{sample_id}_{bc}" for bc in x.obs_names]
    x.obs["sample_id"] = sample_id
    x.obs["dataset"] = dataset
    x.obs["condition"] = condition
    return x

# %% [notebook cell 12]
# ============================================================
# CELL 4 — Load AD and Migraine samples
# ============================================================

AD_ROOT = EXTRACT_DIR / "GSE181279"
MIG_ROOT = EXTRACT_DIR / "GSE269117"

# ----------------------------
# AD: GSE181279
# ----------------------------
ad_objects = []

ad_h5s = discover_h5_files(AD_ROOT)
ad_mtxs = discover_mtx_triples(AD_ROOT)

print(f"AD H5 files detected: {len(ad_h5s)}")
print(f"AD MTX triples detected: {len(ad_mtxs)}")

if len(ad_h5s) > 0:
    for p in ad_h5s:
        sample = clean_sample_name(p.name)
        if not is_expression_file(p):
            continue
        cond = infer_ad_condition(sample)
        if cond is None:
            print("Skipping AD file because condition not inferred:", p.name)
            continue
        ad_objects.append(read_10x_h5_sample(p, sample, "AD_GSE181279", cond))
else:
    for tr in ad_mtxs:
        sample = clean_sample_name(tr["sample"])
        if not is_expression_file(sample):
            continue
        cond = infer_ad_condition(sample)
        if cond is None:
            print("Skipping AD MTX because condition not inferred:", sample)
            continue
        ad_objects.append(read_10x_mtx_sample(tr, sample, "AD_GSE181279", cond))

print("Loaded AD objects:", len(ad_objects))
assert len(ad_objects) > 0, "No AD expression samples loaded. Check extracted file names."

# ----------------------------
# Migraine: GSE269117
# ----------------------------
mig_objects = []

mig_h5s = discover_h5_files(MIG_ROOT)
mig_mtxs = discover_mtx_triples(MIG_ROOT)

print(f"Migraine H5 files detected: {len(mig_h5s)}")
print(f"Migraine MTX triples detected: {len(mig_mtxs)}")

if len(mig_h5s) > 0:
    for p in mig_h5s:
        sample = clean_sample_name(p.name)
        if not is_expression_file(p):
            continue
        cond = infer_migraine_condition(sample)
        if cond is None:
            print("Skipping non-MI/HC Migraine cohort sample:", p.name)
            continue
        mig_objects.append(read_10x_h5_sample(p, sample, "Migraine_GSE269117", cond))
else:
    for tr in mig_mtxs:
        sample = clean_sample_name(tr["sample"])
        if not is_expression_file(sample):
            continue
        cond = infer_migraine_condition(sample)
        if cond is None:
            print("Skipping non-MI/HC Migraine cohort sample:", sample)
            continue
        mig_objects.append(read_10x_mtx_sample(tr, sample, "Migraine_GSE269117", cond))

print("Loaded Migraine objects:", len(mig_objects))
assert len(mig_objects) > 0, "No Migraine expression samples loaded. Check extracted file names."

# ----------------------------
# Combine
# ----------------------------
adata = sc.concat(
    ad_objects + mig_objects,
    join="outer",
    label="source_object",
    keys=[x.obs["sample_id"][0] for x in ad_objects + mig_objects],
    index_unique=None,
    fill_value=0
)

adata.var_names_make_unique()

print(adata)
print("\nSample summary:")
display(adata.obs[["dataset", "sample_id", "condition"]].drop_duplicates().sort_values(["dataset", "condition", "sample_id"]))

sample_table = (
    adata.obs[["dataset", "sample_id", "condition"]]
    .drop_duplicates()
    .sort_values(["dataset", "condition", "sample_id"])
)
sample_table.to_csv(TABLE_DIR / "Table1_sample_metadata.csv", index=False)

# %% [notebook cell 13]


# %% [notebook cell 14]
# Notebook dependency command removed; use requirements.txt

# %% [notebook cell 15]


# %% [notebook cell 16]
# ============================================================
# CELL 5 — QC, filtering, normalization, clustering, UMAP
# ============================================================

# Preserve raw counts before normalization
adata.layers["counts"] = adata.X.copy()

# Mito genes
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

print("Before QC:", adata.shape)

# Defensive filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=5)

# Additional cell-level filters
adata = adata[
    (adata.obs["n_genes_by_counts"] >= 200) &
    (adata.obs["n_genes_by_counts"] <= 7000) &
    (adata.obs["pct_counts_mt"] <= 25)
].copy()

print("After QC:", adata.shape)

# Re-save counts after filtering
adata.layers["counts"] = adata.layers["counts"].copy()

# Normalize/log
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# HVG, PCA, neighbors, UMAP, Leiden
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor="seurat_v3",
    batch_key="sample_id",
    subset=False
)

adata.raw = adata.copy()

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=40)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_hvg, min_dist=0.35)
sc.tl.leiden(adata_hvg, resolution=0.7, key_added="cluster")

# Transfer embedding and clusters back
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
adata.obs["cluster"] = adata_hvg.obs["cluster"].astype(str).values

print(adata)
adata.write_h5ad(RESULTS_DIR / "AD_Migraine_PBMC_processed_pre_annotation.h5ad")

# %% [notebook cell 17]
# ============================================================
# NEW CELL 5 — QC first, then remove housekeeping genes
# before normalization, HVG, PCA, clustering, DE
# ============================================================

import re
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

print("Starting object:", adata)

# ------------------------------------------------------------
# 1. Preserve raw counts before any processing
# ------------------------------------------------------------
adata.layers["raw_counts_before_gene_filter"] = adata.X.copy()

# ------------------------------------------------------------
# 2. Technical QC genes
# This is NOT biological analysis. We only calculate QC metrics.
# ------------------------------------------------------------
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True
)

print("Before cell QC:", adata.shape)

# ------------------------------------------------------------
# 3. Cell-level QC
# ------------------------------------------------------------
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=5)

adata = adata[
    (adata.obs["n_genes_by_counts"] >= 200) &
    (adata.obs["n_genes_by_counts"] <= 7000) &
    (adata.obs["pct_counts_mt"] <= 25)
].copy()

print("After cell QC:", adata.shape)

# ------------------------------------------------------------
# 4. Define known housekeeping / technical confound genes
# ------------------------------------------------------------
# Conservative known housekeeping genes commonly used as loading controls.
known_housekeeping = {
    "ACTB", "GAPDH", "B2M", "HPRT1", "TBP", "PPIA", "PGK1", "RPLP0",
    "GUSB", "YWHAZ", "TUBB", "TUBA1A", "TUBA1B", "HMBS", "SDHA",
    "LDHA", "NONO", "RPS18", "RPS13", "RPL13A", "EEF1A1", "EEF2",
    "POLR2A", "TFRC", "UBC", "UBB", "VCP", "CANX", "PUM1", "IPO8",
    "ALAS1", "ARBP", "ATP5F1", "CYC1", "TOP1", "RER1", "OAZ1",
    "RAN", "PSMB2", "PSMB4", "PSMB6", "PSMB7", "NACA", "CALM1",
    "CALM2", "CALM3", "HSP90AB1", "HSP90AA1", "HSPA8", "HSPD1",
    "MALAT1"
}

# Broad technical/ribosomal/mitochondrial patterns.
# These often dominate single-cell PCs but are not disease-specific immune biology.
housekeeping_patterns = [
    r"^MT-",          # mitochondrial genes
    r"^RPL\d",        # ribosomal large subunit
    r"^RPLP",         # ribosomal protein lateral stalk
    r"^RPS\d",        # ribosomal small subunit
    r"^MRPL\d",       # mitochondrial ribosomal large subunit
    r"^MRPS\d",       # mitochondrial ribosomal small subunit
    r"^HIST\d",       # histone clusters
    r"^HSP90",        # heat shock housekeeping/stress loading genes
]

def is_housekeeping_gene(gene):
    g = str(gene).upper()
    if g in known_housekeeping:
        return True
    for pat in housekeeping_patterns:
        if re.match(pat, g):
            return True
    return False

adata.var["remove_housekeeping"] = [is_housekeeping_gene(g) for g in adata.var_names]

removed_hkg = adata.var[adata.var["remove_housekeeping"]].copy()
kept_genes = adata.var[~adata.var["remove_housekeeping"]].copy()

print("Housekeeping/technical genes removed:", removed_hkg.shape[0])
print("Genes kept for biological analysis:", kept_genes.shape[0])

removed_hkg[["remove_housekeeping"]].to_csv(
    TABLE_DIR / "Removed_housekeeping_technical_genes.csv"
)

# ------------------------------------------------------------
# 5. Remove housekeeping genes before all downstream analysis
# ------------------------------------------------------------
adata = adata[:, ~adata.var["remove_housekeeping"]].copy()

# Preserve biological-count layer after gene filtering
adata.layers["counts"] = adata.X.copy()

print("After removing housekeeping genes:", adata.shape)

# ------------------------------------------------------------
# 6. Normalize, log-transform, HVG, PCA, neighbors, UMAP, Leiden
# ------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Use data-driven HVGs after housekeeping removal
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor="seurat",
    batch_key="sample_id",
    subset=False
)

adata.raw = adata.copy()

adata_hvg = adata[:, adata.var["highly_variable"]].copy()

sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=40)

sc.pp.neighbors(
    adata_hvg,
    n_neighbors=15,
    n_pcs=30
)

sc.tl.umap(
    adata_hvg,
    min_dist=0.35
)

sc.tl.leiden(
    adata_hvg,
    resolution=0.7,
    key_added="data_driven_cluster"
)

# Transfer results back to full non-housekeeping gene object
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
adata.obs["data_driven_cluster"] = adata_hvg.obs["data_driven_cluster"].astype(str).values

print("Final analysis object:", adata)
print("Clusters:")
display(adata.obs["data_driven_cluster"].value_counts().sort_index())

adata.write_h5ad(
    RESULTS_DIR / "AD_Migraine_PBMC_after_housekeeping_removed_clustered.h5ad"
)

# %% [notebook cell 18]


# %% [notebook cell 19]
# ============================================================
# NEW CELL 6 — Data-driven cluster marker discovery
# No manual marker-gene annotation
# ============================================================

import numpy as np
import pandas as pd
import scanpy as sc

# ------------------------------------------------------------
# 1. Find marker genes for each unsupervised cluster
# ------------------------------------------------------------
sc.tl.rank_genes_groups(
    adata,
    groupby="data_driven_cluster",
    method="wilcoxon",
    use_raw=False,
    pts=True,
    key_added="cluster_markers"
)

cluster_marker_df = sc.get.rank_genes_groups_df(
    adata,
    group=None,
    key="cluster_markers"
)

cluster_marker_df = cluster_marker_df.rename(
    columns={
        "group": "cluster",
        "names": "gene",
        "scores": "score",
        "logfoldchanges": "logFC",
        "pvals": "pval",
        "pvals_adj": "padj"
    }
)

cluster_marker_df = cluster_marker_df.sort_values(
    ["cluster", "padj", "logFC"],
    ascending=[True, True, False]
)

cluster_marker_df.to_csv(
    TABLE_DIR / "Table2A_data_driven_cluster_marker_genes.csv",
    index=False
)

print("Top data-driven cluster markers:")
display(cluster_marker_df.head(30))

# ------------------------------------------------------------
# 2. Create data-driven cluster labels from top enriched genes
# ------------------------------------------------------------
def make_cluster_signature(df, cluster_id, top_n=5):
    sub = df[
        (df["cluster"].astype(str) == str(cluster_id)) &
        (df["padj"] < 0.05) &
        (df["logFC"] > 0)
    ].copy()

    if len(sub) == 0:
        sub = df[df["cluster"].astype(str) == str(cluster_id)].copy()

    genes = (
        sub.sort_values(["padj", "logFC"], ascending=[True, False])
        .head(top_n)["gene"]
        .astype(str)
        .tolist()
    )

    if len(genes) == 0:
        return f"Cluster_{cluster_id}"

    return f"C{cluster_id}: " + "/".join(genes)

cluster_ids = sorted(
    adata.obs["data_driven_cluster"].astype(str).unique(),
    key=lambda x: int(x) if str(x).isdigit() else str(x)
)

cluster_signature_map = {
    cid: make_cluster_signature(cluster_marker_df, cid, top_n=5)
    for cid in cluster_ids
}

adata.obs["cell_type"] = adata.obs["data_driven_cluster"].map(cluster_signature_map).astype(str)

# We keep column name "cell_type" only for compatibility with downstream code.
# It now means DATA-DRIVEN TRANSCRIPTOMIC CLUSTER, not manual PBMC cell type.
adata.obs["analysis_cluster"] = adata.obs["cell_type"]

cluster_signature_table = pd.DataFrame({
    "cluster": list(cluster_signature_map.keys()),
    "data_driven_cluster_label": list(cluster_signature_map.values())
})

cluster_signature_table.to_csv(
    TABLE_DIR / "Table2B_data_driven_cluster_signatures.csv",
    index=False
)

print("Data-driven cluster signatures:")
display(cluster_signature_table)

# ------------------------------------------------------------
# 3. Count cells per data-driven cluster
# ------------------------------------------------------------
celltype_counts = (
    adata.obs.groupby(["dataset", "condition", "sample_id", "cell_type"])
    .size()
    .reset_index(name="n_cells")
)

celltype_counts.to_csv(
    TABLE_DIR / "Table2C_data_driven_cluster_counts.csv",
    index=False
)

display(celltype_counts.head())

adata.write_h5ad(
    RESULTS_DIR / "AD_Migraine_PBMC_data_driven_clusters_annotated.h5ad"
)

# %% [notebook cell 20]


# %% [notebook cell 21]
# ============================================================
# REPLACEMENT FIGURE 5 — Data-driven cluster marker plot
# ============================================================

import matplotlib.pyplot as plt
import scanpy as sc

# Select top data-driven marker genes per cluster
top_marker_genes = []

for cid in cluster_ids:
    sub = cluster_marker_df[
        (cluster_marker_df["cluster"].astype(str) == str(cid)) &
        (cluster_marker_df["padj"] < 0.05) &
        (cluster_marker_df["logFC"] > 0)
    ].copy()

    if len(sub) == 0:
        sub = cluster_marker_df[
            cluster_marker_df["cluster"].astype(str) == str(cid)
        ].copy()

    genes = (
        sub.sort_values(["padj", "logFC"], ascending=[True, False])
        .head(3)["gene"]
        .astype(str)
        .tolist()
    )

    top_marker_genes.extend(genes)

top_marker_genes = list(dict.fromkeys([g for g in top_marker_genes if g in adata.var_names]))

print("Top data-driven marker genes used for dotplot:")
print(top_marker_genes)

# ------------------------------------------------------------
# Figure 5A: UMAP by data-driven clusters
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata,
    color="cell_type",
    ax=ax,
    show=False,
    frameon=False,
    size=6,
    legend_loc="right margin"
)

ax.set_title("A. Data-driven immune transcriptional clusters")

fig.tight_layout()
fig.savefig(FIG_DIR / "Figure5A_Data_driven_cluster_UMAP.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "Figure5A_Data_driven_cluster_UMAP.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Figure 5B: Dotplot of discovered cluster markers
# ------------------------------------------------------------
dp = sc.pl.dotplot(
    adata,
    var_names=top_marker_genes,
    groupby="cell_type",
    standard_scale="var",
    show=False,
    return_fig=True
)

dp.savefig(FIG_DIR / "Figure5B_Data_driven_cluster_marker_dotplot.png", dpi=600, bbox_inches="tight")
dp.savefig(FIG_DIR / "Figure5B_Data_driven_cluster_marker_dotplot.pdf", bbox_inches="tight")

# %% [notebook cell 22]


# %% [notebook cell 23]
# ============================================================
# CELL 7 — Final proof-analysis setup
# Run this after the new data-driven CELL 6
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import sparse, stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
assert "data_driven_cluster" in adata.obs.columns, "Run data-driven clustering first."
assert "condition" in adata.obs.columns, "condition column missing."
assert "dataset" in adata.obs.columns, "dataset column missing."
assert "sample_id" in adata.obs.columns, "sample_id column missing."
assert "counts" in adata.layers.keys(), "counts layer missing. Use raw count layer before normalization."

# ------------------------------------------------------------
# Short clean cluster labels
# ------------------------------------------------------------
adata.obs["cluster_short"] = "C" + adata.obs["data_driven_cluster"].astype(str)

# Keep data-driven biological signature separately
if "cell_type" in adata.obs.columns:
    adata.obs["cluster_signature"] = adata.obs["cell_type"].astype(str)
else:
    adata.obs["cluster_signature"] = adata.obs["cluster_short"].astype(str)

cluster_signature_map = (
    adata.obs[["cluster_short", "cluster_signature"]]
    .drop_duplicates()
    .sort_values("cluster_short")
)

cluster_signature_map.to_csv(
    TABLE_DIR / "Final_cluster_signature_map.csv",
    index=False
)

display(cluster_signature_map)

# ------------------------------------------------------------
# Cleaner figure saving
# ------------------------------------------------------------
def save_final_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Useful group order
# ------------------------------------------------------------
adata.obs["disease_group"] = adata.obs["dataset"].astype(str) + "_" + adata.obs["condition"].astype(str)

group_order = [
    "AD_GSE181279_Control",
    "AD_GSE181279_AD",
    "Migraine_GSE269117_Control",
    "Migraine_GSE269117_Migraine"
]

group_label_map = {
    "AD_GSE181279_Control": "AD-control",
    "AD_GSE181279_AD": "AD",
    "Migraine_GSE269117_Control": "Migraine-control",
    "Migraine_GSE269117_Migraine": "Migraine"
}

print("Available disease groups:")
display(adata.obs["disease_group"].value_counts())

# %% [notebook cell 24]
# ============================================================
# CELL 8 — Final Table 1: disease-associated cluster shifts
# ============================================================

# ------------------------------------------------------------
# Cell counts per sample and cluster
# ------------------------------------------------------------
sample_meta = (
    adata.obs[["dataset", "condition", "sample_id", "disease_group"]]
    .drop_duplicates()
    .sort_values(["dataset", "condition", "sample_id"])
)

clusters = sorted(
    adata.obs["cluster_short"].unique(),
    key=lambda x: int(x.replace("C", "")) if x.replace("C", "").isdigit() else x
)

raw_counts = (
    adata.obs.groupby(
        ["dataset", "condition", "sample_id", "disease_group", "cluster_short"],
        observed=True
    )
    .size()
    .reset_index(name="n_cells")
)

# Fill missing sample-cluster combinations with zero
full_index = (
    sample_meta.assign(key=1)
    .merge(pd.DataFrame({"cluster_short": clusters, "key": 1}), on="key")
    .drop(columns="key")
)

cluster_counts = full_index.merge(
    raw_counts,
    on=["dataset", "condition", "sample_id", "disease_group", "cluster_short"],
    how="left"
)

cluster_counts["n_cells"] = cluster_counts["n_cells"].fillna(0).astype(int)

sample_totals = (
    cluster_counts.groupby(["dataset", "condition", "sample_id", "disease_group"], observed=True)["n_cells"]
    .sum()
    .reset_index(name="total_cells")
)

cluster_counts = cluster_counts.merge(
    sample_totals,
    on=["dataset", "condition", "sample_id", "disease_group"],
    how="left"
)

cluster_counts["fraction"] = cluster_counts["n_cells"] / cluster_counts["total_cells"].replace(0, np.nan)

cluster_counts.to_csv(
    TABLE_DIR / "Final_cluster_fraction_per_sample.csv",
    index=False
)

# ------------------------------------------------------------
# Disease-control shift statistics
# ------------------------------------------------------------
def compute_cluster_shift(df, dataset, case_label, control_label="Control"):
    rows = []
    sub = df[df["dataset"] == dataset].copy()

    for cl in clusters:
        temp = sub[sub["cluster_short"] == cl].copy()

        case_vals = temp.loc[temp["condition"] == case_label, "fraction"].dropna().values
        ctrl_vals = temp.loc[temp["condition"] == control_label, "fraction"].dropna().values

        if len(case_vals) >= 2 and len(ctrl_vals) >= 2:
            pval = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False).pvalue
        else:
            pval = np.nan

        mean_case = np.mean(case_vals) if len(case_vals) else np.nan
        mean_ctrl = np.mean(ctrl_vals) if len(ctrl_vals) else np.nan
        delta = mean_case - mean_ctrl

        # Effect size: Cohen-like standardized mean difference
        pooled_sd = np.sqrt((np.var(case_vals, ddof=1) + np.var(ctrl_vals, ddof=1)) / 2) if len(case_vals) > 1 and len(ctrl_vals) > 1 else np.nan
        effect_size = delta / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

        rows.append({
            "dataset": dataset,
            "case_label": case_label,
            "cluster_short": cl,
            "mean_case_fraction": mean_case,
            "mean_control_fraction": mean_ctrl,
            "delta_fraction": delta,
            "abs_delta_fraction": abs(delta) if pd.notna(delta) else np.nan,
            "effect_size": effect_size,
            "pval": pval,
            "n_case_samples": len(case_vals),
            "n_control_samples": len(ctrl_vals)
        })

    out = pd.DataFrame(rows)
    out["padj"] = multipletests(out["pval"].fillna(1.0), method="fdr_bh")[1]
    return out

ad_shift = compute_cluster_shift(
    cluster_counts,
    dataset="AD_GSE181279",
    case_label="AD"
)

mig_shift = compute_cluster_shift(
    cluster_counts,
    dataset="Migraine_GSE269117",
    case_label="Migraine"
)

# Merge AD and migraine shifts
shift_merge = ad_shift.merge(
    mig_shift,
    on="cluster_short",
    suffixes=("_AD", "_Migraine")
)

shift_merge["same_direction"] = (
    np.sign(shift_merge["delta_fraction_AD"]) ==
    np.sign(shift_merge["delta_fraction_Migraine"])
)

shift_merge["concordance_class"] = np.where(
    (shift_merge["same_direction"]) & (shift_merge["delta_fraction_AD"] > 0),
    "Shared expansion",
    np.where(
        (shift_merge["same_direction"]) & (shift_merge["delta_fraction_AD"] < 0),
        "Shared reduction",
        "Divergent shift"
    )
)

shift_merge["composition_convergence_score"] = (
    np.abs(shift_merge["delta_fraction_AD"]) *
    np.abs(shift_merge["delta_fraction_Migraine"])
)

shift_merge = shift_merge.merge(
    cluster_signature_map,
    on="cluster_short",
    how="left"
)

shift_merge = shift_merge.sort_values(
    ["same_direction", "composition_convergence_score"],
    ascending=[False, False]
)

shift_merge.to_csv(
    TABLE_DIR / "Table_Final_1_cluster_fraction_shifts.csv",
    index=False
)

print("Final Table 1: cluster fraction shifts")
display(shift_merge.head(25))

# %% [notebook cell 25]
# ============================================================
# CELL 9 — Figure 6: cluster composition convergence
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# ------------------------------------------------------------
# A. Mean fraction heatmap
# ------------------------------------------------------------
mean_comp = (
    cluster_counts.groupby(["disease_group", "cluster_short"], observed=True)["fraction"]
    .mean()
    .reset_index()
)

heat = mean_comp.pivot(
    index="cluster_short",
    columns="disease_group",
    values="fraction"
).fillna(0)

existing_order = [g for g in group_order if g in heat.columns]
heat = heat[existing_order]
heat = heat.loc[clusters]

im = axes[0, 0].imshow(heat.values, aspect="auto")
axes[0, 0].set_title("A. Mean abundance of data-driven immune clusters")
axes[0, 0].set_yticks(range(len(heat.index)))
axes[0, 0].set_yticklabels(heat.index)
axes[0, 0].set_xticks(range(len(heat.columns)))
axes[0, 0].set_xticklabels([group_label_map.get(x, x) for x in heat.columns], rotation=35, ha="right")
cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
cbar.set_label("Mean cluster fraction")

# ------------------------------------------------------------
# B. AD shift vs migraine shift scatter
# ------------------------------------------------------------
plot_df = shift_merge.copy()

axes[0, 1].scatter(
    plot_df["delta_fraction_AD"],
    plot_df["delta_fraction_Migraine"],
    s=80,
    alpha=0.75
)

axes[0, 1].axhline(0, color="black", linewidth=1)
axes[0, 1].axvline(0, color="black", linewidth=1)
axes[0, 1].set_xlabel("AD - control cluster fraction")
axes[0, 1].set_ylabel("Migraine - control cluster fraction")
axes[0, 1].set_title("B. Concordance of disease-associated cluster shifts")

top_lab = plot_df.sort_values("composition_convergence_score", ascending=False).head(10)
texts = []
for _, r in top_lab.iterrows():
    texts.append(
        axes[0, 1].text(
            r["delta_fraction_AD"],
            r["delta_fraction_Migraine"],
            r["cluster_short"],
            fontsize=9
        )
    )
adjust_text(texts, ax=axes[0, 1], arrowprops=dict(arrowstyle="-", lw=0.5))

# ------------------------------------------------------------
# C. Concordance class counts
# ------------------------------------------------------------
class_counts = plot_df["concordance_class"].value_counts()
axes[1, 0].bar(class_counts.index, class_counts.values)
axes[1, 0].set_title("C. Direction of AD-migraine cluster shifts")
axes[1, 0].set_ylabel("Number of clusters")
axes[1, 0].tick_params(axis="x", rotation=20)

# ------------------------------------------------------------
# D. Top concordant clusters
# ------------------------------------------------------------
top_concordant = (
    plot_df[plot_df["same_direction"] == True]
    .sort_values("composition_convergence_score", ascending=True)
    .tail(10)
)

y_labels = top_concordant["cluster_short"].astype(str)
x_vals = top_concordant["composition_convergence_score"]

axes[1, 1].barh(y_labels, x_vals)
axes[1, 1].set_title("D. Strongest concordant cluster-shift signals")
axes[1, 1].set_xlabel("|AD shift| × |Migraine shift|")

save_final_fig(fig, "Figure6_Cluster_composition_convergence")

# %% [notebook cell 26]


# %% [notebook cell 27]
# ============================================================
# CELL 10 — Final Table 2: pseudobulk DEG and shared genes
# ============================================================

# ------------------------------------------------------------
# Pseudobulk aggregation
# ------------------------------------------------------------
def make_pseudobulk_counts(
    adata,
    group_cols=["dataset", "condition", "sample_id", "cluster_short"],
    layer="counts",
    min_cells_per_group=20
):
    obs = adata.obs[group_cols].copy()
    obs["group_id"] = obs[group_cols].astype(str).agg("|".join, axis=1)

    X = adata.layers[layer]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    meta_rows = []
    count_rows = []

    for gid, idx in obs.groupby("group_id", observed=True).groups.items():
        idx = np.array(list(idx))

        if len(idx) < min_cells_per_group:
            continue

        summed = np.asarray(X[idx, :].sum(axis=0)).ravel()

        row = dict(zip(group_cols, gid.split("|")))
        row["group_id"] = gid
        row["n_cells"] = len(idx)

        meta_rows.append(row)
        count_rows.append(summed)

    pb_counts = pd.DataFrame(count_rows, columns=adata.var_names)
    pb_meta = pd.DataFrame(meta_rows)

    return pb_counts, pb_meta

pb_counts, pb_meta = make_pseudobulk_counts(
    adata,
    group_cols=["dataset", "condition", "sample_id", "cluster_short"],
    layer="counts",
    min_cells_per_group=20
)

pb_counts.to_csv(TABLE_DIR / "Final_pseudobulk_counts.csv", index=False)
pb_meta.to_csv(TABLE_DIR / "Final_pseudobulk_metadata.csv", index=False)

print("Pseudobulk count matrix:", pb_counts.shape)
display(pb_meta.head())

# ------------------------------------------------------------
# DEG function
# ------------------------------------------------------------
def log2_cpm(count_df):
    arr = count_df.astype(float).values
    lib = arr.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    cpm = arr / lib * 1e6
    return pd.DataFrame(
        np.log2(cpm + 1),
        index=count_df.index,
        columns=count_df.columns
    )

def run_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset,
    case_label,
    control_label="Control",
    min_samples_per_group=2,
    min_mean_log2cpm=0.25
):
    all_results = []

    cluster_list = sorted(
        pb_meta.loc[pb_meta["dataset"] == dataset, "cluster_short"].unique(),
        key=lambda x: int(x.replace("C", "")) if x.replace("C", "").isdigit() else x
    )

    for cl in cluster_list:
        meta_sub = pb_meta[
            (pb_meta["dataset"] == dataset) &
            (pb_meta["cluster_short"] == cl)
        ].copy()

        if meta_sub.empty:
            continue

        idx = meta_sub.index.tolist()
        X = log2_cpm(pb_counts.loc[idx])
        meta_sub = meta_sub.reset_index(drop=True)
        X.index = meta_sub.index

        case_idx = meta_sub.index[meta_sub["condition"] == case_label].tolist()
        ctrl_idx = meta_sub.index[meta_sub["condition"] == control_label].tolist()

        if len(case_idx) < min_samples_per_group or len(ctrl_idx) < min_samples_per_group:
            print(f"Skipping {dataset} {cl}: low samples case={len(case_idx)}, control={len(ctrl_idx)}")
            continue

        case_mat = X.loc[case_idx]
        ctrl_mat = X.loc[ctrl_idx]

        mean_case = case_mat.mean(axis=0)
        mean_ctrl = ctrl_mat.mean(axis=0)
        mean_all = X.mean(axis=0)

        keep_genes = mean_all[mean_all >= min_mean_log2cpm].index.tolist()

        case_mat = case_mat[keep_genes]
        ctrl_mat = ctrl_mat[keep_genes]

        logFC = case_mat.mean(axis=0) - ctrl_mat.mean(axis=0)

        pvals = []
        for g in keep_genes:
            try:
                p = stats.ttest_ind(
                    case_mat[g],
                    ctrl_mat[g],
                    equal_var=False,
                    nan_policy="omit"
                ).pvalue
            except Exception:
                p = 1.0

            if pd.isna(p):
                p = 1.0
            pvals.append(p)

        res = pd.DataFrame({
            "dataset": dataset,
            "cluster_short": cl,
            "gene": keep_genes,
            "case_label": case_label,
            "control_label": control_label,
            "n_case_samples": len(case_idx),
            "n_control_samples": len(ctrl_idx),
            "mean_case_log2cpm": case_mat.mean(axis=0).values,
            "mean_control_log2cpm": ctrl_mat.mean(axis=0).values,
            "logFC": logFC.values,
            "pval": pvals
        })

        res["padj"] = multipletests(res["pval"].fillna(1.0), method="fdr_bh")[1]
        res["abs_logFC"] = res["logFC"].abs()
        res["neglog10_pval"] = -np.log10(res["pval"].clip(lower=1e-300))
        res["neglog10_padj"] = -np.log10(res["padj"].clip(lower=1e-300))

        res = res.sort_values(["pval", "abs_logFC"], ascending=[True, False])
        all_results.append(res)

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)

de_ad = run_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset="AD_GSE181279",
    case_label="AD",
    control_label="Control"
)

de_mig = run_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset="Migraine_GSE269117",
    case_label="Migraine",
    control_label="Control"
)

de_ad.to_csv(TABLE_DIR / "Final_AD_DE_by_data_driven_cluster.csv", index=False)
de_mig.to_csv(TABLE_DIR / "Final_Migraine_DE_by_data_driven_cluster.csv", index=False)

print("AD DE:", de_ad.shape)
print("Migraine DE:", de_mig.shape)

# ------------------------------------------------------------
# Shared AD-migraine DE genes
# ------------------------------------------------------------
shared_de = de_ad.merge(
    de_mig,
    on=["cluster_short", "gene"],
    suffixes=("_AD", "_Migraine")
)

shared_de["same_direction"] = (
    np.sign(shared_de["logFC_AD"]) ==
    np.sign(shared_de["logFC_Migraine"])
)

shared_de["shared_direction_class"] = np.where(
    (shared_de["same_direction"]) & (shared_de["logFC_AD"] > 0),
    "Shared up",
    np.where(
        (shared_de["same_direction"]) & (shared_de["logFC_AD"] < 0),
        "Shared down",
        "Opposite direction"
    )
)

# Because AD has only 3 vs 2 samples, use both p-value and effect-size evidence.
shared_de["candidate_relaxed"] = (
    (shared_de["same_direction"]) &
    (shared_de["abs_logFC_AD"] >= 0.25) &
    (shared_de["abs_logFC_Migraine"] >= 0.25) &
    (shared_de["pval_AD"] <= 0.10) &
    (shared_de["pval_Migraine"] <= 0.10)
)

shared_de["candidate_effect_only"] = (
    (shared_de["same_direction"]) &
    (shared_de["abs_logFC_AD"] >= 0.40) &
    (shared_de["abs_logFC_Migraine"] >= 0.40)
)

shared_de["combined_gene_score"] = (
    shared_de["same_direction"].astype(int) *
    np.minimum(shared_de["abs_logFC_AD"], shared_de["abs_logFC_Migraine"]) *
    shared_de["neglog10_pval_AD"].clip(upper=20) *
    shared_de["neglog10_pval_Migraine"].clip(upper=20)
)

shared_de = shared_de.merge(
    cluster_signature_map,
    on="cluster_short",
    how="left"
)

shared_de = shared_de.sort_values(
    ["candidate_relaxed", "candidate_effect_only", "combined_gene_score"],
    ascending=[False, False, False]
)

shared_de.to_csv(
    TABLE_DIR / "Table_Final_2_shared_AD_migraine_DE_genes.csv",
    index=False
)

print("Final Table 2: shared AD-migraine genes")
display(shared_de.head(40))

# %% [notebook cell 28]
# ============================================================
# CELL 10 — Final Table 2: pseudobulk DEG and shared genes
# ============================================================

# ------------------------------------------------------------
# Pseudobulk aggregation
# ------------------------------------------------------------
def make_pseudobulk_counts(
    adata,
    group_cols=["dataset", "condition", "sample_id", "cluster_short"],
    layer="counts",
    min_cells_per_group=20
):
    obs = adata.obs[group_cols].copy()
    obs["group_id"] = obs[group_cols].astype(str).agg("|".join, axis=1)

    X = adata.layers[layer]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    meta_rows = []
    count_rows = []

    for gid, cell_ids in obs.groupby("group_id", observed=True).groups.items():

        # FIX: convert string cell IDs → integer indices
        idx = adata.obs_names.get_indexer(cell_ids)
        idx = idx[idx >= 0]

        if len(idx) < min_cells_per_group:
            continue

        summed = np.asarray(X[idx, :].sum(axis=0)).ravel()

        row = dict(zip(group_cols, gid.split("|")))
        row["group_id"] = gid
        row["n_cells"] = len(idx)

        meta_rows.append(row)
        count_rows.append(summed)

    pb_counts = pd.DataFrame(count_rows, columns=adata.var_names)
    pb_meta = pd.DataFrame(meta_rows)

    return pb_counts, pb_meta


pb_counts, pb_meta = make_pseudobulk_counts(
    adata,
    group_cols=["dataset", "condition", "sample_id", "cluster_short"],
    layer="counts",
    min_cells_per_group=20
)

pb_counts.to_csv(TABLE_DIR / "Final_pseudobulk_counts.csv", index=False)
pb_meta.to_csv(TABLE_DIR / "Final_pseudobulk_metadata.csv", index=False)

print("Pseudobulk count matrix:", pb_counts.shape)
display(pb_meta.head())

# ------------------------------------------------------------
# DEG function
# ------------------------------------------------------------
def log2_cpm(count_df):
    arr = count_df.astype(float).values
    lib = arr.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    cpm = arr / lib * 1e6
    return pd.DataFrame(
        np.log2(cpm + 1),
        index=count_df.index,
        columns=count_df.columns
    )

def run_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset,
    case_label,
    control_label="Control",
    min_samples_per_group=2,
    min_mean_log2cpm=0.25
):
    all_results = []

    cluster_list = sorted(
        pb_meta.loc[pb_meta["dataset"] == dataset, "cluster_short"].unique(),
        key=lambda x: int(x.replace("C", "")) if x.replace("C", "").isdigit() else x
    )

    for cl in cluster_list:
        meta_sub = pb_meta[
            (pb_meta["dataset"] == dataset) &
            (pb_meta["cluster_short"] == cl)
        ].copy()

        if meta_sub.empty:
            continue

        idx = meta_sub.index.tolist()
        X = log2_cpm(pb_counts.loc[idx])

        meta_sub = meta_sub.reset_index(drop=True)
        X.index = meta_sub.index

        case_idx = meta_sub.index[meta_sub["condition"] == case_label].tolist()
        ctrl_idx = meta_sub.index[meta_sub["condition"] == control_label].tolist()

        if len(case_idx) < min_samples_per_group or len(ctrl_idx) < min_samples_per_group:
            print(f"Skipping {dataset} {cl}: low samples case={len(case_idx)}, control={len(ctrl_idx)}")
            continue

        case_mat = X.loc[case_idx]
        ctrl_mat = X.loc[ctrl_idx]

        mean_all = X.mean(axis=0)

        keep_genes = mean_all[mean_all >= min_mean_log2cpm].index.tolist()

        case_mat = case_mat[keep_genes]
        ctrl_mat = ctrl_mat[keep_genes]

        logFC = case_mat.mean(axis=0) - ctrl_mat.mean(axis=0)

        pvals = []
        for g in keep_genes:
            try:
                p = stats.ttest_ind(
                    case_mat[g],
                    ctrl_mat[g],
                    equal_var=False,
                    nan_policy="omit"
                ).pvalue
            except Exception:
                p = 1.0

            if pd.isna(p):
                p = 1.0

            pvals.append(p)

        res = pd.DataFrame({
            "dataset": dataset,
            "cluster_short": cl,
            "gene": keep_genes,
            "case_label": case_label,
            "control_label": control_label,
            "n_case_samples": len(case_idx),
            "n_control_samples": len(ctrl_idx),
            "mean_case_log2cpm": case_mat.mean(axis=0).values,
            "mean_control_log2cpm": ctrl_mat.mean(axis=0).values,
            "logFC": logFC.values,
            "pval": pvals
        })

        res["padj"] = multipletests(res["pval"].fillna(1.0), method="fdr_bh")[1]
        res["abs_logFC"] = res["logFC"].abs()
        res["neglog10_pval"] = -np.log10(res["pval"].clip(lower=1e-300))
        res["neglog10_padj"] = -np.log10(res["padj"].clip(lower=1e-300))

        res = res.sort_values(["pval", "abs_logFC"], ascending=[True, False])
        all_results.append(res)

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


de_ad = run_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset="AD_GSE181279",
    case_label="AD",
    control_label="Control"
)

de_mig = run_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset="Migraine_GSE269117",
    case_label="Migraine",
    control_label="Control"
)

de_ad.to_csv(TABLE_DIR / "Final_AD_DE_by_data_driven_cluster.csv", index=False)
de_mig.to_csv(TABLE_DIR / "Final_Migraine_DE_by_data_driven_cluster.csv", index=False)

print("AD DE:", de_ad.shape)
print("Migraine DE:", de_mig.shape)

# ------------------------------------------------------------
# Shared AD-migraine DE genes
# ------------------------------------------------------------
shared_de = de_ad.merge(
    de_mig,
    on=["cluster_short", "gene"],
    suffixes=("_AD", "_Migraine")
)

shared_de["same_direction"] = (
    np.sign(shared_de["logFC_AD"]) ==
    np.sign(shared_de["logFC_Migraine"])
)

shared_de["shared_direction_class"] = np.where(
    (shared_de["same_direction"]) & (shared_de["logFC_AD"] > 0),
    "Shared up",
    np.where(
        (shared_de["same_direction"]) & (shared_de["logFC_AD"] < 0),
        "Shared down",
        "Opposite direction"
    )
)

shared_de["candidate_relaxed"] = (
    (shared_de["same_direction"]) &
    (shared_de["abs_logFC_AD"] >= 0.25) &
    (shared_de["abs_logFC_Migraine"] >= 0.25) &
    (shared_de["pval_AD"] <= 0.10) &
    (shared_de["pval_Migraine"] <= 0.10)
)

shared_de["candidate_effect_only"] = (
    (shared_de["same_direction"]) &
    (shared_de["abs_logFC_AD"] >= 0.40) &
    (shared_de["abs_logFC_Migraine"] >= 0.40)
)

shared_de["combined_gene_score"] = (
    shared_de["same_direction"].astype(int) *
    np.minimum(shared_de["abs_logFC_AD"], shared_de["abs_logFC_Migraine"]) *
    shared_de["neglog10_pval_AD"].clip(upper=20) *
    shared_de["neglog10_pval_Migraine"].clip(upper=20)
)

shared_de = shared_de.merge(
    cluster_signature_map,
    on="cluster_short",
    how="left"
)

shared_de = shared_de.sort_values(
    ["candidate_relaxed", "candidate_effect_only", "combined_gene_score"],
    ascending=[False, False, False]
)

shared_de.to_csv(
    TABLE_DIR / "Table_Final_2_shared_AD_migraine_DE_genes.csv",
    index=False
)

print("Final Table 2: shared AD-migraine genes")
display(shared_de.head(40))    

# %% [notebook cell 29]


# %% [notebook cell 30]
# ============================================================
# FIXED CELL 10 — Final Table 1: cluster abundance shifts
# ============================================================

import os, re, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

# ------------------------------------------------------------
# Safety setup
# ------------------------------------------------------------
assert "dataset" in adata.obs.columns
assert "condition" in adata.obs.columns
assert "sample_id" in adata.obs.columns
assert "data_driven_cluster" in adata.obs.columns

if "cluster_short" not in adata.obs.columns:
    adata.obs["cluster_short"] = "C" + adata.obs["data_driven_cluster"].astype(str)

adata.obs["dataset"] = adata.obs["dataset"].astype(str)
adata.obs["condition"] = adata.obs["condition"].astype(str)
adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)
adata.obs["cluster_short"] = adata.obs["cluster_short"].astype(str)

# cluster signature table
if "cluster_signature" not in adata.obs.columns:
    if "cell_type" in adata.obs.columns:
        adata.obs["cluster_signature"] = adata.obs["cell_type"].astype(str)
    else:
        adata.obs["cluster_signature"] = adata.obs["cluster_short"].astype(str)

cluster_signature_map = (
    adata.obs[["cluster_short", "cluster_signature"]]
    .drop_duplicates()
    .sort_values("cluster_short")
    .reset_index(drop=True)
)

cluster_signature_map.to_csv(
    TABLE_DIR / "Final_cluster_signature_map.csv",
    index=False
)

clusters = sorted(
    adata.obs["cluster_short"].unique(),
    key=lambda x: int(x.replace("C", "")) if x.replace("C", "").isdigit() else x
)

print("Datasets:")
display(adata.obs[["dataset", "condition", "sample_id"]].drop_duplicates().sort_values(["dataset", "condition", "sample_id"]))

print("Clusters:", clusters)

# ------------------------------------------------------------
# Cell fraction per sample per cluster
# ------------------------------------------------------------
sample_meta = (
    adata.obs[["dataset", "condition", "sample_id"]]
    .drop_duplicates()
    .sort_values(["dataset", "condition", "sample_id"])
    .reset_index(drop=True)
)

raw_counts = (
    adata.obs.groupby(
        ["dataset", "condition", "sample_id", "cluster_short"],
        observed=True
    )
    .size()
    .reset_index(name="n_cells")
)

# complete all sample-cluster combinations
full_index = (
    sample_meta.assign(key=1)
    .merge(pd.DataFrame({"cluster_short": clusters, "key": 1}), on="key")
    .drop(columns="key")
)

cluster_counts = full_index.merge(
    raw_counts,
    on=["dataset", "condition", "sample_id", "cluster_short"],
    how="left"
)

cluster_counts["n_cells"] = cluster_counts["n_cells"].fillna(0).astype(int)

sample_totals = (
    cluster_counts.groupby(
        ["dataset", "condition", "sample_id"],
        observed=True
    )["n_cells"]
    .sum()
    .reset_index(name="total_cells")
)

cluster_counts = cluster_counts.merge(
    sample_totals,
    on=["dataset", "condition", "sample_id"],
    how="left"
)

cluster_counts["fraction"] = cluster_counts["n_cells"] / cluster_counts["total_cells"].replace(0, np.nan)

cluster_counts.to_csv(
    TABLE_DIR / "Table_Final_1A_cluster_fraction_per_sample.csv",
    index=False
)

# ------------------------------------------------------------
# Disease-control cluster-shift statistics
# ------------------------------------------------------------
def cluster_shift_table(cluster_counts, dataset, case_label, control_label="Control"):
    rows = []
    sub = cluster_counts[cluster_counts["dataset"] == dataset].copy()

    for cl in clusters:
        temp = sub[sub["cluster_short"] == cl].copy()

        case_vals = temp.loc[temp["condition"] == case_label, "fraction"].dropna().values
        ctrl_vals = temp.loc[temp["condition"] == control_label, "fraction"].dropna().values

        mean_case = np.mean(case_vals) if len(case_vals) else np.nan
        mean_ctrl = np.mean(ctrl_vals) if len(ctrl_vals) else np.nan
        delta = mean_case - mean_ctrl

        if len(case_vals) >= 2 and len(ctrl_vals) >= 2:
            pval = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False).pvalue
        else:
            pval = np.nan

        if len(case_vals) >= 2 and len(ctrl_vals) >= 2:
            pooled = np.sqrt((np.var(case_vals, ddof=1) + np.var(ctrl_vals, ddof=1)) / 2)
            effect = delta / pooled if pooled > 0 else np.nan
        else:
            effect = np.nan

        rows.append({
            "dataset": dataset,
            "case_label": case_label,
            "control_label": control_label,
            "cluster_short": cl,
            "mean_case_fraction": mean_case,
            "mean_control_fraction": mean_ctrl,
            "delta_fraction": delta,
            "abs_delta_fraction": abs(delta) if pd.notna(delta) else np.nan,
            "effect_size": effect,
            "pval": pval,
            "n_case_samples": len(case_vals),
            "n_control_samples": len(ctrl_vals)
        })

    out = pd.DataFrame(rows)
    out["padj"] = multipletests(out["pval"].fillna(1.0), method="fdr_bh")[1]
    return out

ad_shift = cluster_shift_table(
    cluster_counts,
    dataset="AD_GSE181279",
    case_label="AD",
    control_label="Control"
)

mig_shift = cluster_shift_table(
    cluster_counts,
    dataset="Migraine_GSE269117",
    case_label="Migraine",
    control_label="Control"
)

shift_merge = ad_shift.merge(
    mig_shift,
    on="cluster_short",
    suffixes=("_AD", "_Migraine")
)

shift_merge["same_direction"] = (
    np.sign(shift_merge["delta_fraction_AD"]) ==
    np.sign(shift_merge["delta_fraction_Migraine"])
)

shift_merge["concordance_class"] = np.where(
    (shift_merge["same_direction"]) & (shift_merge["delta_fraction_AD"] > 0),
    "Shared expansion",
    np.where(
        (shift_merge["same_direction"]) & (shift_merge["delta_fraction_AD"] < 0),
        "Shared reduction",
        "Divergent shift"
    )
)

shift_merge["composition_convergence_score"] = (
    np.abs(shift_merge["delta_fraction_AD"]) *
    np.abs(shift_merge["delta_fraction_Migraine"])
)

shift_merge = shift_merge.merge(
    cluster_signature_map,
    on="cluster_short",
    how="left"
)

shift_merge = shift_merge.sort_values(
    ["same_direction", "composition_convergence_score"],
    ascending=[False, False]
)

shift_merge.to_csv(
    TABLE_DIR / "Table_Final_1_cluster_abundance_convergence.csv",
    index=False
)

print("Table Final 1: Cluster abundance convergence")
display(shift_merge.head(30))

# %% [notebook cell 31]


# %% [notebook cell 32]


# %% [notebook cell 33]
# ============================================================
# FIXED CELL 10 — Final Table 1: cluster abundance shifts
# ============================================================

import os, re, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

# ------------------------------------------------------------
# Safety setup
# ------------------------------------------------------------
assert "dataset" in adata.obs.columns
assert "condition" in adata.obs.columns
assert "sample_id" in adata.obs.columns
assert "data_driven_cluster" in adata.obs.columns

if "cluster_short" not in adata.obs.columns:
    adata.obs["cluster_short"] = "C" + adata.obs["data_driven_cluster"].astype(str)

adata.obs["dataset"] = adata.obs["dataset"].astype(str)
adata.obs["condition"] = adata.obs["condition"].astype(str)
adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)
adata.obs["cluster_short"] = adata.obs["cluster_short"].astype(str)

# cluster signature table
if "cluster_signature" not in adata.obs.columns:
    if "cell_type" in adata.obs.columns:
        adata.obs["cluster_signature"] = adata.obs["cell_type"].astype(str)
    else:
        adata.obs["cluster_signature"] = adata.obs["cluster_short"].astype(str)

cluster_signature_map = (
    adata.obs[["cluster_short", "cluster_signature"]]
    .drop_duplicates()
    .sort_values("cluster_short")
    .reset_index(drop=True)
)

cluster_signature_map.to_csv(
    TABLE_DIR / "Final_cluster_signature_map.csv",
    index=False
)

clusters = sorted(
    adata.obs["cluster_short"].unique(),
    key=lambda x: int(x.replace("C", "")) if x.replace("C", "").isdigit() else x
)

print("Datasets:")
display(adata.obs[["dataset", "condition", "sample_id"]].drop_duplicates().sort_values(["dataset", "condition", "sample_id"]))

print("Clusters:", clusters)

# ------------------------------------------------------------
# Cell fraction per sample per cluster
# ------------------------------------------------------------
sample_meta = (
    adata.obs[["dataset", "condition", "sample_id"]]
    .drop_duplicates()
    .sort_values(["dataset", "condition", "sample_id"])
    .reset_index(drop=True)
)

raw_counts = (
    adata.obs.groupby(
        ["dataset", "condition", "sample_id", "cluster_short"],
        observed=True
    )
    .size()
    .reset_index(name="n_cells")
)

# complete all sample-cluster combinations
full_index = (
    sample_meta.assign(key=1)
    .merge(pd.DataFrame({"cluster_short": clusters, "key": 1}), on="key")
    .drop(columns="key")
)

cluster_counts = full_index.merge(
    raw_counts,
    on=["dataset", "condition", "sample_id", "cluster_short"],
    how="left"
)

cluster_counts["n_cells"] = cluster_counts["n_cells"].fillna(0).astype(int)

sample_totals = (
    cluster_counts.groupby(
        ["dataset", "condition", "sample_id"],
        observed=True
    )["n_cells"]
    .sum()
    .reset_index(name="total_cells")
)

cluster_counts = cluster_counts.merge(
    sample_totals,
    on=["dataset", "condition", "sample_id"],
    how="left"
)

cluster_counts["fraction"] = cluster_counts["n_cells"] / cluster_counts["total_cells"].replace(0, np.nan)

cluster_counts.to_csv(
    TABLE_DIR / "Table_Final_1A_cluster_fraction_per_sample.csv",
    index=False
)

# ------------------------------------------------------------
# Disease-control cluster-shift statistics
# ------------------------------------------------------------
def cluster_shift_table(cluster_counts, dataset, case_label, control_label="Control"):
    rows = []
    sub = cluster_counts[cluster_counts["dataset"] == dataset].copy()

    for cl in clusters:
        temp = sub[sub["cluster_short"] == cl].copy()

        case_vals = temp.loc[temp["condition"] == case_label, "fraction"].dropna().values
        ctrl_vals = temp.loc[temp["condition"] == control_label, "fraction"].dropna().values

        mean_case = np.mean(case_vals) if len(case_vals) else np.nan
        mean_ctrl = np.mean(ctrl_vals) if len(ctrl_vals) else np.nan
        delta = mean_case - mean_ctrl

        if len(case_vals) >= 2 and len(ctrl_vals) >= 2:
            pval = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False).pvalue
        else:
            pval = np.nan

        if len(case_vals) >= 2 and len(ctrl_vals) >= 2:
            pooled = np.sqrt((np.var(case_vals, ddof=1) + np.var(ctrl_vals, ddof=1)) / 2)
            effect = delta / pooled if pooled > 0 else np.nan
        else:
            effect = np.nan

        rows.append({
            "dataset": dataset,
            "case_label": case_label,
            "control_label": control_label,
            "cluster_short": cl,
            "mean_case_fraction": mean_case,
            "mean_control_fraction": mean_ctrl,
            "delta_fraction": delta,
            "abs_delta_fraction": abs(delta) if pd.notna(delta) else np.nan,
            "effect_size": effect,
            "pval": pval,
            "n_case_samples": len(case_vals),
            "n_control_samples": len(ctrl_vals)
        })

    out = pd.DataFrame(rows)
    out["padj"] = multipletests(out["pval"].fillna(1.0), method="fdr_bh")[1]
    return out

ad_shift = cluster_shift_table(
    cluster_counts,
    dataset="AD_GSE181279",
    case_label="AD",
    control_label="Control"
)

mig_shift = cluster_shift_table(
    cluster_counts,
    dataset="Migraine_GSE269117",
    case_label="Migraine",
    control_label="Control"
)

shift_merge = ad_shift.merge(
    mig_shift,
    on="cluster_short",
    suffixes=("_AD", "_Migraine")
)

shift_merge["same_direction"] = (
    np.sign(shift_merge["delta_fraction_AD"]) ==
    np.sign(shift_merge["delta_fraction_Migraine"])
)

shift_merge["concordance_class"] = np.where(
    (shift_merge["same_direction"]) & (shift_merge["delta_fraction_AD"] > 0),
    "Shared expansion",
    np.where(
        (shift_merge["same_direction"]) & (shift_merge["delta_fraction_AD"] < 0),
        "Shared reduction",
        "Divergent shift"
    )
)

shift_merge["composition_convergence_score"] = (
    np.abs(shift_merge["delta_fraction_AD"]) *
    np.abs(shift_merge["delta_fraction_Migraine"])
)

shift_merge = shift_merge.merge(
    cluster_signature_map,
    on="cluster_short",
    how="left"
)

shift_merge = shift_merge.sort_values(
    ["same_direction", "composition_convergence_score"],
    ascending=[False, False]
)

shift_merge.to_csv(
    TABLE_DIR / "Table_Final_1_cluster_abundance_convergence.csv",
    index=False
)

print("Table Final 1: Cluster abundance convergence")
display(shift_merge.head(30))

# %% [notebook cell 34]


# %% [notebook cell 35]
# ============================================================
# FIXED CELL 11 — Figure 6: cluster convergence proof
# ============================================================

def save_final_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()

cluster_counts["group_label"] = cluster_counts["dataset"] + "_" + cluster_counts["condition"]

group_order = [
    "AD_GSE181279_Control",
    "AD_GSE181279_AD",
    "Migraine_GSE269117_Control",
    "Migraine_GSE269117_Migraine"
]

pretty_group = {
    "AD_GSE181279_Control": "AD-control",
    "AD_GSE181279_AD": "AD",
    "Migraine_GSE269117_Control": "Migraine-control",
    "Migraine_GSE269117_Migraine": "Migraine"
}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# ------------------------------------------------------------
# A. Mean cluster abundance heatmap
# ------------------------------------------------------------
mean_abundance = (
    cluster_counts.groupby(["group_label", "cluster_short"], observed=True)["fraction"]
    .mean()
    .reset_index()
)

heat = mean_abundance.pivot(
    index="cluster_short",
    columns="group_label",
    values="fraction"
).fillna(0)

valid_groups = [g for g in group_order if g in heat.columns]
heat = heat.loc[clusters, valid_groups]

im = axes[0, 0].imshow(heat.values, aspect="auto")
axes[0, 0].set_title("A. Mean abundance of data-driven immune clusters")
axes[0, 0].set_yticks(range(len(heat.index)))
axes[0, 0].set_yticklabels(heat.index)
axes[0, 0].set_xticks(range(len(heat.columns)))
axes[0, 0].set_xticklabels([pretty_group.get(x, x) for x in heat.columns], rotation=35, ha="right")
cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
cbar.set_label("Mean cluster fraction")

# ------------------------------------------------------------
# B. AD shift vs migraine shift
# ------------------------------------------------------------
plot_df = shift_merge.copy()

axes[0, 1].scatter(
    plot_df["delta_fraction_AD"],
    plot_df["delta_fraction_Migraine"],
    s=90,
    alpha=0.75
)

axes[0, 1].axhline(0, color="black", linewidth=1)
axes[0, 1].axvline(0, color="black", linewidth=1)
axes[0, 1].set_xlabel("AD - control cluster fraction")
axes[0, 1].set_ylabel("Migraine - control cluster fraction")
axes[0, 1].set_title("B. AD-migraine concordance of cluster shifts")

top_label = plot_df.sort_values("composition_convergence_score", ascending=False).head(12)
texts = []
for _, r in top_label.iterrows():
    texts.append(
        axes[0, 1].text(
            r["delta_fraction_AD"],
            r["delta_fraction_Migraine"],
            r["cluster_short"],
            fontsize=9
        )
    )
adjust_text(texts, ax=axes[0, 1], arrowprops=dict(arrowstyle="-", lw=0.5))

# ------------------------------------------------------------
# C. Concordance class counts
# ------------------------------------------------------------
class_counts = plot_df["concordance_class"].value_counts()
axes[1, 0].bar(class_counts.index, class_counts.values)
axes[1, 0].set_title("C. Direction of cluster abundance shifts")
axes[1, 0].set_ylabel("Number of clusters")
axes[1, 0].tick_params(axis="x", rotation=20)

# ------------------------------------------------------------
# D. Top concordant cluster shifts
# ------------------------------------------------------------
top_concordant = (
    plot_df[plot_df["same_direction"] == True]
    .sort_values("composition_convergence_score", ascending=True)
    .tail(12)
)

axes[1, 1].barh(
    top_concordant["cluster_short"],
    top_concordant["composition_convergence_score"]
)

axes[1, 1].set_title("D. Strongest concordant immune-state shifts")
axes[1, 1].set_xlabel("|AD shift| × |Migraine shift|")

save_final_fig(fig, "Figure6_Cluster_abundance_convergence")

# %% [notebook cell 36]
# ============================================================
# FIXED CELL 12 — Final Table 2: global pseudobulk DEG and shared genes
# This is the main gene-level proof.
# ============================================================

# ------------------------------------------------------------
# Check raw count layer
# ------------------------------------------------------------
if "counts" not in adata.layers.keys():
    raise ValueError("adata.layers['counts'] is missing. Rerun the housekeeping-removal cell that creates adata.layers['counts'].")

X_counts = adata.layers["counts"]
if X_counts.shape != adata.shape:
    raise ValueError("adata.layers['counts'] shape does not match adata. Rerun from the corrected QC/housekeeping cell.")

# ------------------------------------------------------------
# Pseudobulk by biological sample, not by cluster
# ------------------------------------------------------------
def make_sample_pseudobulk(
    adata,
    group_cols=["dataset", "condition", "sample_id"],
    layer="counts",
    min_cells_per_sample=50
):
    obs = adata.obs[group_cols].copy()
    obs["group_id"] = obs[group_cols].astype(str).agg("|".join, axis=1)

    X = adata.layers[layer]
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    meta_rows = []
    count_rows = []

    for gid, cell_ids in obs.groupby("group_id", observed=True).groups.items():
        idx = adata.obs_names.get_indexer(cell_ids)
        idx = idx[idx >= 0]

        if len(idx) < min_cells_per_sample:
            continue

        summed = np.asarray(X[idx, :].sum(axis=0)).ravel()

        row = dict(zip(group_cols, gid.split("|")))
        row["group_id"] = gid
        row["n_cells"] = len(idx)

        meta_rows.append(row)
        count_rows.append(summed)

    pb_counts = pd.DataFrame(count_rows, columns=adata.var_names)
    pb_meta = pd.DataFrame(meta_rows)

    return pb_counts, pb_meta

pb_global_counts, pb_global_meta = make_sample_pseudobulk(
    adata,
    group_cols=["dataset", "condition", "sample_id"],
    layer="counts",
    min_cells_per_sample=50
)

pb_global_counts.to_csv(TABLE_DIR / "Final_global_pseudobulk_counts.csv", index=False)
pb_global_meta.to_csv(TABLE_DIR / "Final_global_pseudobulk_metadata.csv", index=False)

print("Global pseudobulk matrix:", pb_global_counts.shape)
display(pb_global_meta)

# ------------------------------------------------------------
# DEG function
# ------------------------------------------------------------
def log2_cpm(count_df):
    arr = count_df.astype(float).values
    lib = arr.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    cpm = arr / lib * 1e6
    return pd.DataFrame(
        np.log2(cpm + 1),
        index=count_df.index,
        columns=count_df.columns
    )

def run_global_pseudobulk_de(
    pb_counts,
    pb_meta,
    dataset,
    case_label,
    control_label="Control",
    min_mean_log2cpm=0.25
):
    meta_sub = pb_meta[pb_meta["dataset"] == dataset].copy()

    if meta_sub.empty:
        raise ValueError(f"No pseudobulk samples found for dataset={dataset}")

    idx = meta_sub.index.tolist()
    X = log2_cpm(pb_counts.loc[idx])

    meta_sub = meta_sub.reset_index(drop=True)
    X.index = meta_sub.index

    case_idx = meta_sub.index[meta_sub["condition"] == case_label].tolist()
    ctrl_idx = meta_sub.index[meta_sub["condition"] == control_label].tolist()

    print(f"{dataset}: case={case_label} n={len(case_idx)}, control={control_label} n={len(ctrl_idx)}")

    if len(case_idx) < 2 or len(ctrl_idx) < 2:
        raise ValueError(f"Too few samples for {dataset}: case={len(case_idx)}, control={len(ctrl_idx)}")

    mean_all = X.mean(axis=0)
    keep_genes = mean_all[mean_all >= min_mean_log2cpm].index.tolist()

    case_mat = X.loc[case_idx, keep_genes]
    ctrl_mat = X.loc[ctrl_idx, keep_genes]

    mean_case = case_mat.mean(axis=0)
    mean_ctrl = ctrl_mat.mean(axis=0)
    logFC = mean_case - mean_ctrl

    pvals = []
    effect_sizes = []

    for g in keep_genes:
        x = case_mat[g].values
        y = ctrl_mat[g].values

        try:
            p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue
        except Exception:
            p = 1.0

        if pd.isna(p):
            p = 1.0

        pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
        effect = (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else np.nan

        pvals.append(p)
        effect_sizes.append(effect)

    res = pd.DataFrame({
        "dataset": dataset,
        "gene": keep_genes,
        "case_label": case_label,
        "control_label": control_label,
        "n_case_samples": len(case_idx),
        "n_control_samples": len(ctrl_idx),
        "mean_case_log2cpm": mean_case.values,
        "mean_control_log2cpm": mean_ctrl.values,
        "logFC": logFC.values,
        "abs_logFC": np.abs(logFC.values),
        "effect_size": effect_sizes,
        "pval": pvals
    })

    res["padj"] = multipletests(res["pval"].fillna(1.0), method="fdr_bh")[1]
    res["neglog10_pval"] = -np.log10(res["pval"].clip(lower=1e-300))
    res["neglog10_padj"] = -np.log10(res["padj"].clip(lower=1e-300))

    res = res.sort_values(["pval", "abs_logFC"], ascending=[True, False])

    return res

de_ad_global = run_global_pseudobulk_de(
    pb_global_counts,
    pb_global_meta,
    dataset="AD_GSE181279",
    case_label="AD",
    control_label="Control"
)

de_mig_global = run_global_pseudobulk_de(
    pb_global_counts,
    pb_global_meta,
    dataset="Migraine_GSE269117",
    case_label="Migraine",
    control_label="Control"
)

de_ad_global.to_csv(TABLE_DIR / "Final_AD_global_pseudobulk_DE.csv", index=False)
de_mig_global.to_csv(TABLE_DIR / "Final_Migraine_global_pseudobulk_DE.csv", index=False)

print("AD global DE:", de_ad_global.shape)
print("Migraine global DE:", de_mig_global.shape)

# ------------------------------------------------------------
# Shared genes: merge by gene only
# ------------------------------------------------------------
shared_gene = de_ad_global.merge(
    de_mig_global,
    on="gene",
    suffixes=("_AD", "_Migraine")
)

shared_gene["same_direction"] = (
    np.sign(shared_gene["logFC_AD"]) ==
    np.sign(shared_gene["logFC_Migraine"])
)

shared_gene["shared_direction_class"] = np.where(
    (shared_gene["same_direction"]) & (shared_gene["logFC_AD"] > 0),
    "Shared up",
    np.where(
        (shared_gene["same_direction"]) & (shared_gene["logFC_AD"] < 0),
        "Shared down",
        "Opposite direction"
    )
)

# Small cohort: use effect-size evidence + nominal p-value evidence.
shared_gene["candidate_relaxed"] = (
    (shared_gene["same_direction"]) &
    (shared_gene["abs_logFC_AD"] >= 0.25) &
    (shared_gene["abs_logFC_Migraine"] >= 0.25) &
    (
        (shared_gene["pval_AD"] <= 0.20) |
        (shared_gene["pval_Migraine"] <= 0.20)
    )
)

shared_gene["candidate_effect_only"] = (
    (shared_gene["same_direction"]) &
    (shared_gene["abs_logFC_AD"] >= 0.40) &
    (shared_gene["abs_logFC_Migraine"] >= 0.40)
)

shared_gene["combined_gene_score"] = (
    shared_gene["same_direction"].astype(int) *
    np.minimum(shared_gene["abs_logFC_AD"], shared_gene["abs_logFC_Migraine"]) *
    shared_gene["neglog10_pval_AD"].clip(upper=20) *
    shared_gene["neglog10_pval_Migraine"].clip(upper=20)
)

# ------------------------------------------------------------
# Map shared genes to data-driven clusters using cluster markers
# ------------------------------------------------------------
if "cluster_marker_df" not in globals():
    print("cluster_marker_df not found. Recomputing cluster markers.")
    sc.tl.rank_genes_groups(
        adata,
        groupby="data_driven_cluster",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        key_added="cluster_markers_for_final"
    )
    cluster_marker_df = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key="cluster_markers_for_final"
    )
    cluster_marker_df = cluster_marker_df.rename(
        columns={
            "group": "cluster",
            "names": "gene",
            "scores": "score",
            "logfoldchanges": "logFC",
            "pvals": "pval",
            "pvals_adj": "padj"
        }
    )

cluster_marker_small = cluster_marker_df.copy()
cluster_marker_small["cluster_short"] = "C" + cluster_marker_small["cluster"].astype(str)

# take best marker cluster per gene
best_marker_cluster = (
    cluster_marker_small
    .sort_values(["padj", "logFC"], ascending=[True, False])
    .drop_duplicates("gene")
    [["gene", "cluster_short", "logFC", "padj"]]
    .rename(columns={
        "cluster_short": "marker_cluster",
        "logFC": "marker_cluster_logFC",
        "padj": "marker_cluster_padj"
    })
)

shared_gene = shared_gene.merge(
    best_marker_cluster,
    on="gene",
    how="left"
)

shared_gene = shared_gene.merge(
    cluster_signature_map.rename(columns={"cluster_short": "marker_cluster"}),
    on="marker_cluster",
    how="left"
)

shared_gene = shared_gene.sort_values(
    ["candidate_relaxed", "candidate_effect_only", "combined_gene_score"],
    ascending=[False, False, False]
)

shared_gene.to_csv(
    TABLE_DIR / "Table_Final_2_shared_global_AD_migraine_DE_genes.csv",
    index=False
)

print("Final Table 2: shared global AD-migraine genes")
print("Shared gene table shape:", shared_gene.shape)
display(shared_gene.head(50))

# %% [notebook cell 37]


# %% [notebook cell 38]


# %% [notebook cell 39]
# ============================================================
# FIXED CELL 13 — Figure 7: shared gene proof
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# ------------------------------------------------------------
# A. AD vs migraine logFC concordance
# ------------------------------------------------------------
same = shared_gene[shared_gene["same_direction"] == True].copy()
opp = shared_gene[shared_gene["same_direction"] == False].copy()

axes[0, 0].scatter(
    opp["logFC_AD"],
    opp["logFC_Migraine"],
    s=8,
    alpha=0.25,
    label="Opposite direction"
)

axes[0, 0].scatter(
    same["logFC_AD"],
    same["logFC_Migraine"],
    s=12,
    alpha=0.55,
    label="Same direction"
)

axes[0, 0].axhline(0, color="black", linewidth=1)
axes[0, 0].axvline(0, color="black", linewidth=1)
axes[0, 0].set_xlabel("AD log2 fold-change")
axes[0, 0].set_ylabel("Migraine log2 fold-change")
axes[0, 0].set_title("A. AD-migraine gene regulation concordance")
axes[0, 0].legend(frameon=False)

top_label = (
    shared_gene[shared_gene["same_direction"] == True]
    .sort_values("combined_gene_score", ascending=False)
    .head(15)
)

texts = []
for _, r in top_label.iterrows():
    texts.append(
        axes[0, 0].text(
            r["logFC_AD"],
            r["logFC_Migraine"],
            r["gene"],
            fontsize=8
        )
    )
adjust_text(texts, ax=axes[0, 0], arrowprops=dict(arrowstyle="-", lw=0.5))

# ------------------------------------------------------------
# B. Top shared genes
# ------------------------------------------------------------
candidate_gene = shared_gene[
    (shared_gene["candidate_relaxed"] == True) |
    (shared_gene["candidate_effect_only"] == True)
].copy()

if candidate_gene.empty:
    candidate_gene = (
        shared_gene[shared_gene["same_direction"] == True]
        .sort_values("combined_gene_score", ascending=False)
        .head(300)
        .copy()
    )

top_bar = (
    candidate_gene
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(20)
    .sort_values("combined_gene_score")
)

axes[0, 1].barh(top_bar["gene"], top_bar["combined_gene_score"])
axes[0, 1].set_title("B. Top shared AD-migraine candidate genes")
axes[0, 1].set_xlabel("Combined gene score")

# ------------------------------------------------------------
# C. logFC heatmap for top shared genes
# ------------------------------------------------------------
top_heat = (
    candidate_gene
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(25)
)

mat = top_heat[["logFC_AD", "logFC_Migraine"]].values

im = axes[1, 0].imshow(mat, aspect="auto", cmap="coolwarm")
axes[1, 0].set_yticks(range(len(top_heat)))
axes[1, 0].set_yticklabels(top_heat["gene"].tolist(), fontsize=8)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["AD", "Migraine"])
axes[1, 0].set_title("C. Concordant regulation of top shared genes")
cbar = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
cbar.set_label("log2 fold-change")

# ------------------------------------------------------------
# D. Shared candidate genes mapped to data-driven clusters
# ------------------------------------------------------------
if "marker_cluster" in candidate_gene.columns:
    cluster_gene_counts = (
        candidate_gene.dropna(subset=["marker_cluster"])
        .groupby("marker_cluster", observed=True)
        .size()
        .reindex(clusters)
        .fillna(0)
    )
else:
    cluster_gene_counts = pd.Series(dtype=float)

axes[1, 1].bar(cluster_gene_counts.index, cluster_gene_counts.values)
axes[1, 1].set_title("D. Shared candidate genes mapped to clusters")
axes[1, 1].set_ylabel("Number of genes")
axes[1, 1].tick_params(axis="x", rotation=45)

save_final_fig(fig, "Figure7_Shared_gene_convergence")

# %% [notebook cell 40]


# %% [notebook cell 41]


# %% [notebook cell 42]
# ============================================================
# FIXED CELL 14 — Final Table 3: shared pathway enrichment
# ============================================================

IMMUNE_MODULES = {
    "Cytotoxic T/NK activity": [
        "NKG7", "GNLY", "PRF1", "GZMA", "GZMB", "GZMH", "CST7", "CTSW",
        "KLRD1", "KLRF1", "KLRB1", "XCL1", "XCL2", "CX3CR1", "TBX21"
    ],
    "T-cell activation": [
        "CD3D", "CD3E", "CD3G", "TRAC", "IL7R", "CCR7", "LCK", "LAT",
        "ZAP70", "CD247", "ICOS", "CD40LG", "CD69", "DUSP1", "JUN", "JUNB"
    ],
    "B-cell identity and activation": [
        "MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "IGHM", "IGHD",
        "IGKC", "TCL1A", "FCER2", "SPIB", "POU2AF1", "FCRLA", "TNFRSF13B"
    ],
    "Antigen presentation and HLA": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1",
        "HLA-DQB1", "HLA-DMA", "HLA-DMB", "HLA-A", "HLA-B", "HLA-C",
        "CD74", "TAP1", "TAP2"
    ],
    "Inflammatory monocyte program": [
        "LYZ", "LST1", "S100A8", "S100A9", "S100A12", "FCN1", "CTSS",
        "CST3", "SERPINA1", "AIF1", "TYROBP", "TREM1", "LGALS3"
    ],
    "Fc-receptor TYROBP innate signaling": [
        "TYROBP", "FCER1G", "FCGR3A", "FCGR2A", "FCGR1A", "LILRB1",
        "LILRB2", "LST1", "AIF1", "CTSS", "SH2D1B", "SPON2"
    ],
    "Interferon response": [
        "ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI27", "IFI44",
        "IFI44L", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "IRF7",
        "STAT1", "STAT2", "RSAD2"
    ],
    "TNF NF-kB inflammatory response": [
        "TNF", "NFKB1", "NFKBIA", "RELA", "TNFAIP3", "BIRC3",
        "CXCL8", "CXCL2", "CCL2", "CCL3", "CCL4", "ICAM1",
        "JUN", "FOS", "DUSP1", "IL6", "PTGS2"
    ],
    "S100 calgranulin inflammation": [
        "S100A8", "S100A9", "S100A12", "S100A4", "S100A6", "S100A10",
        "LGALS3", "FCN1", "LYZ", "LST1"
    ],
    "Chemokine signaling": [
        "CCL2", "CCL3", "CCL4", "CCL5", "CXCL8", "CXCL10", "CXCL16",
        "CCR1", "CCR2", "CCR5", "CXCR3", "CXCR4", "XCL1", "XCL2"
    ]
}

def pathway_enrichment(query_genes, universe_genes, modules):
    query_genes = set([g for g in query_genes if isinstance(g, str)])
    universe_genes = set(universe_genes)

    rows = []

    for pathway, genes in modules.items():
        module_genes = set(genes).intersection(universe_genes)

        if len(module_genes) == 0:
            continue

        overlap = query_genes.intersection(module_genes)

        a = len(overlap)
        b = len(query_genes) - a
        c = len(module_genes) - a
        d = len(universe_genes) - a - b - c

        if a == 0:
            odds_ratio = 0.0
            pval = 1.0
        else:
            odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

        rows.append({
            "pathway": pathway,
            "overlap_n": a,
            "query_gene_n": len(query_genes),
            "pathway_gene_n": len(module_genes),
            "odds_ratio": odds_ratio,
            "pval": pval,
            "overlap_genes": ";".join(sorted(overlap))
        })

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out["padj"] = multipletests(out["pval"].fillna(1.0), method="fdr_bh")[1]
        out["neglog10_padj"] = -np.log10(out["padj"].clip(lower=1e-300))
        out = out.sort_values(["padj", "overlap_n"], ascending=[True, False])

    return out

# ------------------------------------------------------------
# Select shared candidate genes
# ------------------------------------------------------------
candidate_gene = shared_gene[
    (shared_gene["candidate_relaxed"] == True) |
    (shared_gene["candidate_effect_only"] == True)
].copy()

if candidate_gene.empty:
    candidate_gene = (
        shared_gene[shared_gene["same_direction"] == True]
        .sort_values("combined_gene_score", ascending=False)
        .head(300)
        .copy()
    )

candidate_gene = candidate_gene[candidate_gene["same_direction"] == True].copy()

shared_candidate_genes = (
    candidate_gene.sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")["gene"]
    .tolist()
)

print("Number of shared candidate genes for pathway enrichment:", len(shared_candidate_genes))

universe_genes = list(adata.var_names)

global_enrichment = pathway_enrichment(
    shared_candidate_genes,
    universe_genes,
    IMMUNE_MODULES
)

global_enrichment["analysis_level"] = "Global shared AD-migraine genes"

# cluster-mapped enrichment
cluster_enrichment_rows = []

if "marker_cluster" in candidate_gene.columns:
    for cl in clusters:
        genes = (
            candidate_gene[
                (candidate_gene["marker_cluster"] == cl) &
                (candidate_gene["same_direction"] == True)
            ]
            .sort_values("combined_gene_score", ascending=False)
            .drop_duplicates("gene")["gene"]
            .tolist()
        )

        if len(genes) < 5:
            continue

        enr = pathway_enrichment(genes, universe_genes, IMMUNE_MODULES)
        if len(enr) > 0:
            enr["analysis_level"] = cl
            cluster_enrichment_rows.append(enr)

if len(cluster_enrichment_rows) > 0:
    cluster_enrichment = pd.concat(cluster_enrichment_rows, ignore_index=True)
else:
    cluster_enrichment = pd.DataFrame()

pathway_table = pd.concat(
    [global_enrichment, cluster_enrichment],
    ignore_index=True
)

pathway_table.to_csv(
    TABLE_DIR / "Table_Final_3_shared_pathway_enrichment.csv",
    index=False
)

print("Table Final 3: Shared pathway enrichment")
display(pathway_table.head(50))

# %% [notebook cell 43]


# %% [notebook cell 44]


# %% [notebook cell 45]
# ============================================================
# FIXED CELL 15 — Figure 8: pathway-level proof
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# ------------------------------------------------------------
# A. Global pathway ranking
# ------------------------------------------------------------
global_pw = pathway_table[
    pathway_table["analysis_level"] == "Global shared AD-migraine genes"
].copy()

top_global_pw = global_pw.sort_values("neglog10_padj", ascending=True).tail(10)

axes[0, 0].barh(
    top_global_pw["pathway"],
    top_global_pw["neglog10_padj"]
)

axes[0, 0].set_title("A. Enriched pathways among shared AD-migraine genes")
axes[0, 0].set_xlabel("-log10(FDR)")

# ------------------------------------------------------------
# B. Overlap gene number per pathway
# ------------------------------------------------------------
top_overlap = global_pw.sort_values("overlap_n", ascending=True).tail(10)

axes[0, 1].barh(
    top_overlap["pathway"],
    top_overlap["overlap_n"]
)

axes[0, 1].set_title("B. Shared-gene support for immune pathways")
axes[0, 1].set_xlabel("Number of overlapping genes")

# ------------------------------------------------------------
# C. Cluster-level pathway heatmap
# ------------------------------------------------------------
cluster_pw = pathway_table[
    pathway_table["analysis_level"].isin(clusters)
].copy()

if len(cluster_pw) > 0:
    pvt = cluster_pw.pivot_table(
        index="pathway",
        columns="analysis_level",
        values="neglog10_padj",
        aggfunc="max",
        fill_value=0
    )

    ordered_cols = [c for c in clusters if c in pvt.columns]
    pvt = pvt[ordered_cols]
    pvt = pvt.loc[pvt.max(axis=1).sort_values(ascending=False).index]

    im = axes[1, 0].imshow(pvt.values, aspect="auto")
    axes[1, 0].set_title("C. Cluster-mapped pathway enrichment")
    axes[1, 0].set_yticks(range(len(pvt.index)))
    axes[1, 0].set_yticklabels(pvt.index, fontsize=8)
    axes[1, 0].set_xticks(range(len(pvt.columns)))
    axes[1, 0].set_xticklabels(pvt.columns, rotation=45)
    cbar = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("-log10(FDR)")
else:
    axes[1, 0].text(
        0.5, 0.5,
        "Cluster-level enrichment not available\nusing global pathway proof",
        ha="center",
        va="center"
    )
    axes[1, 0].set_title("C. Cluster-mapped pathway enrichment")

# ------------------------------------------------------------
# D. Top shared genes with concordant logFC
# ------------------------------------------------------------
top_gene_panel = (
    candidate_gene[candidate_gene["same_direction"] == True]
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(15)
    .sort_values("combined_gene_score")
)

y = np.arange(len(top_gene_panel))

axes[1, 1].barh(
    y - 0.18,
    top_gene_panel["logFC_AD"],
    height=0.35,
    label="AD"
)

axes[1, 1].barh(
    y + 0.18,
    top_gene_panel["logFC_Migraine"],
    height=0.35,
    label="Migraine"
)

axes[1, 1].axvline(0, color="black", linewidth=1)
axes[1, 1].set_yticks(y)
axes[1, 1].set_yticklabels(top_gene_panel["gene"])
axes[1, 1].set_xlabel("log2 fold-change")
axes[1, 1].set_title("D. Concordant regulation of top shared genes")
axes[1, 1].legend(frameon=False)

save_final_fig(fig, "Figure8_Shared_pathway_biology")

# %% [notebook cell 46]


# %% [notebook cell 47]
# ============================================================
# FIXED CELL 16 — Export final short-paper package
# ============================================================

excel_path = RESULTS_DIR / "Final_AD_Migraine_short_paper_results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    cluster_signature_map.to_excel(writer, sheet_name="Cluster_signatures", index=False)
    cluster_counts.to_excel(writer, sheet_name="Cluster_fractions", index=False)
    shift_merge.to_excel(writer, sheet_name="T1_cluster_convergence", index=False)
    de_ad_global.to_excel(writer, sheet_name="AD_global_DE", index=False)
    de_mig_global.to_excel(writer, sheet_name="Migraine_global_DE", index=False)
    shared_gene.head(10000).to_excel(writer, sheet_name="T2_shared_genes", index=False)
    pathway_table.to_excel(writer, sheet_name="T3_shared_pathways", index=False)

print("Saved Excel workbook:")
print(excel_path)

# Write a small result interpretation text file
summary_txt = f"""
Short paper result summary

Main claim:
A data-driven PBMC single-cell reanalysis reveals that Alzheimer disease and migraine share peripheral immune remodeling at the level of immune-state abundance, gene regulation, and immune pathway activity.

Table/Figure proof:
1. Table_Final_1_cluster_abundance_convergence.csv + Figure6:
   Tests whether data-driven immune clusters shift in the same direction in AD and migraine.

2. Table_Final_2_shared_global_AD_migraine_DE_genes.csv + Figure7:
   Tests whether AD and migraine share same-direction disease-associated genes using sample-level pseudobulk analysis.

3. Table_Final_3_shared_pathway_enrichment.csv + Figure8:
   Tests whether shared genes converge on immune pathways.

Important limitation:
The AD dataset has only 3 AD and 2 control donors, so DEG and pathway results should be interpreted as exploratory and hypothesis-generating.
"""

with open(RESULTS_DIR / "Final_short_paper_result_summary.txt", "w") as f:
    f.write(summary_txt)

final_zip = "./Final_AD_Migraine_short_paper_results"
shutil.make_archive(final_zip, "zip", RESULTS_DIR)

print("Zipped results:")
print(final_zip + ".zip")

print("\nFinal result files:")
for p in sorted(RESULTS_DIR.rglob("*")):
    if p.is_file():
        print(p)

# %% [notebook cell 48]


# %% [notebook cell 49]


# %% [notebook cell 50]
# ============================================================
# FINAL RESULT SECTION CODE
# Nature-style tables and figures for short paper
# ============================================================

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from adjustText import adjust_text

# ============================================================
# 0. Folders and style
# ============================================================

try:
    RESULTS_DIR
except NameError:
    RESULTS_DIR = Path("results")

try:
    TABLE_DIR
except NameError:
    TABLE_DIR = RESULTS_DIR / "tables"

try:
    FIG_DIR
except NameError:
    FIG_DIR = RESULTS_DIR / "figures"

FINAL_DIR = RESULTS_DIR / "final_result_section"
FINAL_TABLE_DIR = FINAL_DIR / "tables"
FINAL_FIG_DIR = FINAL_DIR / "figures"

for d in [FINAL_DIR, FINAL_TABLE_DIR, FINAL_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Nature-like plot style
# ----------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_AD = "#3B6FB6"
COL_MIG = "#D95F02"
COL_SHARED = "#1B9E77"
COL_OPP = "#BDBDBD"
COL_DARK = "#222222"
COL_LIGHT = "#F2F2F2"

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)

def panel_label(ax, label):
    ax.text(
        -0.12, 1.06, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left"
    )

def save_fig(fig, name, wspace=0.35, hspace=0.35):
    fig.tight_layout()
    fig.savefig(FINAL_FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FINAL_FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()

def ensure_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)

print("Final outputs will be saved to:")
print(FINAL_DIR)


# ============================================================
# FORCE DOWNLOAD (KAGGLE / JUPYTER COMPATIBLE)
# ============================================================

import zipfile
from IPython.display import FileLink, display

ZIP_PATH = RESULTS_DIR / "final_result_section.zip"

def zip_directory(folder_path, zip_path):
    folder_path = Path(folder_path)
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(folder_path))

    print(f"✅ Created ZIP: {zip_path}")

zip_directory(FINAL_DIR, ZIP_PATH)

# THIS is the closest to "direct download"
display(FileLink(str(ZIP_PATH)))

print("\n👉 Click the link above to download instantly.")

# %% [notebook cell 51]
# ============================================================
# 1. Safety checks and loading fallback
# ============================================================

def load_if_missing(var_name, file_candidates):
    """
    Load dataframe from CSV if variable is missing from memory.
    """
    if var_name in globals():
        return globals()[var_name]

    for f in file_candidates:
        f = Path(f)
        if f.exists():
            print(f"Loaded {var_name} from {f}")
            return pd.read_csv(f)

    raise ValueError(f"{var_name} not found in memory or files.")

# Load fallback if needed
cluster_counts = load_if_missing(
    "cluster_counts",
    [
        TABLE_DIR / "Table_Final_1A_cluster_fraction_per_sample.csv",
        TABLE_DIR / "Final_cluster_fraction_per_sample.csv"
    ]
)

shift_merge = load_if_missing(
    "shift_merge",
    [
        TABLE_DIR / "Table_Final_1_cluster_abundance_convergence.csv",
        TABLE_DIR / "Table_Final_1_cluster_fraction_shifts.csv"
    ]
)

shared_gene = load_if_missing(
    "shared_gene",
    [
        TABLE_DIR / "Table_Final_2_shared_global_AD_migraine_DE_genes.csv",
        TABLE_DIR / "Table_Final_2_shared_AD_migraine_DE_genes.csv"
    ]
)

pathway_table = load_if_missing(
    "pathway_table",
    [
        TABLE_DIR / "Table_Final_3_shared_pathway_enrichment.csv"
    ]
)

de_ad_global = load_if_missing(
    "de_ad_global",
    [
        TABLE_DIR / "Final_AD_global_pseudobulk_DE.csv"
    ]
)

de_mig_global = load_if_missing(
    "de_mig_global",
    [
        TABLE_DIR / "Final_Migraine_global_pseudobulk_DE.csv"
    ]
)

cluster_signature_map = load_if_missing(
    "cluster_signature_map",
    [
        TABLE_DIR / "Final_cluster_signature_map.csv"
    ]
)

# Normalize boolean columns
for df in [shared_gene, shift_merge]:
    for col in ["same_direction", "candidate_relaxed", "candidate_effect_only"]:
        if col in df.columns:
            df[col] = ensure_bool(df[col])

# Cluster order
clusters = sorted(
    cluster_counts["cluster_short"].dropna().unique(),
    key=lambda x: int(str(x).replace("C", "")) if str(x).replace("C", "").isdigit() else str(x)
)

print("Available clusters:", clusters)
print("Shared gene table:", shared_gene.shape)
print("Pathway table:", pathway_table.shape)

# %% [notebook cell 52]
# ============================================================
# 2. TABLE 1 — Dataset and analysis statistics
# Compact table for manuscript
# ============================================================

def dataset_stats_from_adata(adata, dataset_name, disease_label):
    obs = adata.obs.copy()
    sub = obs[obs["dataset"].astype(str) == dataset_name].copy()

    n_cells = sub.shape[0]
    n_genes = adata.n_vars

    samples = (
        sub[["sample_id", "condition"]]
        .drop_duplicates()
        .sort_values(["condition", "sample_id"])
    )

    n_case = samples[samples["condition"] == disease_label]["sample_id"].nunique()
    n_ctrl = samples[samples["condition"] == "Control"]["sample_id"].nunique()
    n_samples = samples["sample_id"].nunique()

    cells_per_sample = sub.groupby("sample_id", observed=True).size()
    median_cells = int(np.median(cells_per_sample))
    min_cells = int(np.min(cells_per_sample))
    max_cells = int(np.max(cells_per_sample))

    if "cluster_short" in sub.columns:
        n_clusters = sub["cluster_short"].nunique()
    else:
        n_clusters = np.nan

    return {
        "Dataset": dataset_name,
        "Disease comparison": f"{disease_label} vs Control",
        "Case donors": n_case,
        "Control donors": n_ctrl,
        "Total donors": n_samples,
        "Cells after QC": n_cells,
        "Genes retained after filtering": n_genes,
        "Median cells/donor": median_cells,
        "Cell range/donor": f"{min_cells}-{max_cells}",
        "Data-driven clusters observed": n_clusters
    }

table1 = pd.DataFrame([
    dataset_stats_from_adata(adata, "AD_GSE181279", "AD"),
    dataset_stats_from_adata(adata, "Migraine_GSE269117", "Migraine")
])

# Add DEG summary columns
ad_sig = de_ad_global[
    (de_ad_global["abs_logFC"] >= 0.25) &
    (de_ad_global["pval"] <= 0.05)
].shape[0]

mig_sig = de_mig_global[
    (de_mig_global["abs_logFC"] >= 0.25) &
    (de_mig_global["pval"] <= 0.05)
].shape[0]

table1.loc[table1["Dataset"] == "AD_GSE181279", "Nominal DE genes |logFC|≥0.25,p≤0.05"] = ad_sig
table1.loc[table1["Dataset"] == "Migraine_GSE269117", "Nominal DE genes |logFC|≥0.25,p≤0.05"] = mig_sig

table1.to_csv(FINAL_TABLE_DIR / "Table1_dataset_and_analysis_statistics.csv", index=False)

display(table1)

# %% [notebook cell 53]
# ============================================================
# 3. TABLE 2 — Top shared AD–migraine genes
# ============================================================

candidate_gene = shared_gene[
    (
        (shared_gene.get("candidate_relaxed", False) == True) |
        (shared_gene.get("candidate_effect_only", False) == True)
    ) &
    (shared_gene["same_direction"] == True)
].copy()

if candidate_gene.empty:
    candidate_gene = (
        shared_gene[shared_gene["same_direction"] == True]
        .sort_values("combined_gene_score", ascending=False)
        .head(500)
        .copy()
    )

table2_cols = [
    "gene",
    "shared_direction_class",
    "logFC_AD",
    "pval_AD",
    "padj_AD",
    "logFC_Migraine",
    "pval_Migraine",
    "padj_Migraine",
    "combined_gene_score",
    "marker_cluster",
    "cluster_signature"
]

table2_cols = [c for c in table2_cols if c in candidate_gene.columns]

table2 = (
    candidate_gene
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(40)[table2_cols]
    .copy()
)

table2.to_csv(FINAL_TABLE_DIR / "Table2_top_shared_AD_migraine_genes.csv", index=False)

display(table2.head(25))

# %% [notebook cell 54]


# %% [notebook cell 55]
# ============================================================
# 4. TABLE 3 — Shared immune pathway enrichment
# ============================================================

global_pathways = pathway_table[
    pathway_table["analysis_level"].astype(str).str.contains("Global shared", case=False, na=False)
].copy()

if global_pathways.empty:
    global_pathways = pathway_table.copy()

table3_cols = [
    "pathway",
    "overlap_n",
    "query_gene_n",
    "pathway_gene_n",
    "odds_ratio",
    "pval",
    "padj",
    "neglog10_padj",
    "overlap_genes"
]

table3_cols = [c for c in table3_cols if c in global_pathways.columns]

table3 = (
    global_pathways
    .sort_values(["padj", "overlap_n"], ascending=[True, False])
    .head(10)[table3_cols]
    .copy()
)

table3.to_csv(FINAL_TABLE_DIR / "Table3_shared_immune_pathway_enrichment.csv", index=False)

display(table3)

# %% [notebook cell 56]
# ============================================================
# 5. Save all manuscript tables into one Excel workbook
# ============================================================

excel_path = FINAL_TABLE_DIR / "Final_Result_Section_Tables.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    table1.to_excel(writer, sheet_name="Table1_dataset_stats", index=False)
    table2.to_excel(writer, sheet_name="Table2_shared_genes", index=False)
    table3.to_excel(writer, sheet_name="Table3_shared_pathways", index=False)
    cluster_signature_map.to_excel(writer, sheet_name="Cluster_signatures", index=False)
    shift_merge.to_excel(writer, sheet_name="Cluster_abundance", index=False)
    shared_gene.head(10000).to_excel(writer, sheet_name="Shared_gene_full_top10000", index=False)
    pathway_table.to_excel(writer, sheet_name="Pathway_full", index=False)

print("Saved:", excel_path)

# %% [notebook cell 57]


# %% [notebook cell 58]
# ============================================================
# 6. FIGURE 1 — Dataset and data-driven immune atlas overview
# ============================================================

# Prepare UMAP dataframe
assert "X_umap" in adata.obsm.keys(), "UMAP not found in adata.obsm['X_umap']"

umap_df = pd.DataFrame(
    adata.obsm["X_umap"],
    columns=["UMAP1", "UMAP2"],
    index=adata.obs_names
)

umap_df["dataset"] = adata.obs["dataset"].astype(str).values
umap_df["condition"] = adata.obs["condition"].astype(str).values
umap_df["sample_id"] = adata.obs["sample_id"].astype(str).values
umap_df["cluster_short"] = adata.obs["cluster_short"].astype(str).values
umap_df["group"] = umap_df["dataset"] + "_" + umap_df["condition"]

group_pretty = {
    "AD_GSE181279_AD": "AD",
    "AD_GSE181279_Control": "AD-control",
    "Migraine_GSE269117_Migraine": "Migraine",
    "Migraine_GSE269117_Control": "Migraine-control",
}

group_colors = {
    "AD_GSE181279_AD": "#3B6FB6",
    "AD_GSE181279_Control": "#9BB7E5",
    "Migraine_GSE269117_Migraine": "#D95F02",
    "Migraine_GSE269117_Control": "#F3B27C",
}

cluster_cmap = plt.get_cmap("tab20", len(clusters))
cluster_colors = {cl: cluster_cmap(i) for i, cl in enumerate(clusters)}

# Shuffle for fair plotting
plot_umap = umap_df.sample(frac=1.0, random_state=7)

fig = plt.figure(figsize=(7.2, 6.4))
gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.42)

# A. UMAP by disease group
ax = fig.add_subplot(gs[0, 0])
for g, sub in plot_umap.groupby("group"):
    ax.scatter(
        sub["UMAP1"], sub["UMAP2"],
        s=1.2,
        alpha=0.55,
        linewidths=0,
        color=group_colors.get(g, "#999999"),
        label=group_pretty.get(g, g)
    )
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("Disease and control PBMC profiles")
ax.legend(frameon=False, markerscale=5, loc="best")
clean_axis(ax)
panel_label(ax, "A")

# B. UMAP by data-driven clusters
ax = fig.add_subplot(gs[0, 1])
for cl in clusters:
    sub = plot_umap[plot_umap["cluster_short"] == cl]
    ax.scatter(
        sub["UMAP1"], sub["UMAP2"],
        s=1.2,
        alpha=0.55,
        linewidths=0,
        color=cluster_colors[cl]
    )

# cluster labels at medians
for cl in clusters:
    sub = umap_df[umap_df["cluster_short"] == cl]
    if len(sub) > 0:
        ax.text(
            sub["UMAP1"].median(),
            sub["UMAP2"].median(),
            cl,
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="center"
        )

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("Data-driven immune transcriptional clusters")
clean_axis(ax)
panel_label(ax, "B")

# C. Cells per donor
ax = fig.add_subplot(gs[1, 0])
sample_cells = (
    adata.obs.groupby(["dataset", "condition", "sample_id"], observed=True)
    .size()
    .reset_index(name="cells")
    .sort_values(["dataset", "condition", "sample_id"])
)

x = np.arange(sample_cells.shape[0])
bar_colors = [
    group_colors.get(d + "_" + c, "#999999")
    for d, c in zip(sample_cells["dataset"], sample_cells["condition"])
]

ax.bar(x, sample_cells["cells"], color=bar_colors, width=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sample_cells["sample_id"], rotation=90)
ax.set_ylabel("Cells after QC")
ax.set_title("Cells retained per donor")
clean_axis(ax)
panel_label(ax, "C")

# D. Compact cluster signature table as text
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")

sig_show = cluster_signature_map.copy()
sig_show["cluster_num"] = sig_show["cluster_short"].str.replace("C", "", regex=False).astype(int)
sig_show = sig_show.sort_values("cluster_num").head(12)

txt_lines = []
for _, r in sig_show.iterrows():
    sig = str(r["cluster_signature"])
    sig = sig.replace(str(r["cluster_short"]) + ": ", "")
    txt_lines.append(f"{r['cluster_short']}: {sig}")

ax.text(
    0.0, 1.0,
    "\n".join(txt_lines),
    ha="left",
    va="top",
    fontsize=6,
    linespacing=1.35
)
ax.set_title("Representative cluster signatures")
panel_label(ax, "D")

save_fig(fig, "Figure1_Dataset_and_data_driven_atlas")

# %% [notebook cell 59]


# %% [notebook cell 60]
# ============================================================
# 7. FIGURE 2 — Shared gene-level AD–migraine convergence
# ============================================================

candidate_gene = shared_gene[
    (
        (shared_gene.get("candidate_relaxed", False) == True) |
        (shared_gene.get("candidate_effect_only", False) == True)
    ) &
    (shared_gene["same_direction"] == True)
].copy()

if candidate_gene.empty:
    candidate_gene = (
        shared_gene[shared_gene["same_direction"] == True]
        .sort_values("combined_gene_score", ascending=False)
        .head(500)
        .copy()
    )

fig = plt.figure(figsize=(7.2, 6.8))
gs = GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.42)

# A. AD vs Migraine logFC scatter
ax = fig.add_subplot(gs[0, 0])

same_df = shared_gene[shared_gene["same_direction"] == True]
opp_df = shared_gene[shared_gene["same_direction"] == False]

ax.scatter(
    opp_df["logFC_AD"],
    opp_df["logFC_Migraine"],
    s=5,
    alpha=0.20,
    color=COL_OPP,
    linewidths=0,
    label="Opposite direction"
)

ax.scatter(
    same_df["logFC_AD"],
    same_df["logFC_Migraine"],
    s=5,
    alpha=0.35,
    color=COL_SHARED,
    linewidths=0,
    label="Same direction"
)

top_label = (
    candidate_gene
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(10)
)

ax.scatter(
    top_label["logFC_AD"],
    top_label["logFC_Migraine"],
    s=18,
    color="#B2182B",
    linewidths=0,
    zorder=5
)

texts = []
for _, r in top_label.iterrows():
    texts.append(
        ax.text(
            r["logFC_AD"],
            r["logFC_Migraine"],
            r["gene"],
            fontsize=6
        )
    )
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"))

ax.axhline(0, color="#333333", lw=0.6)
ax.axvline(0, color="#333333", lw=0.6)
ax.set_xlabel("AD log2 fold-change")
ax.set_ylabel("Migraine log2 fold-change")
ax.set_title("Shared direction of gene regulation")
ax.legend(frameon=False, loc="upper right")
clean_axis(ax)
panel_label(ax, "A")

# B. Top shared genes by combined score
ax = fig.add_subplot(gs[0, 1])

top_bar = (
    candidate_gene
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(15)
    .sort_values("combined_gene_score")
)

ax.barh(
    top_bar["gene"],
    top_bar["combined_gene_score"],
    color=COL_AD,
    height=0.65
)

ax.set_xlabel("Combined AD–migraine score")
ax.set_title("Top shared candidate genes")
clean_axis(ax)
panel_label(ax, "B")

# C. Heatmap of AD/Migraine logFC for top genes
ax = fig.add_subplot(gs[1, 0])

top_heat = (
    candidate_gene
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(20)
)

mat = top_heat[["logFC_AD", "logFC_Migraine"]].values

vmax = np.nanmax(np.abs(mat))
vmax = max(vmax, 0.5)

im = ax.imshow(
    mat,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax
)

ax.set_yticks(np.arange(top_heat.shape[0]))
ax.set_yticklabels(top_heat["gene"].tolist())
ax.set_xticks([0, 1])
ax.set_xticklabels(["AD", "Migraine"])
ax.set_title("Concordant logFC of shared genes")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log2 fold-change")
panel_label(ax, "C")

# D. Shared candidate genes mapped to data-driven clusters
ax = fig.add_subplot(gs[1, 1])

if "marker_cluster" in candidate_gene.columns:
    cluster_gene_counts = (
        candidate_gene
        .dropna(subset=["marker_cluster"])
        .groupby("marker_cluster", observed=True)
        .size()
        .reindex(clusters)
        .fillna(0)
    )
else:
    cluster_gene_counts = pd.Series(0, index=clusters)

ax.bar(
    cluster_gene_counts.index,
    cluster_gene_counts.values,
    color="#6A51A3",
    width=0.75
)
ax.set_ylabel("Shared candidate genes")
ax.set_xlabel("Data-driven cluster")
ax.set_title("Cluster mapping of shared genes")
ax.tick_params(axis="x", rotation=45)
clean_axis(ax)
panel_label(ax, "D")

save_fig(fig, "Figure2_Shared_gene_level_convergence")

# %% [notebook cell 61]


# %% [notebook cell 62]
# ============================================================
# 8. FIGURE 3 — Shared pathway-level immune biology
# ============================================================

global_pw = pathway_table[
    pathway_table["analysis_level"].astype(str).str.contains("Global shared", case=False, na=False)
].copy()

if global_pw.empty:
    global_pw = pathway_table.copy()

global_pw = global_pw.sort_values(["padj", "overlap_n"], ascending=[True, False])

fig = plt.figure(figsize=(7.2, 6.8))
gs = GridSpec(2, 2, figure=fig, wspace=0.48, hspace=0.45)

# A. Pathway enrichment FDR
ax = fig.add_subplot(gs[0, 0])

pw_a = global_pw.sort_values("neglog10_padj", ascending=True).tail(8)
ax.barh(
    pw_a["pathway"],
    pw_a["neglog10_padj"],
    color="#2C7FB8",
    height=0.65
)
ax.set_xlabel("-log10(FDR)")
ax.set_title("Enriched shared immune pathways")
clean_axis(ax)
panel_label(ax, "A")

# B. Overlap genes per pathway
ax = fig.add_subplot(gs[0, 1])

pw_b = global_pw.sort_values("overlap_n", ascending=True).tail(8)
ax.barh(
    pw_b["pathway"],
    pw_b["overlap_n"],
    color="#41AB5D",
    height=0.65
)
ax.set_xlabel("Overlapping shared genes")
ax.set_title("Gene support per pathway")
clean_axis(ax)
panel_label(ax, "B")

# C. Pathway-gene incidence heatmap
ax = fig.add_subplot(gs[1, 0])

sig_pw = global_pw[global_pw["padj"] <= 0.05].copy()
if sig_pw.empty:
    sig_pw = global_pw.head(5).copy()

pathway_gene_pairs = []
for _, r in sig_pw.iterrows():
    genes = str(r["overlap_genes"]).split(";")
    genes = [g for g in genes if g.strip() != ""]
    for g in genes:
        pathway_gene_pairs.append((r["pathway"], g))

pair_df = pd.DataFrame(pathway_gene_pairs, columns=["pathway", "gene"]).drop_duplicates()

if not pair_df.empty:
    genes_order = (
        pair_df["gene"]
        .value_counts()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    pathways_order = sig_pw["pathway"].tolist()

    incidence = pd.DataFrame(
        0,
        index=pathways_order,
        columns=genes_order
    )

    for _, r in pair_df.iterrows():
        incidence.loc[r["pathway"], r["gene"]] = 1

    im = ax.imshow(
        incidence.values,
        aspect="auto",
        cmap=mpl.colors.ListedColormap(["#FFFFFF", "#252525"]),
        vmin=0,
        vmax=1
    )

    ax.set_yticks(np.arange(len(incidence.index)))
    ax.set_yticklabels(incidence.index)
    ax.set_xticks(np.arange(len(incidence.columns)))
    ax.set_xticklabels(incidence.columns, rotation=90)
    ax.set_title("Genes driving enriched pathways")
else:
    ax.text(0.5, 0.5, "No overlap genes available", ha="center", va="center")
    ax.set_axis_off()

panel_label(ax, "C")

# D. LogFC heatmap for pathway-overlap genes
ax = fig.add_subplot(gs[1, 1])

pathway_genes = list(dict.fromkeys(pair_df["gene"].tolist())) if not pair_df.empty else []
pathway_genes = [g for g in pathway_genes if g in shared_gene["gene"].values]

gene_logfc = (
    shared_gene[shared_gene["gene"].isin(pathway_genes)]
    .sort_values("combined_gene_score", ascending=False)
    .drop_duplicates("gene")
    .head(18)
)

if not gene_logfc.empty:
    mat = gene_logfc[["logFC_AD", "logFC_Migraine"]].values
    vmax = max(np.nanmax(np.abs(mat)), 0.5)

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_yticks(np.arange(gene_logfc.shape[0]))
    ax.set_yticklabels(gene_logfc["gene"].tolist())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AD", "Migraine"])
    ax.set_title("Direction of pathway genes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2 fold-change")
else:
    ax.text(0.5, 0.5, "No pathway genes found", ha="center", va="center")
    ax.set_axis_off()

panel_label(ax, "D")

save_fig(fig, "Figure3_Shared_immune_pathway_proof")

# %% [notebook cell 63]
# ============================================================
# 9. OPTIONAL FIGURE 4 — Immune-state abundance contrast
# This is not the main shared result, but useful to show that
# convergence is pathway-level, not cluster-abundance level.
# ============================================================

fig = plt.figure(figsize=(7.2, 3.4))
gs = GridSpec(1, 2, figure=fig, wspace=0.42)

# A. Cluster abundance heatmap
ax = fig.add_subplot(gs[0, 0])

cluster_counts["group_label"] = cluster_counts["dataset"] + "_" + cluster_counts["condition"]

group_order = [
    "AD_GSE181279_Control",
    "AD_GSE181279_AD",
    "Migraine_GSE269117_Control",
    "Migraine_GSE269117_Migraine"
]

pretty_group = {
    "AD_GSE181279_Control": "AD-control",
    "AD_GSE181279_AD": "AD",
    "Migraine_GSE269117_Control": "Migraine-control",
    "Migraine_GSE269117_Migraine": "Migraine"
}

mean_abundance = (
    cluster_counts
    .groupby(["group_label", "cluster_short"], observed=True)["fraction"]
    .mean()
    .reset_index()
)

heat = (
    mean_abundance
    .pivot(index="cluster_short", columns="group_label", values="fraction")
    .fillna(0)
)

valid_groups = [g for g in group_order if g in heat.columns]
heat = heat.loc[clusters, valid_groups]

im = ax.imshow(heat.values, aspect="auto", cmap="magma")
ax.set_yticks(np.arange(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_xticks(np.arange(len(heat.columns)))
ax.set_xticklabels([pretty_group.get(x, x) for x in heat.columns], rotation=45, ha="right")
ax.set_title("Immune-state abundance")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mean fraction")
panel_label(ax, "A")

# B. Cluster-shift scatter
ax = fig.add_subplot(gs[0, 1])

ax.scatter(
    shift_merge["delta_fraction_AD"],
    shift_merge["delta_fraction_Migraine"],
    s=22,
    color="#4D4D4D",
    alpha=0.75,
    linewidths=0
)

ax.axhline(0, color="#333333", lw=0.6)
ax.axvline(0, color="#333333", lw=0.6)

top_shift = (
    shift_merge
    .sort_values("composition_convergence_score", ascending=False)
    .head(8)
)

texts = []
for _, r in top_shift.iterrows():
    texts.append(
        ax.text(
            r["delta_fraction_AD"],
            r["delta_fraction_Migraine"],
            r["cluster_short"],
            fontsize=6
        )
    )
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.4, color="#555555"))

ax.set_xlabel("AD - control fraction")
ax.set_ylabel("Migraine - control fraction")
ax.set_title("Cluster shifts are mostly disease-specific")
clean_axis(ax)
panel_label(ax, "B")

save_fig(fig, "Figure4_Immune_state_abundance_contrast")

# %% [notebook cell 64]
# ============================================================
# 10. Auto-write short result text based on real values
# ============================================================

# Extract main values
n_ad_case = int(table1.loc[table1["Dataset"] == "AD_GSE181279", "Case donors"].iloc[0])
n_ad_ctrl = int(table1.loc[table1["Dataset"] == "AD_GSE181279", "Control donors"].iloc[0])
n_mig_case = int(table1.loc[table1["Dataset"] == "Migraine_GSE269117", "Case donors"].iloc[0])
n_mig_ctrl = int(table1.loc[table1["Dataset"] == "Migraine_GSE269117", "Control donors"].iloc[0])

n_cells_total = int(table1["Cells after QC"].sum())
n_genes = int(table1["Genes retained after filtering"].iloc[0])
n_clusters = len(clusters)

n_shared_candidates = int(candidate_gene.drop_duplicates("gene").shape[0])

top_pathways = (
    table3.sort_values("padj")
    .head(5)["pathway"]
    .tolist()
)

top_pathway_text = ", ".join(top_pathways)

result_text = f"""
Result-section interpretation based on current outputs

Dataset overview:
The final analysis included {n_cells_total:,} PBMCs and {n_genes:,} retained genes after QC and housekeeping/technical-gene filtering. The AD comparison contained {n_ad_case} AD donors and {n_ad_ctrl} controls, while the migraine comparison contained {n_mig_case} migraine donors and {n_mig_ctrl} controls. Unsupervised clustering identified {n_clusters} data-driven immune transcriptional clusters.

Main shared finding:
AD and migraine shared {n_shared_candidates:,} same-direction candidate genes under the current relaxed/effect-size criteria. The strongest shared pathway-level signals were: {top_pathway_text}.

Main interpretation:
The results support pathway-level convergence between AD and migraine, especially inflammatory, Fc-receptor/TYROBP innate, monocyte, chemokine, and cytotoxic T/NK immune programs. Cluster abundance shifts should be interpreted as disease-specific rather than broadly shared.
"""

with open(FINAL_DIR / "Auto_Result_Section_Interpretation.txt", "w") as f:
    f.write(result_text)

print(result_text)

# %% [notebook cell 65]
# ============================================================
# 11. Zip final result-section package
# ============================================================

zip_path = "./Final_Result_Section_AD_Migraine"
shutil.make_archive(zip_path, "zip", FINAL_DIR)

print("Final package saved:")
print(zip_path + ".zip")

print("\nMain figure files:")
for p in sorted(FINAL_FIG_DIR.glob("*.png")):
    print(p)

print("\nMain table files:")
for p in sorted(FINAL_TABLE_DIR.glob("*")):
    print(p)

# %% [notebook cell 66]


# %% [notebook cell 67]


# %% [notebook cell 68]


# %% [notebook cell 69]
# ============================================================
# REPLACEMENT FIGURE 1 — Nature-style dataset + immune atlas
# Better Panel D: cluster-size + top marker summary, not text block
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from adjustText import adjust_text

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)

def panel_label(ax, label):
    ax.text(
        -0.13, 1.08, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left"
    )

def save_final_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FINAL_FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FINAL_FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Prepare UMAP data
# ------------------------------------------------------------
umap_df = pd.DataFrame(
    adata.obsm["X_umap"],
    columns=["UMAP1", "UMAP2"],
    index=adata.obs_names
)

umap_df["dataset"] = adata.obs["dataset"].astype(str).values
umap_df["condition"] = adata.obs["condition"].astype(str).values
umap_df["sample_id"] = adata.obs["sample_id"].astype(str).values
umap_df["cluster_short"] = adata.obs["cluster_short"].astype(str).values
umap_df["group"] = umap_df["dataset"] + "_" + umap_df["condition"]

group_pretty = {
    "AD_GSE181279_AD": "AD",
    "AD_GSE181279_Control": "AD-control",
    "Migraine_GSE269117_Migraine": "Migraine",
    "Migraine_GSE269117_Control": "Migraine-control",
}

group_colors = {
    "AD_GSE181279_AD": "#2C5AA0",
    "AD_GSE181279_Control": "#9DB7E5",
    "Migraine_GSE269117_Migraine": "#D55E00",
    "Migraine_GSE269117_Control": "#F3B27C",
}

clusters = sorted(
    adata.obs["cluster_short"].unique(),
    key=lambda x: int(str(x).replace("C", "")) if str(x).replace("C", "").isdigit() else str(x)
)

cluster_cmap = plt.get_cmap("tab20", len(clusters))
cluster_colors = {cl: cluster_cmap(i) for i, cl in enumerate(clusters)}

plot_umap = umap_df.sample(frac=1.0, random_state=7)

# ------------------------------------------------------------
# Prepare cluster-size and marker table for Panel D
# ------------------------------------------------------------
cluster_size = (
    adata.obs.groupby("cluster_short", observed=True)
    .size()
    .reset_index(name="n_cells")
)

cluster_size["fraction"] = cluster_size["n_cells"] / cluster_size["n_cells"].sum()

# Get top marker per cluster from cluster_marker_df
# If cluster_marker_df is not available, use cluster_signature_map fallback.
if "cluster_marker_df" in globals():
    marker_tmp = cluster_marker_df.copy()
    marker_tmp["cluster_short"] = "C" + marker_tmp["cluster"].astype(str)

    top_marker = (
        marker_tmp[
            (marker_tmp["logFC"] > 0)
        ]
        .sort_values(["cluster_short", "padj", "logFC"], ascending=[True, True, False])
        .groupby("cluster_short", observed=True)
        .head(1)[["cluster_short", "gene"]]
    )
else:
    top_marker = cluster_signature_map.copy()
    top_marker["gene"] = (
        top_marker["cluster_signature"]
        .astype(str)
        .str.replace(r"^C\d+:\s*", "", regex=True)
        .str.split("/")
        .str[0]
    )
    top_marker = top_marker[["cluster_short", "gene"]]

cluster_summary = cluster_size.merge(top_marker, on="cluster_short", how="left")
cluster_summary["gene"] = cluster_summary["gene"].fillna("")
cluster_summary["cluster_num"] = cluster_summary["cluster_short"].str.replace("C", "", regex=False).astype(int)
cluster_summary = cluster_summary.sort_values("cluster_num")

cluster_summary["label"] = cluster_summary["cluster_short"] + "  " + cluster_summary["gene"]

# Save full clean cluster summary as Table S1
cluster_summary.to_csv(
    FINAL_TABLE_DIR / "Supplementary_Table_cluster_size_and_top_marker.csv",
    index=False
)

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 6.6))
gs = GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[1.0, 1.0],
    height_ratios=[1.0, 1.05],
    wspace=0.34,
    hspace=0.40
)

# ------------------------------------------------------------
# A. UMAP by disease group
# ------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])

for g, sub in plot_umap.groupby("group"):
    ax.scatter(
        sub["UMAP1"],
        sub["UMAP2"],
        s=1.0,
        alpha=0.50,
        linewidths=0,
        color=group_colors.get(g, "#999999"),
        label=group_pretty.get(g, g)
    )

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("PBMC profiles by disease group")
ax.legend(frameon=False, markerscale=5, loc="best")
clean_axis(ax)
panel_label(ax, "A")

# ------------------------------------------------------------
# B. UMAP by data-driven clusters
# ------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])

for cl in clusters:
    sub = plot_umap[plot_umap["cluster_short"] == cl]
    ax.scatter(
        sub["UMAP1"],
        sub["UMAP2"],
        s=1.0,
        alpha=0.55,
        linewidths=0,
        color=cluster_colors[cl]
    )

# Label only the largest clusters to avoid clutter
largest_clusters = (
    cluster_summary
    .sort_values("n_cells", ascending=False)
    .head(10)["cluster_short"]
    .tolist()
)

for cl in largest_clusters:
    sub = umap_df[umap_df["cluster_short"] == cl]
    ax.text(
        sub["UMAP1"].median(),
        sub["UMAP2"].median(),
        cl,
        fontsize=6,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.75
        )
    )

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("Data-driven immune clusters")
clean_axis(ax)
panel_label(ax, "B")

# ------------------------------------------------------------
# C. Cells retained per donor
# ------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])

sample_cells = (
    adata.obs.groupby(["dataset", "condition", "sample_id"], observed=True)
    .size()
    .reset_index(name="cells")
    .sort_values(["dataset", "condition", "sample_id"])
)

sample_cells["group"] = sample_cells["dataset"] + "_" + sample_cells["condition"]

x = np.arange(sample_cells.shape[0])
bar_colors = [group_colors.get(g, "#999999") for g in sample_cells["group"]]

ax.bar(
    x,
    sample_cells["cells"],
    color=bar_colors,
    width=0.75,
    edgecolor="none"
)

ax.set_xticks(x)
ax.set_xticklabels(sample_cells["sample_id"], rotation=90)
ax.set_ylabel("Cells after QC")
ax.set_title("Donor-level cell recovery")
clean_axis(ax)
panel_label(ax, "C")

# ------------------------------------------------------------
# D. Cluster size and top data-driven marker
# ------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])

# Show all clusters, but compact and readable
plot_cluster = cluster_summary.sort_values("fraction", ascending=True).copy()

y = np.arange(plot_cluster.shape[0])

bar_colors = [cluster_colors[c] for c in plot_cluster["cluster_short"]]

ax.barh(
    y,
    plot_cluster["fraction"] * 100,
    color=bar_colors,
    height=0.68,
    edgecolor="none"
)

ax.set_yticks(y)
ax.set_yticklabels(plot_cluster["label"], fontsize=5.5)
ax.set_xlabel("Cells in cluster (%)")
ax.set_title("Cluster size and leading marker")
clean_axis(ax)
panel_label(ax, "D")


# ============================================================
# Save figure at 600 dpi
# ============================================================

def save_final_fig(fig, name):
    out_png  = FINAL_FIG_DIR / f"{name}.png"
    out_jpg  = FINAL_FIG_DIR / f"{name}.jpg"
    out_tiff = FINAL_FIG_DIR / f"{name}.tiff"
    out_pdf  = FINAL_FIG_DIR / f"{name}.pdf"

    fig.tight_layout()

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_jpg, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_tiff, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")

    print("Saved 600 dpi files:")
    print(out_png)
    print(out_jpg)
    print(out_tiff)
    print(out_pdf)

    plt.show()



save_final_fig(fig, "Figure1_Dataset_and_data_driven_atlas_REDESIGNED")

# %% [notebook cell 70]


# %% [notebook cell 71]
from PIL import Image
from IPython.display import FileLink, display

# Path to your image
img_path = "results/final_result_section/figures/Figure1_Dataset_and_data_driven_atlas_REDESIGNED.jpg"

# Load image
img = Image.open(img_path)

# Convert to high-quality PNG (recommended for papers)
out_path_png = img_path.replace(".jpg", "_600dpi.png")

# Save with high resolution metadata
img.save(out_path_png, format="PNG", dpi=(600, 600), optimize=True)

print("✅ Saved 600 DPI version at:", out_path_png)

# Trigger download
display(FileLink(out_path_png))

# %% [notebook cell 72]
from IPython.display import FileLink, display

path = "results/final_result_section/figures/Figure1_Dataset_and_data_driven_atlas_REDESIGNED.jpg"

display(FileLink(path))

# %% [notebook cell 73]
from IPython.display import HTML

path = "results/final_result_section/figures/Figure1_Dataset_and_data_driven_atlas_REDESIGNED.jpg"

HTML(f"""
<a href="{path}" download="Figure1.jpg">CLICK HERE TO DOWNLOAD Figure1</a>
""")

# %% [notebook cell 74]


# %% [notebook cell 75]


# %% [notebook cell 76]


# %% [notebook cell 77]


# %% [notebook cell 78]


# %% [notebook cell 79]


# %% [notebook cell 80]


# %% [notebook cell 81]


# %% [notebook cell 82]


# %% [notebook cell 83]


# %% [notebook cell 84]


# %% [notebook cell 85]


# %% [notebook cell 86]


# %% [notebook cell 87]


# %% [notebook cell 88]
# ============================================================
# CELL 5 — FIXED Helper functions for GEO 10x parsing (KAGGLE SAFE)
# ============================================================

import re
import gzip
import pandas as pd
import scipy.io as sio
import scanpy as sc
import anndata as ad
from pathlib import Path


# ----------------------------
# Sample inference
# ----------------------------
def infer_sample_from_path(path):
    """Infer sample ID from GEO file/folder names."""
    s = str(path)

    # AD / NC cohorts
    m = re.search(r"(AD[1-9][0-9]*|NC[1-9][0-9]*)[_\-.]?(?:GEX)?", s, flags=re.I)
    if m:
        return m.group(1).upper()

    # Migraine / clinical cohorts
    m = re.search(r"\b((?:MI|HC|VM|MD)[0-9]+)\b", s, flags=re.I)
    if m:
        return m.group(1).upper()

    m = re.search(r"((?:MI|HC|VM|MD)[0-9]+)", s, flags=re.I)
    if m:
        return m.group(1).upper()

    return None


# ----------------------------
# Condition mapping
# ----------------------------
def condition_from_sample(sample):
    if sample is None:
        return "Unknown"
    s = sample.upper()

    if s.startswith("AD"):
        return "AD"
    if s.startswith("NC"):
        return "Control"
    if s.startswith("MI"):
        return "Migraine"
    if s.startswith("HC"):
        return "Control"
    if s.startswith("VM"):
        return "Vestibular_Migraine"
    if s.startswith("MD"):
        return "Meniere"

    return "Unknown"


# ----------------------------
# gz-safe reader
# ----------------------------
def open_maybe_gz(path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


# ----------------------------
# 10x MTX reader (FIXED)
# ----------------------------
def read_10x_mtx_manual(matrix_path, features_path, barcodes_path, sample, dataset):
    print(f"Reading MTX sample={sample}: {matrix_path.name}")

    with open_maybe_gz(matrix_path) as f:
        X = sio.mmread(f).T.tocsr()

    # FIX: proper tab separator
    features = pd.read_csv(features_path, sep="\t", header=None)
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str).values

    if features.shape[1] >= 2:
        gene_names = features.iloc[:, 1].astype(str).values
        gene_ids = features.iloc[:, 0].astype(str).values
    else:
        gene_names = features.iloc[:, 0].astype(str).values
        gene_ids = gene_names

    obs = pd.DataFrame(index=[f"{sample}_{bc}" for bc in barcodes])
    var = pd.DataFrame(index=gene_names)
    var["gene_ids"] = gene_ids

    a = ad.AnnData(X=X, obs=obs, var=var)
    a.var_names_make_unique()

    a.obs["sample"] = sample
    a.obs["condition"] = condition_from_sample(sample)
    a.obs["dataset"] = dataset

    return a


# ----------------------------
# Find companion files (robust GEO search)
# ----------------------------
def find_companion_file(root, sample, tokens):
    candidates = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        name_low = p.name.lower()
        path_low = str(p).lower()

        if sample and sample.lower() not in path_low:
            continue

        if any(tok in name_low for tok in tokens):
            candidates.append(p)

    return sorted(candidates)[0] if candidates else None


# ----------------------------
# GSE181279 loader (MTX-based)
# ----------------------------
def read_ad_gse181279(root):
    root = Path(root)

    matrix_files = []
    for p in root.rglob("*matrix*.mtx*"):
        path_low = str(p).lower()

        if ("gex" in path_low) or re.search(r"(ad[0-9]+|nc[0-9]+)", path_low):
            matrix_files.append(p)

    matrix_files = sorted(matrix_files)

    adatas = []

    for mtx in matrix_files:

        sample = infer_sample_from_path(mtx)
        if sample is None:
            continue

        if "bcr" in str(mtx).lower() or "tcr" in str(mtx).lower():
            continue

        features = find_companion_file(root, sample, ["features", "genes"])
        barcodes = find_companion_file(root, sample, ["barcodes"])

        # fallback in same directory
        if features is None or barcodes is None:
            same_dir = mtx.parent
            feature_candidates = list(same_dir.glob("*features*.tsv*")) + list(same_dir.glob("*genes*.tsv*"))
            barcode_candidates = list(same_dir.glob("*barcodes*.tsv*"))

            if features is None and feature_candidates:
                features = feature_candidates[0]
            if barcodes is None and barcode_candidates:
                barcodes = barcode_candidates[0]

        if features is None or barcodes is None:
            print(f"Skipping {mtx.name} (missing features/barcodes)")
            continue

        adatas.append(
            read_10x_mtx_manual(
                mtx,
                features,
                barcodes,
                sample,
                dataset="GSE181279_AD"
            )
        )

    if len(adatas) == 0:
        raise RuntimeError("No valid MTX samples found in GSE181279")

    a = ad.concat(
        adatas,
        join="outer",
        label="batch",
        keys=[x.obs["sample"][0] for x in adatas],
        fill_value=0
    )

    a.obs_names_make_unique()
    a.var_names_make_unique()
    return a


# ----------------------------
# GSE269117 loader (H5-based)
# ----------------------------
def read_migraine_gse269117(root):
    root = Path(root)

    h5_files = sorted([p for p in root.rglob("*.h5") if "atac" not in str(p).lower()])

    adatas = []

    for h5 in h5_files:

        sample = infer_sample_from_path(h5)
        if sample is None:
            print(f"Skipping unknown sample: {h5.name}")
            continue

        print(f"Reading H5 sample={sample}: {h5.name}")

        try:
            a = sc.read_10x_h5(str(h5), gex_only=True)
        except TypeError:
            a = sc.read_10x_h5(str(h5))

        a.var_names_make_unique()

        a.obs_names = [f"{sample}_{bc}" for bc in a.obs_names.astype(str)]
        a.obs["sample"] = sample
        a.obs["condition"] = condition_from_sample(sample)
        a.obs["dataset"] = "GSE269117_Migraine"

        adatas.append(a)

    if len(adatas) == 0:
        raise RuntimeError("No valid H5 files found in GSE269117")

    a = ad.concat(
        adatas,
        join="outer",
        label="batch",
        keys=[x.obs["sample"][0] for x in adatas],
        fill_value=0
    )

    a.obs_names_make_unique()
    a.var_names_make_unique()
    return a


print("✅ Helper functions loaded successfully (Kaggle-safe)")

# %% [notebook cell 89]
# ============================================================
# CELL 6 — Read both datasets (FIXED + SAFE)
# ============================================================

ad_ad_raw = read_ad_gse181279(DATA_DIR / "GSE181279_RAW")
ad_mi_raw = read_migraine_gse269117(DATA_DIR / "GSE269117_RAW")

# ----------------------------
# Filter safely
# ----------------------------
ad_ad_raw = ad_ad_raw[ad_ad_raw.obs["condition"].isin(["AD", "Control"])].copy()
ad_mi_raw = ad_mi_raw[ad_mi_raw.obs["condition"].isin(["Migraine", "Control"])].copy()


# ----------------------------
# Safe summary function (prevents crashes)
# ----------------------------
def safe_summary(adata):
    if adata.shape[0] == 0:
        return "EMPTY DATASET"

    df = adata.obs[["sample", "condition"]].drop_duplicates()

    if "sample" not in df.columns:
        return df.to_string(index=False)

    try:
        df = df.sort_values("sample")
    except Exception:
        pass

    return df.to_string(index=False)


# ----------------------------
# Print results
# ----------------------------
print("\nRaw dataset shapes:")
print("AD:", ad_ad_raw.shape)
print(safe_summary(ad_ad_raw))

print("\nMigraine:", ad_mi_raw.shape)
print(safe_summary(ad_mi_raw))

# %% [notebook cell 90]
def read_ad_gse181279(root):
    root = Path(root)

    print("Scanning GEO MTX structure...")

    # ----------------------------
    # STEP 1: find ALL mtx files
    # ----------------------------
    all_mtx = list(root.rglob("*.mtx")) + list(root.rglob("*.mtx.gz"))

    if len(all_mtx) == 0:
        print("❌ No MTX files found. Debug listing:")
        for p in sorted(root.rglob("*"))[:50]:
            print(p)
        raise RuntimeError("No MTX files in dataset")

    print(f"Found {len(all_mtx)} MTX files")

    adatas = []

    # ----------------------------
    # STEP 2: process by directory
    # ----------------------------
    processed_dirs = set()

    for mtx in all_mtx:
        mdir = mtx.parent

        if mdir in processed_dirs:
            continue

        processed_dirs.add(mdir)

        # ----------------------------
        # find companions (robust)
        # ----------------------------
        features = (
            list(mdir.glob("*features*.tsv*")) +
            list(mdir.glob("*genes*.tsv*")) +
            list(mdir.glob("*features*.txt*"))
        )

        barcodes = (
            list(mdir.glob("*barcodes*.tsv*")) +
            list(mdir.glob("*barcodes*.txt*"))
        )

        if not features or not barcodes:
            continue

        features = features[0]
        barcodes = barcodes[0]

        # ----------------------------
        # infer sample (optional)
        # ----------------------------
        sample = infer_sample_from_path(mtx)
        if sample is None:
            sample = mtx.stem[:10]  # fallback safe ID

        try:
            adatas.append(
                read_10x_mtx_manual(
                    mtx,
                    features,
                    barcodes,
                    sample,
                    dataset="GSE181279_AD"
                )
            )
        except Exception as e:
            print(f"Skipping {mtx.name}: {e}")
            continue

    # ----------------------------
    # STEP 3: final check
    # ----------------------------
    if len(adatas) == 0:
        print("\n❌ Debug: folder content")
        for p in sorted(root.rglob("*"))[:100]:
            print(p)
        raise RuntimeError("Still no valid MTX parsed. Dataset structure is different.")

    # ----------------------------
    # STEP 4: merge
    # ----------------------------
    a = ad.concat(
        adatas,
        join="outer",
        label="batch",
        keys=[x.obs["sample"][0] for x in adatas],
        fill_value=0
    )

    a.obs_names_make_unique()
    a.var_names_make_unique()

    print("✅ GSE181279 loaded:", a.shape)

    return a

# %% [notebook cell 91]


# %% [notebook cell 92]
# ============================================================
# RELOAD DATA (SAFE CHECKPOINT CELL)
# ============================================================

# Make sure root paths exist
GSE181279_ROOT = DATA_DIR / "GSE181279_RAW"
GSE269117_ROOT = DATA_DIR / "GSE269117_RAW"

print("Checking datasets...")
print("181279 exists:", GSE181279_ROOT.exists())
print("269117 exists:", GSE269117_ROOT.exists())

# Reload datasets safely
ad_ad_raw = read_ad_gse181279(GSE181279_ROOT)
ad_mi_raw = read_migraine_gse269117(GSE269117_ROOT)

print("\nLoaded:")
print("AD raw:", ad_ad_raw.shape)
print("MI raw:", ad_mi_raw.shape)

# %% [notebook cell 93]


# %% [notebook cell 94]


# %% [notebook cell 95]


# %% [notebook cell 96]
# ============================================================
# CELL 7 — QC + Normalization + PCA/UMAP/Clustering (FIXED)
# ============================================================

def preprocess_scrna(adata, dataset_label):

    adata = adata.copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    print(f"\n{dataset_label} initial shape:", adata.shape)

    # ----------------------------
    # QC metrics (safe MT handling)
    # ----------------------------
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        inplace=True
    )

    # ----------------------------
    # QC filtering (SAFE guards)
    # ----------------------------
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)

    # Only apply MT filter if column exists
    if "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] <= MAX_MT_PERCENT].copy()

    print(f"{dataset_label} after QC:", adata.shape)

    # ----------------------------
    # Save raw counts
    # ----------------------------
    adata.layers["counts"] = adata.X.copy()

    # ----------------------------
    # Normalization
    # ----------------------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.raw = adata

    # ----------------------------
    # HVG (safe upper bound)
    # ----------------------------
    n_hvg = min(3000, adata.n_vars)
    if n_hvg < 200:
        raise RuntimeError(f"Too few genes after QC: {adata.n_vars}")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="seurat"
    )

    # ----------------------------
    # PCA / neighbors / UMAP
    # ----------------------------
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(
        adata,
        n_comps=min(30, adata.n_vars - 1),
        svd_solver="arpack",
        random_state=RANDOM_STATE
    )

    sc.pp.neighbors(
        adata,
        n_neighbors=15,
        n_pcs=min(30, adata.n_vars - 1)
    )

    sc.tl.umap(
        adata,
        min_dist=0.35,
        random_state=RANDOM_STATE
    )

    sc.tl.leiden(
        adata,
        resolution=0.6,
        key_added="leiden"
    )

    return adata


# ----------------------------
# Run preprocessing
# ----------------------------
ad_ad = preprocess_scrna(ad_ad_raw, "GSE181279 AD PBMC")
ad_mi = preprocess_scrna(ad_mi_raw, "GSE269117 Migraine PBMC")

# ----------------------------
# Save outputs
# ----------------------------
ad_ad.write_h5ad(OBJ_DIR / "GSE181279_AD_preprocessed.h5ad")
ad_mi.write_h5ad(OBJ_DIR / "GSE269117_Migraine_preprocessed.h5ad")

print("\n✅ Saved preprocessed AnnData objects")

# %% [notebook cell 97]


# %% [notebook cell 98]


# %% [notebook cell 99]


# %% [notebook cell 100]


# %% [notebook cell 101]


# %% [notebook cell 102]


# %% [notebook cell 103]


# %% [notebook cell 104]
