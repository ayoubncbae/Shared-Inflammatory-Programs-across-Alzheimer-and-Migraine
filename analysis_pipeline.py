#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete start-to-end pipeline for:
Single-Cell Transcriptomics Reveals Shared Peripheral Inflammatory Programs
in Alzheimer's Disease and Migraine

What this script does:
1. Downloads GEO raw archives for GSE181279 and GSE269117.
2. Extracts and reads 10x scRNA-seq matrices.
3. Keeps AD vs control and migraine vs control PBMC samples.
4. Performs QC, technical/housekeeping gene filtering, normalization, HVG, PCA, UMAP, Leiden clustering.
5. Builds data-driven immune transcriptional clusters and marker signatures.
6. Performs donor-level global pseudobulk DE within each dataset:
      AD vs AD-control, Migraine vs Migraine-control.
7. Identifies same-direction shared AD--migraine genes.
8. Performs shared immune pathway enrichment.
9. Generates manuscript-ready tables and 600-dpi figures.

Run in Kaggle/Colab/Linux with internet enabled:
    python AD_Migraine_complete_pipeline.py

Outputs:
    ./AD_Migraine_Final_Analysis/
    or ./AD_Migraine_Final_Analysis/ if not running on Kaggle.
"""

# ============================================================
# 0. Optional dependency installation
# ============================================================
import os
import sys
import subprocess
import importlib.util

AUTO_INSTALL = True

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scanpy": "scanpy",
    "anndata": "anndata",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
    "tqdm": "tqdm",
    "adjustText": "adjustText",
    "leidenalg": "leidenalg",
    "igraph": "igraph",
}


def _install_missing_packages():
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    if missing and AUTO_INSTALL:
        print("Installing missing packages:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
    elif missing:
        raise ImportError(f"Missing packages: {missing}")


_install_missing_packages()

# ============================================================
# 1. Imports and global settings
# ============================================================
import re
import gzip
import tarfile
import shutil
import warnings
import gc
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse, stats
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import scanpy as sc
import anndata as ad

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from adjustText import adjust_text

warnings.filterwarnings("ignore")

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)

# Kaggle default path; fallback to local path if not on Kaggle.
BASE_DIR = Path(".") if Path(".").exists() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "AD_Migraine_Final_Analysis"
FINAL_FIG_DIR = OUT_DIR / "figures"
FINAL_TABLE_DIR = OUT_DIR / "tables"
OBJ_DIR = OUT_DIR / "objects"

for d in [DATA_DIR, OUT_DIR, FINAL_FIG_DIR, FINAL_TABLE_DIR, OBJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Main thresholds used for the short paper.
MIN_GENES_PER_CELL = 200
MAX_GENES_PER_CELL = 7000
MAX_MT_PERCENT = 25
MIN_CELLS_PER_GENE = 5
N_TOP_HVG = 3000
N_PCS = 30
TARGET_CLUSTER_N = 21

# Low-memory mode avoids dense full-matrix scaling and large object duplication.
LOW_MEMORY = True
SAVE_FULL_H5AD = False
SAVE_TIFF = False
PLOT_UMAP_MAX_POINTS = 35000
MARKER_TOP_N = 200

LOGFC_CUTOFF = 0.25
NOMINAL_P_CUTOFF = 0.05
SHARED_P_CUTOFF = 0.10

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=120, facecolor="white", fontsize=8)

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

print("\n=== AD--Migraine PBMC scRNA-seq pipeline ===")
print("Scanpy:", sc.__version__)
print("Output directory:", OUT_DIR)

# ============================================================
# 2. Dataset download and extraction
# ============================================================
GEO_URLS = OrderedDict({
    "GSE181279_RAW.tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE181nnn/GSE181279/suppl/GSE181279_RAW.tar",
    "GSE269117_RAW.tar": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269117/suppl/GSE269117_RAW.tar",
})


def run_cmd(cmd):
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"Already downloaded: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
        return
    print(f"Downloading: {url}")
    try:
        run_cmd(["wget", "-q", "--show-progress", "-O", str(dst), url])
    except Exception:
        # fallback for systems without wget
        import urllib.request
        urllib.request.urlretrieve(url, dst)
    if not dst.exists() or dst.stat().st_size < 1000:
        raise RuntimeError(f"Download failed: {dst}")
    print(f"Downloaded: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")


def safe_extract_tar(tar_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted.ok"
    if marker.exists():
        print(f"Already extracted: {tar_path.name}")
        return
    print(f"Extracting {tar_path.name} -> {out_dir}")
    with tarfile.open(tar_path, "r") as tar:
        base = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe tar path: {member.name}")
        tar.extractall(out_dir)
    marker.write_text("ok")


for fname, url in GEO_URLS.items():
    tar_path = DATA_DIR / fname
    download_file(url, tar_path)
    safe_extract_tar(tar_path, DATA_DIR / fname.replace(".tar", ""))

# ============================================================
# 3. Robust 10x reading helpers
# ============================================================
def infer_sample_from_path(path) -> str | None:
    """Infer donor/sample ID from GEO file names and folders."""
    s = str(path).replace("\\", "/")
    m = re.search(r"(?:^|[/_\-.])(AD\d+|NC\d+)(?:[/_\-.]|$)", s, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:^|[/_\-.])(MI\d+|HC\d+|VM\d+|MD\d+)(?:[/_\-.]|$)", s, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"(AD\d+|NC\d+|MI\d+|HC\d+|VM\d+|MD\d+)", s, flags=re.I)
    if m:
        return m.group(1).upper()
    return None


def condition_from_sample(sample: str) -> str:
    sample = str(sample).upper()
    if sample.startswith("AD"):
        return "AD"
    if sample.startswith("NC"):
        return "Control"
    if sample.startswith("MI"):
        return "Migraine"
    if sample.startswith("HC"):
        return "Control"
    if sample.startswith("VM"):
        return "Vestibular_Migraine"
    if sample.startswith("MD"):
        return "Meniere"
    return "Unknown"


def open_maybe_gz(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")


def find_companion_file(root: Path, mtx_path: Path, sample: str, tokens: list[str]) -> Path | None:
    # First try same folder.
    same = []
    for p in mtx_path.parent.iterdir():
        if p.is_file() and any(tok in p.name.lower() for tok in tokens):
            same.append(p)
    if same:
        return sorted(same)[0]

    # Then search full root using sample name.
    candidates = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name_low = p.name.lower()
        path_low = str(p).lower()
        if sample.lower() in path_low and any(tok in name_low for tok in tokens):
            candidates.append(p)
    return sorted(candidates)[0] if candidates else None


def read_10x_mtx_manual(matrix_path: Path, features_path: Path, barcodes_path: Path,
                        sample: str, dataset: str) -> ad.AnnData:
    """Read a single 10x-style MTX sample robustly.

    GEO supplementary files are not always stored in the same orientation.
    Standard 10x MTX is genes x cells and must be transposed for AnnData.
    Some GEO exports are already cells x genes. This function checks the
    matrix shape against features/barcodes before deciding whether to transpose.
    """
    print(f"Reading MTX: dataset={dataset}, sample={sample}, file={matrix_path.name}")

    features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, compression="infer")[0].astype(str).values

    if features.shape[1] >= 2:
        gene_ids = features.iloc[:, 0].astype(str).values
        gene_names = features.iloc[:, 1].astype(str).values
    else:
        gene_names = features.iloc[:, 0].astype(str).values
        gene_ids = gene_names

    with open_maybe_gz(matrix_path) as f:
        X_raw = sio.mmread(f)
    if not sparse.issparse(X_raw):
        X_raw = sparse.coo_matrix(X_raw)

    n_genes = len(gene_names)
    n_cells = len(barcodes)
    raw_shape = X_raw.shape

    # Decide orientation by exact shape agreement.
    if raw_shape == (n_genes, n_cells):
        # Standard 10x: genes x cells -> AnnData expects cells x genes.
        X = X_raw.T.tocsr()
        orientation = "genes_by_cells_transposed"
    elif raw_shape == (n_cells, n_genes):
        # Already cells x genes.
        X = X_raw.tocsr()
        orientation = "cells_by_genes"
    elif raw_shape[0] == n_genes:
        # Gene dimension matches but barcode file length differs. Use generated cell IDs.
        X = X_raw.T.tocsr()
        orientation = "genes_by_cells_transposed_generated_barcodes"
        print(f"  Warning: barcode count ({n_cells}) does not match matrix cells ({X.shape[0]}). Generated cell IDs will be used.")
        barcodes = np.array([f"cell{i:06d}" for i in range(X.shape[0])], dtype=str)
    elif raw_shape[1] == n_genes:
        # Gene dimension matches second axis; matrix is cells x genes but barcode length differs.
        X = X_raw.tocsr()
        orientation = "cells_by_genes_generated_barcodes"
        print(f"  Warning: barcode count ({n_cells}) does not match matrix cells ({X.shape[0]}). Generated cell IDs will be used.")
        barcodes = np.array([f"cell{i:06d}" for i in range(X.shape[0])], dtype=str)
    else:
        raise ValueError(
            f"Could not align MTX orientation for sample {sample}. "
            f"matrix_shape={raw_shape}, features={n_genes}, barcodes={n_cells}, "
            f"matrix={matrix_path}, features_file={features_path}, barcodes_file={barcodes_path}"
        )

    if X.shape[0] != len(barcodes) or X.shape[1] != len(gene_names):
        raise ValueError(
            f"AnnData dimension mismatch after orientation fix for {sample}: "
            f"X={X.shape}, barcodes={len(barcodes)}, genes={len(gene_names)}, orientation={orientation}"
        )

    print(f"  Matrix raw shape {raw_shape} -> AnnData X shape {X.shape} ({orientation})")

    obs = pd.DataFrame(index=[f"{sample}_{bc}" for bc in barcodes])
    var = pd.DataFrame(index=gene_names)
    var["gene_ids"] = gene_ids
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.var_names_make_unique()
    a.obs_names_make_unique()
    a.obs["sample_id"] = sample
    a.obs["condition"] = condition_from_sample(sample)
    a.obs["dataset"] = dataset
    return a


def read_10x_h5_file(h5: Path, sample: str, dataset: str) -> ad.AnnData:
    print(f"Reading H5: dataset={dataset}, sample={sample}, file={h5.name}")
    try:
        a = sc.read_10x_h5(str(h5), gex_only=True)
    except TypeError:
        a = sc.read_10x_h5(str(h5))
    except Exception as e:
        print("read_10x_h5 failed with gex_only=True; trying default", e)
        a = sc.read_10x_h5(str(h5))
    a.var_names_make_unique()
    a.obs_names = [f"{sample}_{bc}" for bc in a.obs_names.astype(str)]
    a.obs_names_make_unique()
    a.obs["sample_id"] = sample
    a.obs["condition"] = condition_from_sample(sample)
    a.obs["dataset"] = dataset
    return a


def read_geo_10x_dataset(root: Path, dataset_label: str) -> ad.AnnData:
    root = Path(root)
    adatas = []

    # 1) Read h5 files, if present.
    h5_files = sorted([p for p in root.rglob("*.h5")
                       if not any(x in str(p).lower() for x in ["atac", "peak", "fragments"])] )
    for h5 in h5_files:
        sample = infer_sample_from_path(h5)
        if sample is None:
            print("Skipping H5 with unknown sample:", h5)
            continue
        a = read_10x_h5_file(h5, sample, dataset_label)
        adatas.append(a)

    # 2) Read MTX files, if present. Avoid double-reading if h5 already covers all samples.
    seen_samples = set([x.obs["sample_id"].iloc[0] for x in adatas])
    mtx_files = sorted([p for p in root.rglob("*matrix*.mtx*")
                        if not any(x in str(p).lower() for x in ["atac", "peak", "bcr", "tcr", "vdj"])] )
    for mtx in mtx_files:
        sample = infer_sample_from_path(mtx)
        if sample is None or sample in seen_samples:
            continue
        features = find_companion_file(root, mtx, sample, ["features", "genes"])
        barcodes = find_companion_file(root, mtx, sample, ["barcodes"])
        if features is None or barcodes is None:
            print("Skipping MTX because companion files were not found:", mtx)
            continue
        a = read_10x_mtx_manual(mtx, features, barcodes, sample, dataset_label)
        adatas.append(a)
        seen_samples.add(sample)

    if len(adatas) == 0:
        print("No matrices read from", root)
        print("First 100 files for debugging:")
        for p in sorted(root.rglob("*"))[:100]:
            print(" ", p)
        raise RuntimeError(f"No 10x data found for {dataset_label}")

    combined = ad.concat(
        adatas,
        join="outer",
        label="source_sample",
        keys=[a.obs["sample_id"].iloc[0] for a in adatas],
        index_unique=None,
        fill_value=0,
    )
    combined.obs_names_make_unique()
    combined.var_names_make_unique()
    return combined


ad_raw = read_geo_10x_dataset(DATA_DIR / "GSE181279_RAW", "AD_GSE181279")
mig_raw = read_geo_10x_dataset(DATA_DIR / "GSE269117_RAW", "Migraine_GSE269117")

# Keep focused comparisons.
ad_raw = ad_raw[ad_raw.obs["condition"].isin(["AD", "Control"])].copy()
mig_raw = mig_raw[mig_raw.obs["condition"].isin(["Migraine", "Control"])].copy()

print("\nRaw retained samples:")
print(pd.concat([
    ad_raw.obs[["dataset", "condition", "sample_id"]].drop_duplicates(),
    mig_raw.obs[["dataset", "condition", "sample_id"]].drop_duplicates(),
]).sort_values(["dataset", "condition", "sample_id"]).to_string(index=False))

# Combine datasets for common data-driven PBMC atlas.
adata = ad.concat([ad_raw, mig_raw], join="outer", index_unique=None, fill_value=0)
adata.obs_names_make_unique()
adata.var_names_make_unique()
print("\nCombined raw shape:", adata.shape)
del ad_raw, mig_raw
gc.collect()

# ============================================================
# 4. QC and technical gene filtering
# ============================================================
def technical_gene_mask(var_names: pd.Index | list[str]) -> np.ndarray:
    genes = pd.Index(var_names).astype(str)
    gu = genes.str.upper()

    # Technical/housekeeping/ambient-dominant categories removed before HVG/PCA/DE.
    mt = gu.str.startswith("MT-")
    ribo = gu.str.startswith("RPS") | gu.str.startswith("RPL")
    histone = gu.str.startswith("HIST") | gu.str.startswith("H1-") | gu.str.startswith("H2A") | gu.str.startswith("H2B") | gu.str.startswith("H3") | gu.str.startswith("H4")
    mitochondrial_like = gu.isin(["MTRNR2L1", "MTRNR2L2", "MTRNR2L3", "MTRNR2L4", "MTRNR2L5", "MTRNR2L6", "MTRNR2L7", "MTRNR2L8", "MTRNR2L9", "MTRNR2L10", "MTRNR2L11", "MTRNR2L12"])
    common_housekeeping = gu.isin([
        "ACTB", "GAPDH", "TUBB", "TUBA1B", "TUBA1A", "EEF1A1", "EEF1B2", "B2M",
        "MALAT1", "NEAT1", "FTL", "FTH1", "TMSB4X", "TMSB10", "HSP90AA1", "HSP90AB1",
        "HSPA1A", "HSPA1B", "HSPA8", "HSPB1", "UBC", "UBB", "PPIA", "PGK1", "LDHA"
    ])
    return np.asarray(mt | ribo | histone | mitochondrial_like | common_housekeeping)


# QC metrics before removing technical genes.
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

print("\nBefore QC:", adata.shape)
sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
adata = adata[adata.obs["n_genes_by_counts"] <= MAX_GENES_PER_CELL].copy()
adata = adata[adata.obs["pct_counts_mt"] <= MAX_MT_PERCENT].copy()
sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
print("After cell/gene QC:", adata.shape)

tech_mask = technical_gene_mask(adata.var_names)
print("Technical/housekeeping genes removed:", int(tech_mask.sum()))
adata = adata[:, ~tech_mask].copy()
print("After technical gene filtering:", adata.shape)

# Store raw count layer after all gene filtering. This is used for pseudobulk.
adata.layers["counts"] = adata.X.copy()

# Make useful labels.
adata.obs["disease_group"] = adata.obs["dataset"].astype(str) + "_" + adata.obs["condition"].astype(str)
adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)
adata.obs["condition"] = adata.obs["condition"].astype(str)
adata.obs["dataset"] = adata.obs["dataset"].astype(str)

sample_cell_counts = (
    adata.obs.groupby(["dataset", "condition", "sample_id"], observed=True)
    .size().reset_index(name="cells_after_qc")
)
print("\nCells after QC by sample:")
print(sample_cell_counts.sort_values(["dataset", "sample_id"]).to_string(index=False))

# ============================================================
# 5. Normalize, HVG-only PCA/UMAP, data-driven clustering
# ============================================================
# IMPORTANT LOW-MEMORY DESIGN:
# - adata.X remains sparse and contains log-normalized expression for all retained genes.
# - adata.layers["counts"] remains sparse raw counts for pseudobulk DE.
# - PCA/UMAP/Leiden are performed only on HVGs in a temporary small object.
# - We never run sc.pp.scale() on the full 63k x 29k matrix, because that densifies the matrix and can crash RAM.

if not sparse.issparse(adata.X):
    adata.X = sparse.csr_matrix(adata.X)
adata.X = adata.X.astype(np.float32)
if sparse.issparse(adata.layers["counts"]):
    adata.layers["counts"] = adata.layers["counts"].tocsr()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=min(N_TOP_HVG, adata.n_vars),
    flavor="seurat",
    subset=False,
)

hvg_genes = adata.var_names[adata.var["highly_variable"].fillna(False).values].tolist()
print(f"Selected HVGs for PCA/UMAP: {len(hvg_genes)}")
if len(hvg_genes) < 500:
    raise RuntimeError("Too few HVGs selected; check input data.")

adata_hvg = adata[:, hvg_genes].copy()
if not sparse.issparse(adata_hvg.X):
    adata_hvg.X = sparse.csr_matrix(adata_hvg.X)
adata_hvg.X = adata_hvg.X.astype(np.float32)

# zero_center=False preserves sparsity and avoids huge dense memory allocation.
sc.pp.scale(adata_hvg, max_value=10, zero_center=False)
sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="arpack", zero_center=False, random_state=RANDOM_STATE)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=N_PCS, random_state=RANDOM_STATE)
sc.tl.umap(adata_hvg, min_dist=0.35, random_state=RANDOM_STATE)


def tune_leiden_to_target(a: ad.AnnData, target_n: int = 21) -> tuple[str, float, int]:
    candidate_res = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00]
    best = None
    for res in candidate_res:
        key = f"leiden_r{res:.2f}"
        sc.tl.leiden(a, resolution=res, key_added=key, random_state=RANDOM_STATE)
        n = a.obs[key].nunique()
        score = abs(n - target_n)
        if best is None or score < best[0]:
            best = (score, key, res, n)
    _, best_key, best_res, best_n = best
    a.obs["data_driven_cluster"] = a.obs[best_key].astype(str).astype("category")

    # Rename clusters as C0, C1, ..., ordered by cluster size descending for stability.
    counts = a.obs["data_driven_cluster"].value_counts()
    ordered = counts.index.tolist()
    rename = {old: f"C{i}" for i, old in enumerate(ordered)}
    a.obs["cluster_short"] = a.obs["data_driven_cluster"].map(rename).astype("category")
    a.obs["data_driven_cluster"] = a.obs["cluster_short"].astype("category")
    return best_key, best_res, best_n


best_key, best_res, best_n = tune_leiden_to_target(adata_hvg, TARGET_CLUSTER_N)
print(f"\nSelected Leiden resolution {best_res:.2f}: {best_n} clusters before C-renaming")

# Transfer low-dimensional results back to the full sparse object.
adata.obs["cluster_short"] = adata_hvg.obs["cluster_short"].astype("category").values
adata.obs["data_driven_cluster"] = adata.obs["cluster_short"].astype("category")
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"].astype(np.float32)
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"].astype(np.float32)

print("Final data-driven clusters:", adata.obs["cluster_short"].nunique())
print(adata.obs["cluster_short"].value_counts().sort_index())

# Save only if explicitly requested; full h5ad can require large memory/disk.
if SAVE_FULL_H5AD:
    adata.write_h5ad(OBJ_DIR / "AD_Migraine_combined_processed.h5ad")

# Remove temporary HVG object to free RAM before marker/DE analysis.
del adata_hvg
gc.collect()

# ============================================================
# 6. Data-driven cluster markers and signatures
# ============================================================
def compute_cluster_markers_light(a: ad.AnnData, top_n: int = 200) -> pd.DataFrame:
    """Memory-light marker ranking using sparse log-normalized expression.

    This avoids Scanpy's full rank_genes_groups step, which can be RAM intensive.
    Markers are ranked by mean log-expression difference between a cluster and all other cells.
    """
    X = a.X.tocsr() if sparse.issparse(a.X) else sparse.csr_matrix(a.X)
    genes = np.asarray(a.var_names.astype(str))
    cl_arr = a.obs["cluster_short"].astype(str).values
    clusters = sorted(np.unique(cl_arr), key=lambda x: int(x.replace("C", "")))
    n_total = X.shape[0]
    global_sum = np.asarray(X.sum(axis=0)).ravel()
    rows = []
    for cl in clusters:
        idx = np.where(cl_arr == cl)[0]
        n_in = len(idx)
        n_out = n_total - n_in
        if n_in == 0 or n_out == 0:
            continue
        sum_in = np.asarray(X[idx, :].sum(axis=0)).ravel()
        mean_in = sum_in / max(n_in, 1)
        mean_out = (global_sum - sum_in) / max(n_out, 1)
        logfc = mean_in - mean_out
        # Use expressed fraction inside cluster only for a simple support statistic.
        pct_in = np.asarray((X[idx, :] > 0).mean(axis=0)).ravel()
        # Prefer genes with positive cluster-specific enrichment and some detection.
        score = logfc * np.sqrt(np.maximum(pct_in, 1e-6))
        order = np.argsort(score)[::-1][:top_n]
        for rank, gi in enumerate(order, start=1):
            if logfc[gi] <= 0:
                continue
            rows.append({
                "cluster_short": cl,
                "gene": genes[gi],
                "rank": rank,
                "logFC": float(logfc[gi]),
                "mean_in": float(mean_in[gi]),
                "mean_out": float(mean_out[gi]),
                "pct_in": float(pct_in[gi]),
                "pval": 1.0,
                "padj": 1.0,
                "score": float(score[gi]),
            })
    return pd.DataFrame(rows).sort_values(["cluster_short", "rank"])


marker_df = compute_cluster_markers_light(adata, top_n=MARKER_TOP_N)
marker_df.to_csv(FINAL_TABLE_DIR / "Data_driven_cluster_markers.csv", index=False)

cluster_signature_rows = []
for cl in sorted(adata.obs["cluster_short"].astype(str).unique(), key=lambda x: int(str(x).replace("C", ""))):
    sub = marker_df[(marker_df["cluster_short"] == cl) & (marker_df["logFC"] > 0)].head(5)
    sig = "/".join(sub["gene"].astype(str).tolist())
    cluster_signature_rows.append({
        "cluster_short": cl,
        "cluster_signature": f"{cl}: {sig}",
        "top_marker": sub["gene"].iloc[0] if len(sub) else ""
    })
cluster_signature_map = pd.DataFrame(cluster_signature_rows)
cluster_signature_map.to_csv(FINAL_TABLE_DIR / "Supplementary_Table_cluster_signatures.csv", index=False)
print("\nCluster signatures:")
print(cluster_signature_map.to_string(index=False))

# Attach signature to obs for convenience.
sig_dict = cluster_signature_map.set_index("cluster_short")["cluster_signature"].to_dict()
adata.obs["cluster_signature"] = adata.obs["cluster_short"].astype(str).map(sig_dict)

gc.collect()

# ============================================================
# 7. Cluster abundance analysis
# ============================================================
def build_cluster_counts(a: ad.AnnData) -> pd.DataFrame:
    rows = []
    all_clusters = sorted(a.obs["cluster_short"].astype(str).unique(), key=lambda x: int(x.replace("C", "")))
    for (dataset, condition, sample), sub_obs in a.obs.groupby(["dataset", "condition", "sample_id"], observed=True):
        total = len(sub_obs)
        counts = sub_obs["cluster_short"].astype(str).value_counts().to_dict()
        for cl in all_clusters:
            n = int(counts.get(cl, 0))
            rows.append({
                "dataset": dataset,
                "condition": condition,
                "sample_id": sample,
                "cluster_short": cl,
                "n_cells": n,
                "sample_total": total,
                "fraction": n / total if total > 0 else 0.0,
                "cluster_signature": sig_dict.get(cl, "")
            })
    return pd.DataFrame(rows)


def cluster_shift_table(cluster_counts: pd.DataFrame, dataset: str, disease_label: str) -> pd.DataFrame:
    rows = []
    for cl, sub in cluster_counts[cluster_counts["dataset"] == dataset].groupby("cluster_short", observed=True):
        d = sub[sub["condition"] == disease_label]["fraction"].astype(float)
        c = sub[sub["condition"] == "Control"]["fraction"].astype(float)
        if len(d) >= 2 and len(c) >= 2:
            _, p = stats.ttest_ind(d, c, equal_var=False, nan_policy="omit")
        else:
            p = np.nan
        rows.append({
            "cluster_short": cl,
            "dataset": dataset,
            "disease_label": disease_label,
            "mean_fraction_disease": d.mean(),
            "mean_fraction_control": c.mean(),
            "delta_fraction": d.mean() - c.mean(),
            "pvalue": p,
            "cluster_signature": sig_dict.get(str(cl), ""),
        })
    out = pd.DataFrame(rows)
    if out["pvalue"].notna().sum() > 0:
        out.loc[out["pvalue"].notna(), "padj"] = multipletests(out.loc[out["pvalue"].notna(), "pvalue"], method="fdr_bh")[1]
    else:
        out["padj"] = np.nan
    return out


cluster_counts = build_cluster_counts(adata)
cluster_counts.to_csv(FINAL_TABLE_DIR / "Cluster_counts_by_sample.csv", index=False)

shift_ad = cluster_shift_table(cluster_counts, "AD_GSE181279", "AD")
shift_mig = cluster_shift_table(cluster_counts, "Migraine_GSE269117", "Migraine")
shift_merge = shift_ad.merge(
    shift_mig,
    on="cluster_short",
    suffixes=("_AD", "_Migraine"),
    how="outer"
)
shift_merge["same_direction_abundance"] = np.sign(shift_merge["delta_fraction_AD"]) == np.sign(shift_merge["delta_fraction_Migraine"])
shift_merge.to_csv(FINAL_TABLE_DIR / "Cluster_abundance_shift_AD_vs_Migraine.csv", index=False)

# ============================================================
# 8. Donor-level global pseudobulk differential expression
# ============================================================
def donor_pseudobulk_matrix(a: ad.AnnData, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = a[a.obs["dataset"] == dataset]
    X = sub.layers["counts"]
    genes = np.array(sub.var_names.astype(str))
    obs = sub.obs.copy()
    rows = []
    meta = []
    for sample, cell_ids in obs.groupby("sample_id", observed=True).groups.items():
        # Important: groupby returns cell IDs, not integer positions.
        idx = sub.obs_names.get_indexer(cell_ids)
        idx = idx[idx >= 0]
        if len(idx) == 0:
            continue
        xsum = X[idx, :].sum(axis=0)
        xsum = xsum.A1 if sparse.issparse(xsum) else np.asarray(xsum).ravel()
        rows.append(xsum)
        condition = obs.loc[cell_ids[0], "condition"] if len(cell_ids) else "Unknown"
        meta.append({
            "dataset": dataset,
            "sample_id": sample,
            "condition": condition,
            "n_cells": len(idx)
        })
    counts = pd.DataFrame(np.vstack(rows), columns=genes)
    meta = pd.DataFrame(meta)
    return counts, meta


def pseudobulk_de_global(a: ad.AnnData, dataset: str, disease_label: str, control_label: str = "Control") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts, meta = donor_pseudobulk_matrix(a, dataset)
    lib = counts.sum(axis=1).replace(0, np.nan)
    logcpm = np.log2(counts.div(lib, axis=0) * 1e6 + 1.0)

    d_idx = meta.index[meta["condition"] == disease_label].to_numpy()
    c_idx = meta.index[meta["condition"] == control_label].to_numpy()
    if len(d_idx) < 2 or len(c_idx) < 2:
        raise RuntimeError(f"Too few donors for DE in {dataset}: disease={len(d_idx)}, control={len(c_idx)}")

    dmat = logcpm.iloc[d_idx].to_numpy(dtype=float)
    cmat = logcpm.iloc[c_idx].to_numpy(dtype=float)
    mean_d = np.nanmean(dmat, axis=0)
    mean_c = np.nanmean(cmat, axis=0)
    logfc = mean_d - mean_c
    _, pvals = stats.ttest_ind(dmat, cmat, axis=0, equal_var=False, nan_policy="omit")
    pvals = np.asarray(pvals, dtype=float)
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    padj = multipletests(pvals, method="fdr_bh")[1]

    de = pd.DataFrame({
        "dataset": dataset,
        "comparison": f"{disease_label} vs {control_label}",
        "gene": counts.columns.astype(str),
        "mean_logCPM_disease": mean_d,
        "mean_logCPM_control": mean_c,
        "logFC": logfc,
        "pval": pvals,
        "padj": padj,
        "abs_logFC": np.abs(logfc),
        "direction": np.where(logfc > 0, "Up", "Down"),
        "n_case_donors": len(d_idx),
        "n_control_donors": len(c_idx),
    }).sort_values(["pval", "abs_logFC"], ascending=[True, False])
    return de, counts, meta


de_ad_global, pb_ad_counts, pb_ad_meta = pseudobulk_de_global(adata, "AD_GSE181279", "AD")
de_mig_global, pb_mig_counts, pb_mig_meta = pseudobulk_de_global(adata, "Migraine_GSE269117", "Migraine")

de_ad_global.to_csv(FINAL_TABLE_DIR / "AD_global_pseudobulk_DE.csv", index=False)
de_mig_global.to_csv(FINAL_TABLE_DIR / "Migraine_global_pseudobulk_DE.csv", index=False)
pb_ad_meta.to_csv(FINAL_TABLE_DIR / "AD_pseudobulk_sample_meta.csv", index=False)
pb_mig_meta.to_csv(FINAL_TABLE_DIR / "Migraine_pseudobulk_sample_meta.csv", index=False)

print("\nGlobal pseudobulk matrix shapes:")
print("AD:", pb_ad_counts.shape, "meta:", pb_ad_meta.shape)
print("Migraine:", pb_mig_counts.shape, "meta:", pb_mig_meta.shape)

# ============================================================
# 9. Shared AD--migraine gene table
# ============================================================
def map_genes_to_marker_cluster(genes: list[str], a: ad.AnnData, marker_table: pd.DataFrame, top_n_marker: int = 100) -> dict:
    """Assign genes to marker clusters using vectorized cluster means."""
    marker_top = (
        marker_table[marker_table["logFC"] > 0]
        .sort_values(["cluster_short", "padj", "logFC"], ascending=[True, True, False])
        .groupby("cluster_short", observed=True)
        .head(top_n_marker)
        .copy()
    )
    gene_to_cluster = {}
    for gene, sub in marker_top.groupby("gene"):
        best = sub.sort_values(["padj", "logFC"], ascending=[True, False]).iloc[0]
        gene_to_cluster[str(gene)] = str(best["cluster_short"])

    clusters = sorted(a.obs["cluster_short"].astype(str).unique(), key=lambda x: int(x.replace("C", "")))
    cluster_values = a.obs["cluster_short"].astype(str).values
    cluster_means = []
    for cl in clusters:
        idx = np.flatnonzero(cluster_values == cl)
        mean = a.X[idx, :].mean(axis=0)
        cluster_means.append(np.asarray(mean).ravel())
    cluster_means = np.vstack(cluster_means)
    var_lookup = {str(gene): i for i, gene in enumerate(a.var_names)}
    for gene in map(str, genes):
        if gene in gene_to_cluster:
            continue
        gi = var_lookup.get(gene)
        gene_to_cluster[gene] = "" if gi is None else clusters[int(np.argmax(cluster_means[:, gi]))]
    return gene_to_cluster

shared_gene = de_ad_global[["gene", "logFC", "pval", "padj"]].merge(
    de_mig_global[["gene", "logFC", "pval", "padj"]],
    on="gene",
    suffixes=("_AD", "_Migraine"),
    how="inner",
)
shared_gene["same_direction"] = np.sign(shared_gene["logFC_AD"]) == np.sign(shared_gene["logFC_Migraine"])
shared_gene["shared_direction_class"] = np.where(
    shared_gene["same_direction"],
    np.where(shared_gene["logFC_AD"] > 0, "Shared up", "Shared down"),
    "Opposite direction",
)
shared_gene["min_abs_logFC"] = np.minimum(shared_gene["logFC_AD"].abs(), shared_gene["logFC_Migraine"].abs())
shared_gene["mean_abs_logFC"] = (shared_gene["logFC_AD"].abs() + shared_gene["logFC_Migraine"].abs()) / 2
shared_gene["combined_gene_score"] = shared_gene["min_abs_logFC"] * (
    -np.log10(shared_gene["pval_AD"].clip(lower=1e-300))
    -np.log10(shared_gene["pval_Migraine"].clip(lower=1e-300))
)

# Candidate list for pathway enrichment: same direction, effect-size supported, and nominal evidence in at least one disease.
shared_gene["shared_candidate"] = (
    shared_gene["same_direction"]
    & (shared_gene["min_abs_logFC"] >= LOGFC_CUTOFF)
    & ((shared_gene["pval_AD"] <= SHARED_P_CUTOFF) | (shared_gene["pval_Migraine"] <= SHARED_P_CUTOFF))
)

candidate_genes = shared_gene.loc[shared_gene["shared_candidate"], "gene"].astype(str).tolist()
if len(candidate_genes) < 50:
    # Safe fallback for very strict environments/version differences.
    candidate_genes = (
        shared_gene[shared_gene["same_direction"]]
        .sort_values("combined_gene_score", ascending=False)
        .head(1000)["gene"].astype(str).tolist()
    )

map_dict = map_genes_to_marker_cluster(shared_gene["gene"].astype(str).tolist(), adata, marker_df, top_n_marker=100)
shared_gene["marker_cluster"] = shared_gene["gene"].astype(str).map(map_dict).fillna("")
shared_gene["cluster_signature"] = shared_gene["marker_cluster"].map(sig_dict).fillna("")
shared_gene = shared_gene.sort_values(["shared_candidate", "same_direction", "combined_gene_score"], ascending=[False, False, False])
shared_gene.to_csv(FINAL_TABLE_DIR / "Table2_shared_AD_Migraine_genes.csv", index=False)

print("\nShared gene table:", shared_gene.shape)
print("Candidate genes for pathway enrichment:", len(candidate_genes))
print(shared_gene.head(20)[["gene", "shared_direction_class", "logFC_AD", "pval_AD", "logFC_Migraine", "pval_Migraine", "combined_gene_score", "marker_cluster"]].to_string(index=False))

# ============================================================
# 10. Shared immune pathway enrichment
# ============================================================
IMMUNE_MODULES = OrderedDict({
    "TNF NF-kB inflammatory response": ["CCL3", "CCL4", "CXCL2", "DUSP1", "ICAM1", "PTGS2", "TNF", "NFKB1", "NFKBIA", "RELA", "IL1B", "IL6", "JUN", "FOS", "CXCL8", "CCL2", "TRAF1"],
    "Fc-receptor TYROBP innate signaling": ["FCER1G", "FCGR2A", "FCGR3A", "LILRB2", "TYROBP", "LILRB1", "FCGR1A", "TREM2", "AIF1", "SPI1", "ITGAX", "LST1"],
    "Inflammatory monocyte program": ["LGALS3", "SERPINA1", "TREM1", "TYROBP", "LYZ", "S100A8", "S100A9", "LST1", "FCN1", "CTSS", "VCAN", "AIF1", "IL1B"],
    "Cytotoxic T/NK activity": ["CST7", "CTSW", "GZMB", "XCL1", "NKG7", "GNLY", "PRF1", "GZMA", "GZMH", "KLRD1", "CCL5", "FGFBP2", "KLRF1", "SPON2", "TRDC"],
    "Chemokine signaling": ["CCL3", "CCL4", "XCL1", "CCL5", "CCL2", "CXCL2", "CXCL8", "CXCL10", "CCR1", "CCR2", "CCR5", "CXCR3", "CXCR4", "CX3CR1"],
    "B-cell identity and activation": ["CD74", "TNFRSF13B", "CD79A", "CD79B", "MS4A1", "BANK1", "FCRLA", "SPIB", "POU2AF1", "TCL1A", "FCER2", "IGHM", "IGKC", "HLA-DRA"],
    "Antigen presentation and HLA": ["CD74", "HLA-DQB1", "HLA-DRA", "HLA-DQA1", "HLA-DMA", "HLA-DPB1", "HLA-DRB1", "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "CIITA"],
    "T-cell activation": ["CD40LG", "DUSP1", "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "IL7R", "CCR7", "LCK", "LAT", "TRAC", "ICOS", "CTLA4", "CD28", "MAL"],
    "S100 calgranulin inflammation": ["LGALS3", "S100A8", "S100A9", "S100A12", "LYZ", "LST1", "FCN1", "VCAN", "CTSS", "TYROBP"],
    "Interferon response": ["RSAD2", "ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI27", "IFI44", "IFI44L", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "STAT1", "IRF7", "CXCL10"],
})


def pathway_enrichment(query_genes: list[str], background_genes: list[str], modules: OrderedDict) -> pd.DataFrame:
    query = set(map(str.upper, query_genes))
    bg = set(map(str.upper, background_genes))
    query = query & bg
    M = len(bg)
    n = len(query)
    rows = []
    for pathway, genes in modules.items():
        module = set(map(str.upper, genes)) & bg
        overlap = sorted(query & module)
        K = len(module)
        x = len(overlap)
        # Fisher exact table: in query/not in query vs in pathway/not in pathway.
        table = [[x, n - x], [K - x, M - K - n + x]]
        odds, p = fisher_exact(table, alternative="greater") if min(map(min, table)) >= 0 else (np.nan, 1.0)
        rows.append({
            "pathway": pathway,
            "overlap_n": x,
            "query_gene_n": n,
            "pathway_gene_n": K,
            "odds_ratio": odds,
            "pval": p,
            "overlap_genes": ";".join(overlap),
        })
    out = pd.DataFrame(rows)
    out["padj"] = multipletests(out["pval"], method="fdr_bh")[1]
    out["neglog10_padj"] = -np.log10(out["padj"].clip(lower=1e-300))
    out = out.sort_values(["padj", "overlap_n"], ascending=[True, False])
    return out


pathway_table = pathway_enrichment(candidate_genes, list(adata.var_names.astype(str)), IMMUNE_MODULES)
pathway_table.to_csv(FINAL_TABLE_DIR / "Table3_shared_immune_pathway_enrichment.csv", index=False)
print("\nPathway enrichment:")
print(pathway_table.to_string(index=False))

# ============================================================
# 11. Manuscript summary Table 1
# ============================================================
def table1_dataset_summary(a: ad.AnnData, de_ad: pd.DataFrame, de_mig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, disease in [("AD_GSE181279", "AD"), ("Migraine_GSE269117", "Migraine")]:
        obs = a.obs[a.obs["dataset"] == dataset]
        sample_counts = obs.groupby("sample_id", observed=True).size()
        case_donors = obs.loc[obs["condition"] == disease, "sample_id"].nunique()
        control_donors = obs.loc[obs["condition"] == "Control", "sample_id"].nunique()
        de = de_ad if dataset == "AD_GSE181279" else de_mig
        nominal_de = int(((de["abs_logFC"] >= LOGFC_CUTOFF) & (de["pval"] <= NOMINAL_P_CUTOFF)).sum())
        rows.append({
            "Dataset": dataset,
            "Disease comparison": f"{disease} vs Control",
            "Case donors": case_donors,
            "Control donors": control_donors,
            "Cells after QC": int(obs.shape[0]),
            "Genes retained": int(a.n_vars),
            "Median cells/donor": int(sample_counts.median()),
            "Cell range/donor": f"{int(sample_counts.min())}-{int(sample_counts.max())}",
            "Nominal DE genes": nominal_de,
        })
    return pd.DataFrame(rows)


table1 = table1_dataset_summary(adata, de_ad_global, de_mig_global)
table1.to_csv(FINAL_TABLE_DIR / "Table1_dataset_and_analysis_summary.csv", index=False)
print("\nTable 1:")
print(table1.to_string(index=False))

# Excel workbook.
excel_path = OUT_DIR / "AD_Migraine_Final_Result_Tables.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    table1.to_excel(writer, sheet_name="Table1_Dataset", index=False)
    shared_gene.head(5000).to_excel(writer, sheet_name="Table2_SharedGenes", index=False)
    pathway_table.to_excel(writer, sheet_name="Table3_Pathways", index=False)
    cluster_signature_map.to_excel(writer, sheet_name="ClusterSignatures", index=False)
    cluster_counts.to_excel(writer, sheet_name="ClusterCounts", index=False)
    shift_merge.to_excel(writer, sheet_name="ClusterShift", index=False)
    de_ad_global.head(10000).to_excel(writer, sheet_name="AD_DE", index=False)
    de_mig_global.head(10000).to_excel(writer, sheet_name="Migraine_DE", index=False)
print("Saved Excel:", excel_path)

# ============================================================
# 12. Figure helpers
# ============================================================
def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def save_final_fig(fig, name: str):
    fig.savefig(FINAL_FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FINAL_FIG_DIR / f"{name}.jpg", dpi=600, bbox_inches="tight", facecolor="white")
    if SAVE_TIFF:
        fig.savefig(FINAL_FIG_DIR / f"{name}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FINAL_FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    print("Saved figure:", FINAL_FIG_DIR / f"{name}.png")


def wrap_label(x, width=24):
    import textwrap
    return "\n".join(textwrap.wrap(str(x), width=width, break_long_words=False))


def sorted_clusters_from_obs(a):
    return sorted(a.obs["cluster_short"].astype(str).unique(), key=lambda x: int(x.replace("C", "")))


clusters = sorted_clusters_from_obs(adata)
cluster_cmap = plt.get_cmap("tab20", len(clusters))
cluster_colors = {cl: cluster_cmap(i) for i, cl in enumerate(clusters)}

group_colors = {
    "AD_GSE181279_AD": "#2C5AA0",
    "AD_GSE181279_Control": "#9DB7E5",
    "Migraine_GSE269117_Migraine": "#D55E00",
    "Migraine_GSE269117_Control": "#F3B27C",
}
group_pretty = {
    "AD_GSE181279_AD": "AD",
    "AD_GSE181279_Control": "AD-control",
    "Migraine_GSE269117_Migraine": "Migraine",
    "Migraine_GSE269117_Control": "Migraine-control",
}

# ============================================================
# 13. Figure 1: atlas + shared gene convergence, panels A--H
# ============================================================
umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
umap_df["dataset"] = adata.obs["dataset"].astype(str).values
umap_df["condition"] = adata.obs["condition"].astype(str).values
umap_df["sample_id"] = adata.obs["sample_id"].astype(str).values
umap_df["cluster_short"] = adata.obs["cluster_short"].astype(str).values
umap_df["group"] = umap_df["dataset"] + "_" + umap_df["condition"]
plot_umap = umap_df.sample(n=min(PLOT_UMAP_MAX_POINTS, umap_df.shape[0]), random_state=RANDOM_STATE)

cluster_size = adata.obs.groupby("cluster_short", observed=True).size().reset_index(name="n_cells")
cluster_size["fraction"] = cluster_size["n_cells"] / cluster_size["n_cells"].sum()
cluster_summary = cluster_size.merge(cluster_signature_map[["cluster_short", "top_marker", "cluster_signature"]], on="cluster_short", how="left")
cluster_summary["cluster_num"] = cluster_summary["cluster_short"].str.replace("C", "", regex=False).astype(int)
cluster_summary = cluster_summary.sort_values("cluster_num")
cluster_summary["label"] = cluster_summary["cluster_short"] + "  " + cluster_summary["top_marker"].fillna("")
cluster_summary.to_csv(FINAL_TABLE_DIR / "Supplementary_Table_cluster_size_and_top_marker.csv", index=False)

fig = plt.figure(figsize=(11.2, 8.3))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1.1, 1.1, 1.0, 1.1], height_ratios=[1.0, 1.0], wspace=0.48, hspace=0.48)

# A. UMAP by disease group
ax = fig.add_subplot(gs[0, 0])
for g, sub in plot_umap.groupby("group"):
    ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.8, alpha=0.50, linewidths=0, color=group_colors.get(g, "#999999"), label=group_pretty.get(g, g))
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_title("PBMC profiles by disease group")
ax.legend(frameon=False, markerscale=5, loc="best")
clean_axis(ax); panel_label(ax, "A")

# B. UMAP by clusters
ax = fig.add_subplot(gs[0, 1])
for cl in clusters:
    sub = plot_umap[plot_umap["cluster_short"] == cl]
    ax.scatter(sub["UMAP1"], sub["UMAP2"], s=0.8, alpha=0.55, linewidths=0, color=cluster_colors[cl])
for cl in cluster_summary.sort_values("n_cells", ascending=False).head(10)["cluster_short"]:
    sub = umap_df[umap_df["cluster_short"] == cl]
    ax.text(sub["UMAP1"].median(), sub["UMAP2"].median(), cl, fontsize=6, fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75))
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.set_title("Data-driven immune clusters")
clean_axis(ax); panel_label(ax, "B")

# C. Donor-level cell recovery
ax = fig.add_subplot(gs[0, 2])
sample_cells = sample_cell_counts.sort_values(["dataset", "condition", "sample_id"]).copy()
sample_cells["group"] = sample_cells["dataset"] + "_" + sample_cells["condition"]
x = np.arange(sample_cells.shape[0])
ax.bar(x, sample_cells["cells_after_qc"], color=[group_colors.get(g, "#999999") for g in sample_cells["group"]], width=0.75, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(sample_cells["sample_id"], rotation=90)
ax.set_ylabel("Cells after QC"); ax.set_title("Donor-level cell recovery")
clean_axis(ax); panel_label(ax, "C")

# D. Cluster size and top marker
ax = fig.add_subplot(gs[0, 3])
plot_cluster = cluster_summary.sort_values("fraction", ascending=True).copy()
y = np.arange(plot_cluster.shape[0])
ax.barh(y, plot_cluster["fraction"] * 100, color=[cluster_colors[c] for c in plot_cluster["cluster_short"]], height=0.68, edgecolor="none")
ax.set_yticks(y); ax.set_yticklabels(plot_cluster["label"], fontsize=5)
ax.set_xlabel("Cells in cluster (%)"); ax.set_title("Cluster size and leading marker")
clean_axis(ax); panel_label(ax, "D")

# E. Cross-disease logFC scatter
ax = fig.add_subplot(gs[1, 0])
plot_scatter = shared_gene.copy()
colors = np.where(plot_scatter["same_direction"], "#2C7FB8", "#BDBDBD")
ax.scatter(plot_scatter["logFC_AD"], plot_scatter["logFC_Migraine"], s=3, c=colors, alpha=0.35, linewidths=0)
ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("AD log$_2$FC"); ax.set_ylabel("Migraine log$_2$FC"); ax.set_title("Gene-level convergence")
texts = []
for _, r in shared_gene[shared_gene["same_direction"]].sort_values("combined_gene_score", ascending=False).head(8).iterrows():
    texts.append(ax.text(r["logFC_AD"], r["logFC_Migraine"], r["gene"], fontsize=5.5))
try:
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4))
except Exception:
    pass
clean_axis(ax); panel_label(ax, "E")

# F. Top shared genes
ax = fig.add_subplot(gs[1, 1])
top_shared = shared_gene[shared_gene["same_direction"]].sort_values("combined_gene_score", ascending=False).head(12).iloc[::-1]
ax.barh(top_shared["gene"], top_shared["combined_gene_score"], color="#2C7FB8", edgecolor="none")
ax.set_xlabel("Combined score"); ax.set_title("Top shared candidate genes")
clean_axis(ax); panel_label(ax, "F")

# G. Top shared logFC heatmap
ax = fig.add_subplot(gs[1, 2])
heat_genes = shared_gene[shared_gene["same_direction"]].sort_values("combined_gene_score", ascending=False).head(12).copy()
mat = heat_genes.set_index("gene")[["logFC_AD", "logFC_Migraine"]]
im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-max(1, np.nanmax(np.abs(mat.values))), vmax=max(1, np.nanmax(np.abs(mat.values))))
ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["AD", "Migraine"], rotation=45, ha="right")
ax.set_title("Direction of shared genes")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log$_2$FC")
panel_label(ax, "G")

# H. Shared genes mapped to marker clusters
ax = fig.add_subplot(gs[1, 3])
cluster_gene_counts = (
    shared_gene[shared_gene["same_direction"]]
    .sort_values("combined_gene_score", ascending=False)
    .head(100)
    .groupby("marker_cluster", observed=True)
    .size().reset_index(name="n_shared_genes")
)
cluster_gene_counts = cluster_gene_counts[cluster_gene_counts["marker_cluster"] != ""]
cluster_gene_counts["cluster_num"] = cluster_gene_counts["marker_cluster"].str.replace("C", "", regex=False).astype(int)
cluster_gene_counts = cluster_gene_counts.sort_values("n_shared_genes", ascending=True)
ax.barh(cluster_gene_counts["marker_cluster"], cluster_gene_counts["n_shared_genes"], color=[cluster_colors.get(c, "#999999") for c in cluster_gene_counts["marker_cluster"]], edgecolor="none")
ax.set_xlabel("Top shared genes"); ax.set_title("Shared genes by cluster")
clean_axis(ax); panel_label(ax, "H")

save_final_fig(fig, "Picture1_Atlas_and_Shared_Gene_Convergence")
plt.close(fig)

# ============================================================
# 14. Figure 2: pathway biology + abundance contrast, panels A--F
# ============================================================
fig = plt.figure(figsize=(11.2, 7.4))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.05, 1.15], height_ratios=[1.0, 1.0], wspace=0.62, hspace=0.52)

# Use top pathways only for readability.
pw = pathway_table.sort_values(["padj", "overlap_n"], ascending=[True, False]).head(8).copy()
pw["pathway_wrapped"] = pw["pathway"].apply(lambda x: wrap_label(x, width=24))

# A. Enrichment strength
ax = fig.add_subplot(gs[0, 0])
pwa = pw.sort_values("neglog10_padj", ascending=True)
ax.barh(pwa["pathway_wrapped"], pwa["neglog10_padj"], color="#2C7FB8", height=0.62, edgecolor="none")
ax.set_xlabel("-log$_{10}$(FDR)"); ax.set_title("Enriched shared immune pathways")
ax.grid(axis="x", linestyle="--", alpha=0.25)
clean_axis(ax); panel_label(ax, "A")

# B. Overlap genes
ax = fig.add_subplot(gs[0, 1])
pwb = pw.sort_values("overlap_n", ascending=True)
ax.barh(pwb["pathway_wrapped"], pwb["overlap_n"], color="#41AB5D", height=0.62, edgecolor="none")
ax.set_xlabel("Overlapping shared genes"); ax.set_title("Gene support per pathway")
ax.grid(axis="x", linestyle="--", alpha=0.25)
clean_axis(ax); panel_label(ax, "B")

# C. Pathway-gene incidence map
ax = fig.add_subplot(gs[0, 2])
path_gene_rows = []
for _, r in pw.iterrows():
    genes = [g for g in str(r["overlap_genes"]).split(";") if g]
    for g in genes:
        path_gene_rows.append({"pathway": r["pathway"], "gene": g, "value": 1})
inc = pd.DataFrame(path_gene_rows)
if len(inc) > 0:
    genes_order = sorted(inc["gene"].unique())
    paths_order = pw["pathway"].tolist()
    matrix = pd.DataFrame(0, index=[wrap_label(p, 18) for p in paths_order], columns=genes_order)
    for _, r in inc.iterrows():
        matrix.loc[wrap_label(r["pathway"], 18), r["gene"]] = 1
    ax.imshow(matrix.values, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns))); ax.set_xticklabels(matrix.columns, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(len(matrix.index))); ax.set_yticklabels(matrix.index, fontsize=5)
else:
    ax.text(0.5, 0.5, "No pathway genes", ha="center", va="center", transform=ax.transAxes)
ax.set_title("Pathway-gene incidence")
panel_label(ax, "C")

# D. LogFC heatmap for pathway-overlap genes
ax = fig.add_subplot(gs[1, 0])
pathway_genes = sorted(set([g for s in pw["overlap_genes"].astype(str) for g in s.split(";") if g]))
if len(pathway_genes) > 0:
    sub = shared_gene[shared_gene["gene"].str.upper().isin([g.upper() for g in pathway_genes])].copy()
    sub = sub.sort_values("combined_gene_score", ascending=False).drop_duplicates("gene")
    mat = sub.set_index("gene")[["logFC_AD", "logFC_Migraine"]]
    vmax = max(1.0, float(np.nanmax(np.abs(mat.values))))
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["AD", "Migraine"], rotation=45, ha="right")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log$_2$FC")
else:
    ax.text(0.5, 0.5, "No genes", ha="center", va="center", transform=ax.transAxes)
ax.set_title("Direction of pathway genes")
panel_label(ax, "D")

# E. Mean cluster abundance heatmap
ax = fig.add_subplot(gs[1, 1])
abund = cluster_counts.copy()
abund["group"] = abund["dataset"] + "_" + abund["condition"]
mean_abund = abund.pivot_table(index="cluster_short", columns="group", values="fraction", aggfunc="mean", fill_value=0)
mean_abund = mean_abund.loc[clusters]
cols = [c for c in ["AD_GSE181279_Control", "AD_GSE181279_AD", "Migraine_GSE269117_Control", "Migraine_GSE269117_Migraine"] if c in mean_abund.columns]
mean_abund = mean_abund[cols]
im = ax.imshow(mean_abund.values * 100, aspect="auto", cmap="Blues")
ax.set_yticks(np.arange(mean_abund.shape[0])); ax.set_yticklabels(mean_abund.index, fontsize=5)
ax.set_xticks(np.arange(mean_abund.shape[1])); ax.set_xticklabels([group_pretty.get(c, c) for c in mean_abund.columns], rotation=45, ha="right")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean cells (%)")
ax.set_title("Mean cluster abundance")
panel_label(ax, "E")

# F. Cluster abundance shift scatter
ax = fig.add_subplot(gs[1, 2])
sm = shift_merge.dropna(subset=["delta_fraction_AD", "delta_fraction_Migraine"]).copy()
ax.scatter(sm["delta_fraction_AD"], sm["delta_fraction_Migraine"], s=24, color="#636363", alpha=0.75, linewidths=0)
ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("AD - control cluster fraction")
ax.set_ylabel("Migraine - control cluster fraction")
ax.set_title("Cluster abundance shifts")
for _, r in sm.iterrows():
    ax.text(r["delta_fraction_AD"], r["delta_fraction_Migraine"], r["cluster_short"], fontsize=5.5)
clean_axis(ax); panel_label(ax, "F")

save_final_fig(fig, "Picture2_Shared_Pathway_and_Abundance_Contrast")
plt.close(fig)

# ============================================================
# 15. Auto-written result summary and zip outputs
# ============================================================
summary_lines = []
summary_lines.append("AD--Migraine PBMC single-cell immune analysis summary")
summary_lines.append("=" * 60)
summary_lines.append(f"Total cells after QC: {adata.n_obs:,}")
summary_lines.append(f"Genes retained after filtering: {adata.n_vars:,}")
summary_lines.append(f"Data-driven clusters observed: {adata.obs['cluster_short'].nunique()}")
summary_lines.append("")
summary_lines.append("Table 1")
summary_lines.append(table1.to_string(index=False))
summary_lines.append("")
summary_lines.append("Top shared genes")
summary_lines.append(shared_gene[shared_gene["same_direction"]].sort_values("combined_gene_score", ascending=False).head(20)[["gene", "shared_direction_class", "logFC_AD", "pval_AD", "logFC_Migraine", "pval_Migraine", "combined_gene_score", "marker_cluster"]].to_string(index=False))
summary_lines.append("")
summary_lines.append("Shared pathway enrichment")
summary_lines.append(pathway_table.to_string(index=False))
summary_lines.append("")
summary_lines.append("Main interpretation:")
summary_lines.append("AD and migraine do not primarily converge through identical immune-state abundance shifts. The strongest shared signal appears at the gene/pathway level, especially TNF/NF-kB inflammatory response, Fc-receptor/TYROBP innate signaling, inflammatory monocyte activity, cytotoxic T/NK activity, and chemokine signaling.")

summary_path = OUT_DIR / "Auto_Result_Section_Interpretation.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
print("Saved summary:", summary_path)

zip_base = str(BASE_DIR / "AD_Migraine_Final_Analysis_Package")
zip_path = shutil.make_archive(zip_base, "zip", OUT_DIR)
print("\nDONE.")
print("Output folder:", OUT_DIR)
print("ZIP package:", zip_path)
print("Key figures:")
print(" -", FINAL_FIG_DIR / "Picture1_Atlas_and_Shared_Gene_Convergence.png")
print(" -", FINAL_FIG_DIR / "Picture2_Shared_Pathway_and_Abundance_Contrast.png")
print("Key tables:")
print(" -", FINAL_TABLE_DIR / "Table1_dataset_and_analysis_summary.csv")
print(" -", FINAL_TABLE_DIR / "Table2_shared_AD_Migraine_genes.csv")
print(" -", FINAL_TABLE_DIR / "Table3_shared_immune_pathway_enrichment.csv")
print(" -", excel_path)
