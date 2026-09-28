import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# ADDITIONAL NON-DUPLICATE RESULT
# Figure 3: Immune-state abundance contrast
# Table IV: Top disease-associated cluster-abundance shifts
# Uses real CSV outputs from your pipeline.
# Saves at 600 dpi.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================
# Locate output folder
# ============================================================

possible_table_dirs = [
    Path("./AD_Migraine_Final_Analysis/tables"),
    Path("AD_Migraine_Final_Analysis/tables"),
    Path("./AD_Migraine_Final_Analysis/tables"),
]

TABLE_DIR = None
for p in possible_table_dirs:
    if (p / "Cluster_counts_by_sample.csv").exists() and (p / "Cluster_abundance_shift_AD_vs_Migraine.csv").exists():
        TABLE_DIR = p
        break

if TABLE_DIR is None:
    raise FileNotFoundError(
        "Cannot find Cluster_counts_by_sample.csv and Cluster_abundance_shift_AD_vs_Migraine.csv. "
        "Run the main pipeline first."
    )

OUT_DIR = TABLE_DIR.parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Using table folder:", TABLE_DIR)

# ============================================================
# Load real results
# ============================================================

cluster_counts = pd.read_csv(TABLE_DIR / "Cluster_counts_by_sample.csv")
shift_merge = pd.read_csv(TABLE_DIR / "Cluster_abundance_shift_AD_vs_Migraine.csv")

print("Loaded cluster_counts:", cluster_counts.shape)
print("Loaded shift_merge:", shift_merge.shape)

# ============================================================
# Plot style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)

def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left"
    )

def sort_clusters(x):
    return sorted(
        list(x),
        key=lambda y: int(str(y).replace("C", "")) if str(y).replace("C", "").isdigit() else 999
    )

def save_figure(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.jpg", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    print("Saved:")
    print(FIG_DIR / f"{name}.png")
    print(FIG_DIR / f"{name}.jpg")
    print(FIG_DIR / f"{name}.pdf")
    plt.show()

def extract_top_marker(signature):
    if pd.isna(signature):
        return ""
    s = str(signature)
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.strip().split("/")[0].strip()

# ============================================================
# Prepare labels and cluster order
# ============================================================

def make_group(row):
    dataset = str(row["dataset"])
    condition = str(row["condition"])

    if dataset == "AD_GSE181279" and condition == "Control":
        return "AD-control"
    if dataset == "AD_GSE181279" and condition == "AD":
        return "AD"
    if dataset == "Migraine_GSE269117" and condition == "Control":
        return "Migraine-control"
    if dataset == "Migraine_GSE269117" and condition == "Migraine":
        return "Migraine"
    return dataset + "_" + condition

cluster_counts["Group"] = cluster_counts.apply(make_group, axis=1)

group_order = ["AD-control", "AD", "Migraine-control", "Migraine"]
clusters = sort_clusters(cluster_counts["cluster_short"].unique())

# Use marker/signature from cluster_counts if available
cluster_signature_map = (
    cluster_counts[["cluster_short", "cluster_signature"]]
    .drop_duplicates()
    .copy()
)

cluster_signature_map["Top marker"] = cluster_signature_map["cluster_signature"].apply(extract_top_marker)

# ============================================================
# A. Mean cluster abundance matrix
# ============================================================

mean_abund = (
    cluster_counts
    .groupby(["Group", "cluster_short"], observed=True)["fraction"]
    .mean()
    .reset_index()
)

abundance = (
    mean_abund
    .pivot(index="cluster_short", columns="Group", values="fraction")
    .reindex(index=clusters, columns=group_order)
    .fillna(0)
)

abundance_percent = abundance * 100

# ============================================================
# B. Cluster shift scatter
# ============================================================

required = ["cluster_short", "delta_fraction_AD", "delta_fraction_Migraine"]
missing = [c for c in required if c not in shift_merge.columns]
if missing:
    raise ValueError(f"Missing required columns in shift table: {missing}")

shift_df = shift_merge.dropna(subset=["delta_fraction_AD", "delta_fraction_Migraine"]).copy()

shift_df["AD delta (%)"] = shift_df["delta_fraction_AD"] * 100
shift_df["Migraine delta (%)"] = shift_df["delta_fraction_Migraine"] * 100
shift_df["Magnitude"] = np.sqrt(
    shift_df["AD delta (%)"] ** 2 + shift_df["Migraine delta (%)"] ** 2
)

def direction_class(row):
    ad = row["AD delta (%)"]
    mig = row["Migraine delta (%)"]

    if ad > 0 and mig > 0:
        return "Shared increase"
    if ad < 0 and mig < 0:
        return "Shared decrease"
    return "Disease-specific/opposite"

shift_df["Direction class"] = shift_df.apply(direction_class, axis=1)

# ============================================================
# C. Top cluster shifts table data
# ============================================================

top_shift = (
    shift_df
    .sort_values("Magnitude", ascending=False)
    .head(10)
    .copy()
)

top_shift = top_shift.merge(
    cluster_signature_map[["cluster_short", "Top marker"]],
    on="cluster_short",
    how="left"
)

# Add group-level mean abundances
for g in group_order:
    top_shift[g + " (%)"] = top_shift["cluster_short"].map(abundance_percent[g].to_dict())

table4 = top_shift[[
    "cluster_short",
    "Top marker",
    "AD-control (%)",
    "AD (%)",
    "Migraine-control (%)",
    "Migraine (%)",
    "AD delta (%)",
    "Migraine delta (%)",
    "Direction class"
]].copy()

for col in [
    "AD-control (%)", "AD (%)", "Migraine-control (%)", "Migraine (%)",
    "AD delta (%)", "Migraine delta (%)"
]:
    table4[col] = table4[col].round(2)

table4 = table4.rename(columns={
    "cluster_short": "Cluster",
    "AD-control (%)": "AD-ctrl",
    "AD (%)": "AD",
    "Migraine-control (%)": "Mig-ctrl",
    "Migraine (%)": "Mig",
    "AD delta (%)": "AD Δ",
    "Migraine delta (%)": "Mig Δ",
    "Direction class": "Pattern"
})

table4_path = TABLE_DIR / "Table4_top_cluster_abundance_shifts.csv"
table4.to_csv(table4_path, index=False)

latex_path = TABLE_DIR / "Table4_top_cluster_abundance_shifts.tex"
latex_code = table4.to_latex(
    index=False,
    escape=False,
    caption="Top data-driven immune-state abundance shifts in AD and migraine.",
    label="tab:cluster_abundance_shifts",
    float_format="%.2f"
)
latex_path.write_text(latex_code, encoding="utf-8")

print("\nTable IV: Top cluster-abundance shifts")
print(table4.to_string(index=False))
print("\nSaved:")
print(table4_path)
print(latex_path)

# ============================================================
# Figure 3
# ============================================================

fig = plt.figure(figsize=(17.5, 5.8))
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.05, 1.05, 1.35],
    wspace=0.50
)

# ------------------------------------------------------------
# A. Mean cluster abundance heatmap
# ------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])

im = ax.imshow(
    abundance_percent.values,
    aspect="auto",
    cmap="magma"
)

ax.set_xticks(np.arange(len(group_order)))
ax.set_xticklabels(group_order, rotation=45, ha="right")
ax.set_yticks(np.arange(len(clusters)))
ax.set_yticklabels(clusters, fontsize=7)
ax.set_ylabel("Data-driven cluster")
ax.set_title("Mean immune-state abundance")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
cbar.set_label("Mean cells (%)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

panel_label(ax, "A", x=-0.18)

# ------------------------------------------------------------
# B. AD vs migraine cluster-shift scatter
# ------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])

color_map = {
    "Shared increase": "#D95F02",
    "Shared decrease": "#1B9E77",
    "Disease-specific/opposite": "#636363"
}

for cls, sub in shift_df.groupby("Direction class"):
    ax.scatter(
        sub["AD delta (%)"],
        sub["Migraine delta (%)"],
        s=42,
        color=color_map.get(cls, "gray"),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label=cls
    )

ax.axhline(0, color="black", lw=0.7)
ax.axvline(0, color="black", lw=0.7)

for _, row in shift_df.sort_values("Magnitude", ascending=False).head(8).iterrows():
    ax.annotate(
        row["cluster_short"],
        (row["AD delta (%)"], row["Migraine delta (%)"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.78)
    )

ax.set_xlabel("AD - control fraction (%)")
ax.set_ylabel("Migraine - control fraction (%)")
ax.set_title("Disease-specific cluster shifts")
ax.legend(frameon=False, loc="best")
clean_axis(ax)
panel_label(ax, "B", x=-0.16)

# ------------------------------------------------------------
# C. Top cluster-abundance changes
# ------------------------------------------------------------
ax = fig.add_subplot(gs[0, 2])

plot_top = top_shift.sort_values("Magnitude", ascending=True).copy()
y = np.arange(plot_top.shape[0])
bar_h = 0.35

ax.barh(
    y - bar_h / 2,
    plot_top["AD delta (%)"],
    height=bar_h,
    color="#2459A6",
    edgecolor="none",
    label="AD - control"
)

ax.barh(
    y + bar_h / 2,
    plot_top["Migraine delta (%)"],
    height=bar_h,
    color="#E57200",
    edgecolor="none",
    label="Migraine - control"
)

ax.axvline(0, color="black", lw=0.7)
ax.set_yticks(y)
ax.set_yticklabels(plot_top["cluster_short"])
ax.set_xlabel("Change in cluster fraction (%)")
ax.set_title("Top immune-state abundance changes")
ax.legend(frameon=False, loc="lower right")
clean_axis(ax)
panel_label(ax, "C", x=-0.12)

save_figure(fig, "Figure3_Cluster_Abundance_Contrast_600dpi")
