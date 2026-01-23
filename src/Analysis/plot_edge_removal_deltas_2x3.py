# 3x2 delta figure
# edge removal rates 0/25/50

import os

import matplotlib.pyplot as plt
import pandas as pd

# Global style
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    }
)

# Base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Inputs
CSV_BY_EDGE_REMOVAL = {
    0: os.path.join(
        SCRIPT_DIR,
        "Output_0_categories_agg",
        "f1_matrix_0.csv",
    ),
    25: os.path.join(
        SCRIPT_DIR,
        "Output_25_categories_agg",
        "f1_matrix_25.csv",
    ),
    50: os.path.join(
        SCRIPT_DIR,
        "Output_50_categories_agg",
        "f1_matrix_50.csv",
    ),
}

OUT_FIG = os.path.join(
    SCRIPT_DIR,
    "delta_3x2_edge_removal_rate_0_25_50.png",
)
OUT_FIG_PDF = os.path.join(
    SCRIPT_DIR,
    "delta_3x2_edge_removal_rate_0_25_50.pdf",
)

# Config
BASE_COL = "TRX Only"

CATEGORIES = [
    "Centrality",
    "Cohesion",
    "Community",
    "Proximity",
    "Spectral",
    "Structure",
    "GNN",
]

CLASSIFIERS_ORDER = ["MLP", "XGB", "RF", "LR", "NB", "SVC"]

DPI = 600
LINEWIDTH = 2.0
MARKERSIZE = 6


# Helpers
def load_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_value(df: pd.DataFrame, row: str, col: str) -> float:
    return float(df.loc[row, col])


# Load CSVs
dfs = {er: load_matrix(p) for er, p in CSV_BY_EDGE_REMOVAL.items()}

for er, df in dfs.items():
    if BASE_COL not in df.columns:
        raise KeyError(f"Base column '{BASE_COL}' missing in f1_matrix_{er}.csv")

cats_present = [
    c for c in CATEGORIES
    if all(c in df.columns for df in dfs.values())
]

# Compute deltas
edge_removals = sorted(dfs.keys())

delta_by_clf = {}
for clf in CLASSIFIERS_ORDER:
    rows = []
    for er in edge_removals:
        df = dfs[er]
        base_val = get_value(df, clf, BASE_COL)
        rows.append(
            {cat: get_value(df, clf, cat) - base_val for cat in cats_present}
        )
    delta_by_clf[clf] = pd.DataFrame(rows, index=edge_removals)

# Plot 3x2 layout
fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(18, 20),
    sharex=True,
)
axes = axes.flatten()

for ax, clf in zip(axes, CLASSIFIERS_ORDER):
    d = delta_by_clf[clf]

    for cat in d.columns:
        ax.plot(
            d.index,
            d[cat],
            marker="o",
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=cat,
        )

    ax.axhline(0, linewidth=1.2)
    ax.set_title(clf)
    ax.set_xticks(edge_removals)
    ax.set_xlabel("Edge Removal Rate (%)")
    ax.set_ylabel(r"$\Delta\,F_1$ vs. TRX-only")

# Legend (bottom, compact)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01),
)

plt.tight_layout(rect=[0, 0.12, 1, 1])


plt.savefig(OUT_FIG, dpi=DPI, bbox_inches="tight", pad_inches=0.10)
plt.savefig(OUT_FIG_PDF, bbox_inches="tight", pad_inches=0.10)
plt.show()

print(f"Saved figure to: {OUT_FIG}")
print(f"Saved figure to: {OUT_FIG_PDF}")
