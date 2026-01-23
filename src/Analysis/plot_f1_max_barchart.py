# plot_category_bar_max_f1_with_trx_baseline_zoom.py

import os

import pandas as pd
import matplotlib.pyplot as plt

# Global font size settings (make all labels/ticks 12pt)
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

# Input
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "src",
    "Analysis",
    "Output_0_categories_agg",
    "f1_matrix_0.csv",
)

BASE_COL = "TRX Only"
EXCLUDE_ROWS = {"NB"}  # exclude Naive Bayes

# Categories to plot (explicitly WITHOUT "All")
CATEGORIES = [
    "Centrality",
    "Cohesion",
    "Community",
    "Proximity",
    "Spectral",
    "Structure",
    "GNN",
]

# Output
OUT_PNG = os.path.join(os.path.dirname(CSV_PATH), "bar_category_max_f1.png")
OUT_PDF = os.path.join(os.path.dirname(CSV_PATH), "bar_category_max_f1.pdf")
DPI = 600

# Load
df = pd.read_csv(CSV_PATH, index_col=0)

# Normalize index/columns (avoid whitespace issues)
df.index = df.index.astype(str).str.strip()
df.columns = [str(c).strip() for c in df.columns]

# Filter rows (exclude NB)
df_filt = df[~df.index.isin(EXCLUDE_ROWS)].copy()

# Sanity checks
required = [BASE_COL, *CATEGORIES]
missing = [c for c in required if c not in df_filt.columns]
if missing:
    raise KeyError(
        f"Missing required columns in CSV: {missing}\n"
        f"Available columns: {list(df_filt.columns)}"
    )

# Compute: best TRX-only baseline and per-category max absolute F1
base_series = df_filt[BASE_COL]
max_base = float(base_series.max())
best_base_clf = base_series.idxmax()

max_f1 = {}
best_cat_clf = {}

for cat in CATEGORIES:
    cat_series = df_filt[cat]
    max_f1[cat] = float(cat_series.max())
    best_cat_clf[cat] = cat_series.idxmax()

# Sort (descending by max absolute F1)
max_f1 = dict(sorted(max_f1.items(), key=lambda x: x[1], reverse=True))

# Plot
plt.figure(figsize=(8, 4.5))
plt.bar(list(max_f1.keys()), list(max_f1.values()))

# reference line at best TRX-only baseline
plt.axhline(max_base, linestyle="--", linewidth=1)

# zoom y-axis to make differences visible
ymin = max_base - 0.005
ymax = max(max_f1.values()) + 0.002
plt.ylim(ymin, ymax)

plt.ylabel("Maximum F1-Score")
plt.xlabel("Graph Signal Category")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.show()

# Print details (handy for caption / appendix text)
print("Saved:", OUT_PNG)
print(f"Best TRX-only F1: {max_base:.6f} (best classifier: {best_base_clf})")
print("Maximum F1 per category:")
for cat, v in max_f1.items():
    print(
        f"  {cat}: {v:.6f}\n"
        f"    best classifier for category: {best_cat_clf[cat]}"
    )
