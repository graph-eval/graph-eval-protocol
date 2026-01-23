# plot_category_bar_from_f1_matrix0_no_all.py

import os

import pandas as pd
import matplotlib.pyplot as plt

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

# Output (optional)
OUT_PNG = os.path.join(
    os.path.dirname(CSV_PATH),
    "bar_category_delta_vs_trx_only_no_all.png"
)
OUT_PDF = os.path.join(
    os.path.dirname(CSV_PATH),
    "bar_category_delta_vs_trx_only_no_all.pdf"
)
DPI = 600

# Load
df = pd.read_csv(CSV_PATH, index_col=0)

# normalize index/columns (avoid whitespace issues)
df.index = df.index.astype(str).str.strip()
df.columns = [str(c).strip() for c in df.columns]

# Filter rows (exclude NB)
df_filt = df[~df.index.isin(EXCLUDE_ROWS)].copy()

# sanity checks
missing = [c for c in [BASE_COL, *CATEGORIES] if c not in df_filt.columns]
if missing:
    raise KeyError(
        f"Missing required columns in CSV: {missing}\n"
        f"Available columns: {list(df_filt.columns)}"
    )

# Aggregate: mean per column, then delta vs base mean
col_means = df_filt[[BASE_COL, *CATEGORIES]].mean(numeric_only=True)
base_mean = float(col_means[BASE_COL])

deltas = {cat: float(col_means[cat] - base_mean) for cat in CATEGORIES}

# Plot
deltas = dict(sorted(deltas.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(8, 4.5))
plt.bar(list(deltas.keys()), list(deltas.values()))

plt.ylabel(r"Average $\Delta F_1$ vs. TRX baseline")

plt.xticks(rotation=35, ha="right")
# leave extra space on the left for the y-label
plt.tight_layout(rect=(0.12, 0.0, 1.0, 1.0))

plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.show()

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Base mean (TRX-only):", base_mean)
print("Deltas:", deltas)
