# plot_edge_removal_deltas_f1_no_all_no_75.py
# Edge-removal robustness plot (AVG Top-5 vs TRX-only)
# WITHOUT "All" and WITHOUT 75%

import os

import pandas as pd
import matplotlib.pyplot as plt

# Resolve paths relative to THIS script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILES = {
    0:  os.path.join(SCRIPT_DIR,
                     "Output_0_categories_agg",
                     "f1_matrix_0.csv"
                     ),
    25: os.path.join(SCRIPT_DIR,
                     "Output_25_categories_agg",
                     "f1_matrix_25.csv"
                     ),
    50: os.path.join(SCRIPT_DIR,
                     "Output_50_categories_agg",
                     "f1_matrix_50.csv"
                     ),
}

OUT_FIG = os.path.join(
    SCRIPT_DIR,
    "delta_row_top5_edge_removal_0_25_50.png"
)
OUT_FIG_PDF = os.path.join(
    SCRIPT_DIR,
    "delta_row_top5_edge_removal_0_25_50.pdf"
)

# Config
BASE_COL = "TRX Only"

# no "All"
CATEGORIES = [
    "Centrality",
    "Cohesion",
    "Community",
    "Proximity",
    "Spectral",
    "Structure",
    "GNN",
]

TOP_K = 5
DPI = 300

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})


# Helpers
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_delta_row_topk(df: pd.DataFrame, k: int) -> dict:
    missing = [c for c in [BASE_COL, *CATEGORIES] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    # Top-k classifiers by TRX-only base performance
    top_idx = df[BASE_COL].sort_values(ascending=False).head(k).index
    top = df.loc[top_idx]

    avg_base = float(top[BASE_COL].mean())
    return {cat: float(top[cat].mean() - avg_base) for cat in CATEGORIES}


# Main
def main():
    edge_rates = sorted(CSV_FILES.keys())  # [0, 25, 50]
    delta_series = {cat: [] for cat in CATEGORIES}

    for rate in edge_rates:
        df = load_csv(CSV_FILES[rate])
        deltas = compute_delta_row_topk(df, TOP_K)
        for cat in CATEGORIES:
            delta_series[cat].append(deltas[cat])

    # Plot
    plt.figure(figsize=(10, 6))
    for cat in CATEGORIES:
        plt.plot(edge_rates, delta_series[cat], marker="o", label=cat)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Edge Removal Rate (%)")
    plt.ylabel(r"$\Delta\,F_1$ vs. TRX-only")
    plt.xticks(edge_rates)

    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(OUT_FIG, dpi=DPI, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.show()

    print(f"Saved plot to: {OUT_FIG}")
    print(f"Saved plot to: {OUT_FIG_PDF}")


if __name__ == "__main__":
    main()
