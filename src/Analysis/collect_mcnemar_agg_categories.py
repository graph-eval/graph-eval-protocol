# -*- coding: utf-8 -*-
"""
collect_mcnemar_agg_categories.py

Category-level McNemar aggregation (analogous to collect_mcnemar_agg_V.0.91.py),
but for category runs, i.e., columns correspond to categories (Centrality, Cohesion, etc.).

What the script does
- reads Results_<VARIANT>_1 … Results_<VARIANT>_10
- for each JSON file, loads the corresponding test predictions from *_testpreds.parquet
- computes McNemar tests between the base setup (TRX-only) and each category variant
(per classifier and seed/run)
- aggregates across seeds/runs by counting how often a category is significantly better or worse
(p ≤ P_VALUE_THRESHOLD and direction determined via n10 vs. n01)
- writes CSV files and two heatmaps (discrete 3-color and intensity)

To adjust (in the config section at the top)
- VARIANT
- RESULTS_DIR_BASE
- ANALYSIS_DIR
- CATEGORY_MAPPING / PREFERRED_COLUMN_ORDER

Note
- direction follows the same definition as in the single-feature script:
    n01 = base correct and variant wrong (variant worse)
    n10 = base wrong and variant correct (variant better)
    direction = +1 if n10 > n01, -1 if n01 > n10, otherwise 0
"""

import os
import re
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import warnings

warnings.filterwarnings(
    "ignore", message="divide by zero encountered in scalar divide"
)

# Global plot style
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)

# Heatmap font config (single source of truth)
TICK_FONTSIZE = 22
CELL_FONTSIZE = 20
CELL_FONTSIZE_SUM = 22  # Σ row a bit larger/bolder

# 1) Configuration
VARIANT = "0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_BASE = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "src",
        "Results_HyPa_fix",
        "Results",
    )
)
RUN_SUFFIXES = [f"_{VARIANT}_{i}" for i in range(1, 11)]  # _0_1 .. _0_10

ANALYSIS_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        f"Output_{VARIANT}_categories_mcnemar_agg",
    )
)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

P_VALUE_THRESHOLD = 0.05

# Raw variant string identifying the base configuration in filenames
BASE_RAW_VARIANT = (
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox"
)
BASE_COLUMN_NAME = "Base Performance (TRX Only)"

CLASSIFIER_ORDER = ["MLP", "LR", "NB", "SVC", "RF", "XGB"]

# 2) Category mapping
CATEGORY_MAPPING = {
    # Base
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Base Performance (TRX Only)",
    # Graph Indicator Categories
    "I1+I2+I3+I4+I5+I6+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Centrality",
    "I7+I8+I9+I10+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Cohesion",
    "I11+I12+I13+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Community",
    # Embedding Categories
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBal+N2VDFS+N2VBFS+DeepWalk": "Proximity",
    "noInd+1Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Spectral",
    "noInd+0Spectral+1ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox": "Structure",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+1GCN+1GAT+1GCL+0Prox": "GNN",
}

# Column order (excluding base, which is only used as reference)
PREFERRED_COLUMN_ORDER = [
    "Centrality",
    "Cohesion",
    "Community",
    "Proximity",
    "Spectral",
    "Structure",
    "GNN",
]


# 3) Helper functions (parsing + prediction loading)
def parse_filename(json_name: str):
    """
    Extracts (classifier, raw_variant) from JSON filename.
    Supports both edgesVar and edges naming patterns.
    """
    stem = os.path.splitext(json_name)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None

    classifier = parts[1]

    # Robust: baseXX+<variant> ... _random
    m = re.search(
        r"base\d+\+(.+?)(?:_(?:edgesVar|edges)_?\d+_splitseed\d+)?_random",
        stem,
    )
    if m:
        raw_variant = m.group(1).split("_")[0]
        return classifier, raw_variant

    m = re.search(r"base\d+\+(.+?)_random", stem)
    if m:
        raw_variant = m.group(1).split("_")[0]
        return classifier, raw_variant

    return classifier, None


def find_parquet(base_path_without_ext: str):
    """
    Finds the corresponding parquet file for a JSON file:
    <same prefix>*testpreds.parquet
    """
    directory = os.path.dirname(base_path_without_ext)
    base_name = os.path.basename(base_path_without_ext)

    if not os.path.isdir(directory):
        return None

    for f in os.listdir(directory):
        if f.startswith(base_name) and f.lower().endswith("testpreds.parquet"):
            return os.path.join(directory, f)

    return None


def load_preds(parquet_path: str):
    """
    Loads y_true / y_pred from parquet.
    Flexibly searches for column names containing 'y_true' / 'y_pred'.
    """
    for engine in (None, "fastparquet"):
        try:
            if engine is None:
                df = pd.read_parquet(parquet_path)
            else:
                df = pd.read_parquet(parquet_path, engine=engine)
            break
        except Exception:
            continue
    else:
        return None, None

    y_true_col = next((c for c in df.columns if "y_true" in c.lower()), None)
    y_pred_col = next((c for c in df.columns if "y_pred" in c.lower()), None)

    if y_true_col is None or y_pred_col is None:
        return None, None

    return df[y_true_col].astype(int).values, df[y_pred_col].astype(int).values


def mcnemar_pvalue(y_true, base_pred, var_pred):
    """
    McNemar p-value + discordant counts.
    n01: base correct, var wrong
    n10: base wrong, var correct
    """
    if len(y_true) != len(base_pred) or len(base_pred) != len(var_pred):
        return np.nan, 0, 0

    y_true = np.asarray(y_true)
    base_pred = np.asarray(base_pred)
    var_pred = np.asarray(var_pred)

    n01 = np.logical_and(base_pred == y_true, var_pred != y_true).sum()
    n10 = np.logical_and(base_pred != y_true, var_pred == y_true).sum()

    table = [[0, n01], [n10, 0]]

    try:
        result = mcnemar(table, exact=False, correction=True)
        return float(result.pvalue), int(n01), int(n10)
    except Exception:
        return np.nan, int(n01), int(n10)


def sort_columns(columns, preferred_order):
    cols = [c for c in preferred_order if c in columns]
    rest = [c for c in columns if c not in cols]
    return cols + sorted(rest)


# 4) Collect McNemar results per run
def collect_mcnemar_for_all_runs():
    """
    For each run:
    - collects predictions (y_true, y_pred) per (classifier, raw variant)
    - computes McNemar tests against the base configuration
    - returns two matrices per run:
        pivot_p: p-values [classifier x category]
        pivot_d: direction (+1 better, -1 worse, 0 tie)
    """
    all_p_matrices = []
    all_d_matrices = []

    for run_suffix in RUN_SUFFIXES:
        print(f"Processing run: {run_suffix}")
        results_dir = RESULTS_DIR_BASE + run_suffix

        if not os.path.exists(results_dir):
            print(f"Run directory not found: {results_dir}")
            continue

        records = []
        for root, _, files in os.walk(results_dir):
            if os.path.basename(root).lower() == "archive":
                continue

            for fname in files:
                if not fname.lower().endswith(".json"):
                    continue

                clf, raw_variant = parse_filename(fname)
                if clf is None or raw_variant is None:
                    continue

                full_json = os.path.join(root, fname)
                parquet = find_parquet(full_json[:-5])  # remove .json
                if parquet is None:
                    continue

                y_true, y_pred = load_preds(parquet)
                if y_true is None or y_pred is None:
                    continue

                records.append((clf, raw_variant, y_true, y_pred))

        if not records:
            print(f"No data found for run: {run_suffix}")
            continue

        entries = {
            (clf, raw): {"y_true": yt, "y_pred": yp}
            for clf, raw, yt, yp in records
        }

        mc_entries = []
        for (clf, raw_variant), data in entries.items():
            if raw_variant == BASE_RAW_VARIANT:
                continue

            base_key = (clf, BASE_RAW_VARIANT)
            if base_key not in entries:
                continue

            yt = entries[base_key]["y_true"]
            yp_base = entries[base_key]["y_pred"]
            yp_var = data["y_pred"]

            p, n01, n10 = mcnemar_pvalue(yt, yp_base, yp_var)
            if np.isnan(p):
                continue

            direction = 1 if n10 > n01 else -1 if n01 > n10 else 0

            category = CATEGORY_MAPPING.get(raw_variant, raw_variant)
            if category == BASE_COLUMN_NAME:
                continue  # safety

            mc_entries.append((clf, category, p, direction, n01, n10))

        if not mc_entries:
            print(f"No McNemar calculations for run: {run_suffix}")
            continue

        df = pd.DataFrame(
            mc_entries,
            columns=[
                "Classifier",
                "Category",
                "p_value",
                "direction",
                "n01",
                "n10",
            ],
        )

        pivot_p = df.pivot(
            index="Classifier", columns="Category", values="p_value"
        )
        pivot_d = df.pivot(
            index="Classifier", columns="Category", values="direction"
        )

        pivot_p = pivot_p.reindex(CLASSIFIER_ORDER)
        pivot_d = pivot_d.reindex(CLASSIFIER_ORDER)

        ordered_cols = sort_columns(
            list(pivot_p.columns), PREFERRED_COLUMN_ORDER
        )
        pivot_p = pivot_p[ordered_cols]
        pivot_d = pivot_d[ordered_cols]

        all_p_matrices.append(pivot_p)
        all_d_matrices.append(pivot_d)

    return all_p_matrices, all_d_matrices


# 5) Aggregation across runs
def aggregate_mcnemar_results(all_p_matrices, all_d_matrices):
    """
    Aggregates p-values across runs:
    count_total: number of runs with an available value (not NaN)
    count_sig_better: p<=thr and direction==+1
    count_sig_worse:  p<=thr and direction==-1
    """
    if not all_p_matrices:
        print("No p-values to aggregate found!")
        return None, None, None

    n_classifiers = len(CLASSIFIER_ORDER)
    n_cols = len(PREFERRED_COLUMN_ORDER)

    count_total = np.zeros((n_classifiers, n_cols))
    count_sig_better = np.zeros((n_classifiers, n_cols))
    count_sig_worse = np.zeros((n_classifiers, n_cols))

    for run_idx in range(len(all_p_matrices)):
        p_mat = all_p_matrices[run_idx]
        d_mat = all_d_matrices[run_idx]

        for i, clf in enumerate(CLASSIFIER_ORDER):
            for j, cat in enumerate(PREFERRED_COLUMN_ORDER):
                if (
                    clf in p_mat.index
                    and cat in p_mat.columns
                    and not pd.isna(p_mat.loc[clf, cat])
                ):

                    count_total[i, j] += 1
                    p_val = float(p_mat.loc[clf, cat])
                    d_val = int(d_mat.loc[clf, cat])

                    if p_val <= P_VALUE_THRESHOLD and d_val == 1:
                        count_sig_better[i, j] += 1
                    if p_val <= P_VALUE_THRESHOLD and d_val == -1:
                        count_sig_worse[i, j] += 1

    df_total = pd.DataFrame(
        count_total, index=CLASSIFIER_ORDER, columns=PREFERRED_COLUMN_ORDER
    )
    df_better = pd.DataFrame(
        count_sig_better,
        index=CLASSIFIER_ORDER,
        columns=PREFERRED_COLUMN_ORDER,
    )
    df_worse = pd.DataFrame(
        count_sig_worse, index=CLASSIFIER_ORDER, columns=PREFERRED_COLUMN_ORDER
    )
    return df_total, df_better, df_worse


# 6) Heatmaps
def create_aggregated_heatmap_discrete(df_better, df_worse):
    """
    Discrete 3-color:
    - green: better>worse
    - gray: equal (incl 0/0)
    - red:  worse>better
    Cell text: better/worse, but 0/0 if no sig at all
    """
    n_rows, n_cols = df_better.shape
    color_matrix = np.full((n_rows, n_cols), np.nan)

    for i in range(n_rows):
        for j in range(n_cols):
            b = df_better.iloc[i, j]
            w = df_worse.iloc[i, j]
            if b > w:
                color_matrix[i, j] = 2
            elif w > b:
                color_matrix[i, j] = 0
            else:
                color_matrix[i, j] = 1

    # Bigger figure so the bigger fonts don't feel cramped
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = ListedColormap(["#FF0000", "#CCCCCC", "#00AA00"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    ax.imshow(color_matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        PREFERRED_COLUMN_ORDER, rotation=45, ha="right", fontsize=TICK_FONTSIZE
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_better.index, fontsize=TICK_FONTSIZE)

    for i in range(n_rows):
        for j in range(n_cols):
            b = int(df_better.iloc[i, j])
            w = int(df_worse.iloc[i, j])
            text = "0/0" if (b == 0 and w == 0) else f"{b}/{w}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=(CELL_FONTSIZE_SUM if i == 0 else CELL_FONTSIZE),
                fontweight=("bold" if i == 0 else "normal"),
            )

    # No title (caption in paper/supplement)
    plt.tight_layout()
    return fig


def create_aggregated_heatmap_intensity(df_better, df_worse):
    """
    Intensity-based:
    - gray if equal
    - green intensity scales with (better-worse)
    - red intensity scales with (worse-better)
    """
    n_rows, n_cols = df_better.shape
    colors = np.ones((n_rows, n_cols, 4), dtype=float)

    colors[:, :, :3] = 0.92

    net = (df_better - df_worse).to_numpy(dtype=float)

    # exclude Σ row from scaling
    net_for_scale = net[1:, :]
    max_diff = (
        np.nanmax(np.abs(net_for_scale))
        if np.any(~np.isnan(net_for_scale))
        else 0.0
    )
    max_diff = max_diff if max_diff > 0 else 1.0

    for i in range(n_rows):
        for j in range(n_cols):
            d = net[i, j]
            if np.isnan(d) or d == 0:
                continue

            strength = min(abs(d) / max_diff, 1.0)
            strength = 0.25 + 0.45 * strength  # pastel

            if d > 0:
                colors[i, j, :3] = [
                    0.92 - 0.35 * strength,
                    0.98 - 0.10 * strength,
                    0.92 - 0.35 * strength,
                ]
            else:
                colors[i, j, :3] = [
                    0.98 - 0.10 * strength,
                    0.92 - 0.35 * strength,
                    0.92 - 0.35 * strength,
                ]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(colors, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        PREFERRED_COLUMN_ORDER, rotation=45, ha="right", fontsize=TICK_FONTSIZE
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_better.index, fontsize=TICK_FONTSIZE)

    for i in range(n_rows):
        for j in range(n_cols):
            b = int(df_better.iloc[i, j])
            w = int(df_worse.iloc[i, j])
            text = "0/0" if (b == 0 and w == 0) else f"{b}/{w}"
            fs = CELL_FONTSIZE_SUM if i == 0 else CELL_FONTSIZE
            fw = "bold" if i == 0 else "normal"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=fs,
                fontweight=fw,
            )

    ax.axhline(0.5, color="black", linewidth=1.5)

    # No title (caption in paper/supplement)
    plt.tight_layout()
    return fig


# 7) Main
if __name__ == "__main__":
    print("Starting category-level McNemar aggregation...")

    all_p_matrices, all_d_matrices = collect_mcnemar_for_all_runs()
    if not all_p_matrices:
        raise RuntimeError("No McNemar data found across runs.")

    df_total, df_better, df_worse = aggregate_mcnemar_results(
        all_p_matrices, all_d_matrices
    )

    # Add top summary row across all classifiers (per category)
    summary_label = "Σ (all clfs)"

    sum_better = df_better.sum(axis=0)
    sum_worse = df_worse.sum(axis=0)
    sum_total = df_total.sum(axis=0)

    df_better = pd.concat(
        [pd.DataFrame([sum_better], index=[summary_label]), df_better], axis=0
    )
    df_worse = pd.concat(
        [pd.DataFrame([sum_worse], index=[summary_label]), df_worse], axis=0
    )
    df_total = pd.concat(
        [pd.DataFrame([sum_total], index=[summary_label]), df_total], axis=0
    )

    if df_total is None:
        raise RuntimeError("Aggregation failed (no totals).")

    # Save matrices
    total_path = os.path.join(
        ANALYSIS_DIR, "mcnemar_categories_total_available.csv"
    )
    better_path = os.path.join(
        ANALYSIS_DIR, "mcnemar_categories_counts_better.csv"
    )
    worse_path = os.path.join(
        ANALYSIS_DIR, "mcnemar_categories_counts_worse.csv"
    )
    net_path = os.path.join(ANALYSIS_DIR, "mcnemar_categories_counts_net.csv")

    df_total.to_csv(total_path)
    df_better.to_csv(better_path)
    df_worse.to_csv(worse_path)
    (df_better - df_worse).to_csv(net_path)

    print(f"Saved: {total_path}")
    print(f"Saved: {better_path}")
    print(f"Saved: {worse_path}")
    print(f"Saved: {net_path}")

    # Heatmaps
    fig = create_aggregated_heatmap_discrete(df_better, df_worse)
    hm1 = os.path.join(
        ANALYSIS_DIR,
        f"mcnemar_categories_heatmap_3color_discrete_{VARIANT}.png",
    )
    hm1_pdf = os.path.join(
        ANALYSIS_DIR,
        f"mcnemar_categories_heatmap_3color_discrete_{VARIANT}.pdf",
    )
    fig.savefig(hm1, dpi=300, bbox_inches="tight")
    fig.savefig(hm1_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hm1}")

    fig = create_aggregated_heatmap_intensity(df_better, df_worse)
    hm2 = os.path.join(
        ANALYSIS_DIR, f"mcnemar_categories_heatmap_intensity_{VARIANT}.png"
    )
    hm2_pdf = os.path.join(
        ANALYSIS_DIR, f"mcnemar_categories_heatmap_intensity_{VARIANT}.pdf"
    )
    fig.savefig(hm2, dpi=300, bbox_inches="tight")
    fig.savefig(hm2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hm2}")

    # Quick summary
    total_cells = df_better.size
    cells_with_improvement = int((df_better > 0).sum().sum())
    cells_with_worsening = int((df_worse > 0).sum().sum())
    cells_no_sig = int(((df_better == 0) & (df_worse == 0)).sum().sum())

    print("\n" + "=" * 60)
    print("SUMMARY (Category-level McNemar)")
    print("=" * 60)
    print(f"Total (Classifier×Category) cells: {total_cells}")
    print(
        f"Cells with ≥1 significant improvement: {cells_with_improvement} ({cells_with_improvement/total_cells*100:.1f}%)"
    )
    print(
        f"Cells with ≥1 significant worsening:  {cells_with_worsening} ({cells_with_worsening/total_cells*100:.1f}%)"
    )
    print(
        f"Cells with no significant result:     {cells_no_sig} ({cells_no_sig/total_cells*100:.1f}%)"
    )
    print("Done.")
