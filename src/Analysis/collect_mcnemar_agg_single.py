# -*- coding: utf-8 -*-
"""
Aggregates McNemar results over 10 runs (_0_1 ... _0_10):
For each (classifier, feature) combination, it counts how often p ≤ 0.05 occurred.

Output: C:\Experiments\Analysis\Output_0_agg
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


warnings.filterwarnings(
    "ignore", message="divide by zero encountered in scalar divide"
)


# 1) Configuration
VARIANT = "0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR_BASE = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "src",
        "Results_HyPa_fix_Singles",
        "Results",
    )
)

RUN_SUFFIXES = [f"_{VARIANT}_{i}" for i in range(1, 11)]
ANALYSIS_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        f"Output_{VARIANT}_agg",
    )
)

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Raw variant string describing the base variant used in filenames
BASE_RAW_VARIANT = (
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox"
)
BASE_COLUMN_NAME = "TRX Only"
P_VALUE_THRESHOLD = 0.05


# 2) Mapping from filename variants to human-readable column names
COLUMN_MAPPING = {
    # Base
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "TRX Only",
    # Centrality
    "I1+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "DEGREE_CENTRALITY",
    "I2+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "IN_OUT_DEGREE",
    "I3+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "PAGERANK",
    "I4+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "BETWEENNESS_CENTRALITY",
    "I5+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "EIGENVECTOR_CENTRALITY",
    "I6+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "CLOSENESS_CENTRALITY",
    # Cohesion
    "I7+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "CLUSTERING_COEFFICIENT",
    "I8+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "CORE_NUMBER",
    "I9+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "TRIANGLES",
    "I10+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "SQUARE_CLUSTERING",
    # Community
    "I11+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "COMMUNITY_LOUVAIN",
    "I12+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "COMMUNITY_LEIDEN",
    "I13+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "COMMUNITY_INFOMAP",
    # Proximity Embeddings
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+DeepWalk": "DeepWalk",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBFS": "node2vec-BFS",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VDFS": "node2vec-DFS",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBal": "node2vec-bal",
    # Spectral Embeddings
    "noInd+1Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Spectral",
    # Structural Embeddings
    "noInd+0Spectral+1ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "ffstruc2vec",
    "noInd+0Spectral+0ff2Vec+0Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox": "Role2Vec",
    "noInd+0Spectral+0ff2Vec+1Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Graphwave",
    # GNN Embeddings
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+1GCN+0GAT+0GCL+0Prox": "GCN",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+1GAT+0GCL+0Prox": "GAT",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+1GCL+0Prox": "GCL",
    # Combination variants (optional)
    "I1+I2+I3+I4+I5+I6+I7+I8+I9+I10+I11+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "All GraphIndicators",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBal+N2VDFS+N2VBFS+DeepWalk": "All Proximity Embeddings",
    "noInd+1Spectral+0ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox": "All Structural Embeddings",
    "I1+I2+I3+I4+I5+I6+I7+I8+I9+I10+I11+1Spectral+0ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+N2VBal+N2VDFS+N2VBFS+DeepWalk": "All",
}


ORIGINAL_PREFERRED_ORDER = [
    "TRX Only",
    "DEGREE_CENTRALITY",
    "IN_OUT_DEGREE",
    "PAGERANK",
    "BETWEENNESS_CENTRALITY",
    "EIGENVECTOR_CENTRALITY",
    "CLOSENESS_CENTRALITY",
    "CLUSTERING_COEFFICIENT",
    "CORE_NUMBER",
    "TRIANGLES",
    "SQUARE_CLUSTERING",
    "COMMUNITY_LOUVAIN",
    "COMMUNITY_LEIDEN",
    "COMMUNITY_INFOMAP",
    "DeepWalk",
    "node2vec-BFS",
    "node2vec-DFS",
    "node2vec-bal",
    "Spectral",
    "ffstruc2vec",
    "Role2Vec",
    "Graphwave",
    "GCN",
    "GAT",
    "GCL",
    "All GraphIndicators",
    "All Proximity Embeddings",
    "All Structural Embeddings",
    "All",
]


PREFERRED_COLUMN_ORDER = [
    "DEGREE_CENTRALITY",
    "IN_OUT_DEGREE",
    "PAGERANK",
    "BETWEENNESS_CENTRALITY",
    "EIGENVECTOR_CENTRALITY",
    "CLOSENESS_CENTRALITY",
    "CLUSTERING_COEFFICIENT",
    "CORE_NUMBER",
    "TRIANGLES",
    "SQUARE_CLUSTERING",
    "COMMUNITY_LOUVAIN",
    "COMMUNITY_LEIDEN",
    "COMMUNITY_INFOMAP",
    "DeepWalk",
    "node2vec-BFS",
    "node2vec-DFS",
    "node2vec-bal",
    "Spectral",
    "ffstruc2vec",
    "Role2Vec",
    "Graphwave",
    "GCN",
    "GAT",
    "GCL",
]

classifier_order = ["MLP", "LR", "NB", "SVC", "RF", "XGB"]


# 3) Helper functions
def parse_filename(json_name):
    """Extracts classifier and variant from filename."""
    stem = os.path.splitext(json_name)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None, None

    classifier = parts[1]

    m = re.search(
        r"base\d+\+(.+?)(?:_edgesVar_\d+_splitseed\d+)?_random", stem
    )
    if not m:
        return classifier, None

    raw_variant = m.group(1).split("_")[0]
    return classifier, raw_variant


def find_parquet(base_path_without_ext):
    """Finds matching parquet file for JSON."""
    directory = os.path.dirname(base_path_without_ext)
    base_name = os.path.basename(base_path_without_ext)

    for f in os.listdir(directory):
        if f.startswith(base_name) and f.lower().endswith("testpreds.parquet"):
            return os.path.join(directory, f)

    return None


def load_preds(parquet_path):
    """Loads predictions from parquet file."""
    # Try different engines
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

    # Flexible column search
    y_true_col = next((c for c in df.columns if "y_true" in c.lower()), None)
    y_pred_col = next((c for c in df.columns if "y_pred" in c.lower()), None)

    if y_true_col is None or y_pred_col is None:
        return None, None

    return df[y_true_col].astype(int).values, df[y_pred_col].astype(int).values


def mcnemar_pvalue(y_true, base_pred, var_pred):
    """Calculates McNemar p-value and direction."""
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
        return result.pvalue, int(n01), int(n10)
    except Exception:
        return np.nan, int(n01), int(n10)


def sort_columns_by_category(columns, preferred_order):
    """Sorts columns by preferred order."""
    cols = [c for c in preferred_order if c in columns]
    rest = [c for c in columns if c not in cols]
    return cols + sorted(rest)


def draw_category_separators_and_labels(ax, columns, y_pos=1.04):
    """Draws category separators and labels in English."""
    categories = {
        "Centrality": [
            "DEGREE_CENTRALITY",
            "IN_OUT_DEGREE",
            "PAGERANK",
            "BETWEENNESS_CENTRALITY",
            "EIGENVECTOR_CENTRALITY",
            "CLOSENESS_CENTRALITY",
        ],
        "Cohesion": [
            "CLUSTERING_COEFFICIENT",
            "CORE_NUMBER",
            "TRIANGLES",
            "SQUARE_CLUSTERING",
        ],
        "Community": [
            "COMMUNITY_LOUVAIN",
            "COMMUNITY_LEIDEN",
            "COMMUNITY_INFOMAP",
        ],
        "Proximity": [
            "DeepWalk",
            "node2vec-BFS",
            "node2vec-DFS",
            "node2vec-bal",
        ],
        "Spectral": ["Spectral"],
        "Structure": ["ffstruc2vec", "Graphwave", "Role2Vec"],
        "GNN": ["GCN", "GAT", "GCL"],
    }

    trans = ax.get_xaxis_transform()
    group_indices = {name: [] for name in categories.keys()}

    for idx, col in enumerate(columns):
        for cat, cols in categories.items():
            if col in cols:
                group_indices[cat].append(idx)

    for cat, idxs in group_indices.items():
        if not idxs:
            continue
        start = min(idxs)
        end = max(idxs)

        ax.axvline(start - 0.5, color="black", linewidth=1.0)
        ax.axvline(end + 0.5, color="black", linewidth=1.0)

        ax.text(
            (start + end) / 2.0,
            y_pos,
            cat,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=18,
            clip_on=False,
        )


# 4) Main function: Collect McNemar results for all runs
def collect_mcnemar_for_all_runs():
    """Collects p-values and directions for all 10 runs."""

    # Lists to collect results
    all_p_matrices = []  # List of DataFrames with p-values per run
    all_d_matrices = []  # List of DataFrames with directions per run

    for run_suffix in RUN_SUFFIXES:
        print(f"Processing run: {run_suffix}")

        RESULTS_DIR = RESULTS_DIR_BASE + run_suffix

        if not os.path.exists(RESULTS_DIR):
            print(f"Run directory not found: {RESULTS_DIR}")
            continue

        # Collect all predictions for this run
        records = []
        for root, _, files in os.walk(RESULTS_DIR):
            if os.path.basename(root).lower() == "archive":
                continue

            for fname in files:
                if not fname.lower().endswith(".json"):
                    continue

                classifier, raw_variant = parse_filename(fname)
                if classifier is None or raw_variant is None:
                    continue

                full_json = os.path.join(root, fname)
                parquet = find_parquet(full_json[:-5])

                if parquet is None:
                    continue

                y_true, y_pred = load_preds(parquet)
                if y_true is None or y_pred is None:
                    continue

                records.append((classifier, raw_variant, y_true, y_pred))

        if not records:
            print(f"No data found for run: {run_suffix}")
            continue

        # Create dictionary for quick access
        entries = {
            (clf, raw): {"y_true": yt, "y_pred": yp}
            for clf, raw, yt, yp in records
        }

        # Calculate McNemar for all variants compared to base
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
            mc_entries.append((clf, raw_variant, p, direction, n01, n10))

        if not mc_entries:
            print(f"No McNemar calculations for run: {run_suffix}")
            continue

        # Create DataFrames for this run
        df = pd.DataFrame(
            mc_entries,
            columns=[
                "Classifier",
                "RawVariant",
                "p_value",
                "direction",
                "n01",
                "n10",
            ],
        )

        # Pivot tables
        pivot_p = df.pivot(
            index="Classifier", columns="RawVariant", values="p_value"
        )
        pivot_d = df.pivot(
            index="Classifier", columns="RawVariant", values="direction"
        )

        # Add base column
        for clf in classifier_order:
            if clf not in pivot_p.index:
                pivot_p.loc[clf, :] = np.nan
                pivot_d.loc[clf, :] = np.nan

            if BASE_RAW_VARIANT not in pivot_p.columns:
                pivot_p[BASE_RAW_VARIANT] = np.nan
                pivot_d[BASE_RAW_VARIANT] = 0

        # Classifier order
        pivot_p = pivot_p.reindex(classifier_order)
        pivot_d = pivot_d.reindex(classifier_order)

        # Raw variants → readable names
        pivot_p = pivot_p.rename(columns=lambda c: COLUMN_MAPPING.get(c, c))
        pivot_d = pivot_d.rename(columns=lambda c: COLUMN_MAPPING.get(c, c))

        # Remove unwanted columns before sorting
        columns_to_remove = [
            "TRX Only",
            "All GraphIndicators",
            "All Proximity Embeddings",
            "All Structural Embeddings",
            "All",
        ]

        # Keep only columns that exist in the DataFrame
        existing_columns = set(pivot_p.columns)
        columns_to_remove = [
            col for col in columns_to_remove if col in existing_columns
        ]

        if columns_to_remove:
            pivot_p = pivot_p.drop(columns=columns_to_remove, errors="ignore")
            pivot_d = pivot_d.drop(columns=columns_to_remove, errors="ignore")

        # Sort columns
        ordered_cols = sort_columns_by_category(
            list(pivot_p.columns), PREFERRED_COLUMN_ORDER
        )
        pivot_p = pivot_p[ordered_cols]
        pivot_d = pivot_d[ordered_cols]

        # Save matrices for this run
        all_p_matrices.append(pivot_p)
        all_d_matrices.append(pivot_d)

    return all_p_matrices, all_d_matrices


# 5) Aggregate results over all runs
def aggregate_mcnemar_results(all_p_matrices, all_d_matrices):
    """Aggregates p-values over all runs: counts significant results."""

    if not all_p_matrices:
        print("No p-values to aggregate found!")
        return None, None, None

    # Initialize counting matrices
    n_classifiers = len(classifier_order)
    n_features = len(PREFERRED_COLUMN_ORDER)

    # Matrices for counting
    count_total = np.zeros(
        (n_classifiers, n_features)
    )  # Number of available runs
    count_sig_better = np.zeros(
        (n_classifiers, n_features)
    )  # p <= 0.05 & direction = 1
    count_sig_worse = np.zeros(
        (n_classifiers, n_features)
    )  # p <= 0.05 & direction = -1

    # For each cell: count over all runs
    for run_idx in range(len(all_p_matrices)):
        p_matrix = all_p_matrices[run_idx]
        d_matrix = all_d_matrices[run_idx]

        for i, clf in enumerate(classifier_order):
            for j, feat in enumerate(PREFERRED_COLUMN_ORDER):
                # Check if this combination exists in this run
                if (
                    clf in p_matrix.index
                    and feat in p_matrix.columns
                    and not pd.isna(p_matrix.loc[clf, feat])
                ):

                    count_total[i, j] += 1

                    p_val = p_matrix.loc[clf, feat]
                    d_val = d_matrix.loc[clf, feat]

                    # Significantly better?
                    if p_val <= P_VALUE_THRESHOLD and d_val == 1:
                        count_sig_better[i, j] += 1

                    # Significantly worse?
                    if p_val <= P_VALUE_THRESHOLD and d_val == -1:
                        count_sig_worse[i, j] += 1

    # Create DataFrames for aggregated results
    df_total = pd.DataFrame(
        count_total, index=classifier_order, columns=PREFERRED_COLUMN_ORDER
    )
    df_better = pd.DataFrame(
        count_sig_better,
        index=classifier_order,
        columns=PREFERRED_COLUMN_ORDER,
    )
    df_worse = pd.DataFrame(
        count_sig_worse, index=classifier_order, columns=PREFERRED_COLUMN_ORDER
    )

    return df_total, df_better, df_worse


# 6) New heatmap function with adapted logic
def create_aggregated_heatmap_new(df_total, df_better, df_worse):
    """Creates heatmap with new logic:
    - 0/0 instead of 0/10 if no significant results
    - Green: more improvements than worsenings
    - Gray: equal improvements and worsenings
    - Red: more worsenings than improvements
    """

    # Create categorical matrix for colors
    n_rows, n_cols = df_better.shape
    color_matrix = np.full((n_rows, n_cols), np.nan)

    for i in range(n_rows):
        for j in range(n_cols):
            better = df_better.iloc[i, j]
            worse = df_worse.iloc[i, j]

            # Determine color based on comparison
            if better > worse:
                color_matrix[i, j] = 2  # GREEN: more improvements
            elif worse > better:
                color_matrix[i, j] = 0  # RED: more worsenings
            else:
                color_matrix[i, j] = 1  # GRAY: equal numbers (including 0/0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(22, 10))

    # Discrete colormap with 3 colors: Red, Gray, Green
    colors = ["#FF0000", "#CCCCCC", "#00AA00"]  # Red, Gray, Green
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot
    im = ax.imshow(color_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Axis labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(PREFERRED_COLUMN_ORDER, rotation=90, fontsize=18)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_better.index, fontsize=18)

    # Write values in cells (NEW LOGIC)
    for i in range(n_rows):
        for j in range(n_cols):
            better = int(df_better.iloc[i, j])
            worse = int(df_worse.iloc[i, j])

            # Determine text based on new logic
            if better == 0 and worse == 0:
                text = "0/0"  # No significant results
            else:
                text = f"{better}/{worse}"

            # Text color based on background
            bg_color = color_matrix[i, j]
            if bg_color == 0:  # Red
                text_color = "white"
            elif bg_color == 1:  # Gray
                text_color = "black"
            else:  # Green (2)
                text_color = "white"

            is_sum_row = i == 0
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=18 if is_sum_row else 18,
                fontweight="bold" if is_sum_row else "normal",
            )

    ax.axhline(0.5, color="black", linewidth=2.0)

    # Colorbar with labels (in English)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["More Worsenings", "Equal", "More Improvements"])
    cbar.ax.tick_params(labelsize=10)

    # Category separators
    draw_category_separators_and_labels(ax, PREFERRED_COLUMN_ORDER, y_pos=1.04)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def create_aggregated_heatmap_intensity(df_total, df_better, df_worse):
    """Creates heatmap with intensity scale for larger differences.

    CHANGES:
    1. All cells with equal improvements/worsenings (X/X) are light gray (same as 0/0)
    2. Text color is always black for better readability
    """

    # Calculate net improvements
    df_net = df_better - df_worse

    # Get dimensions
    n_rows, n_cols = df_better.shape

    # Create heatmap
    fig, ax = plt.subplots(figsize=(22, 10))

    # Color intensity scale: the larger the difference, the more intense the color
    colors = np.ones((n_rows, n_cols, 4), dtype=float)  # RGBA Matrix

    # --- exclude Σ row from color scaling ---
    df_net_for_scale = df_net.copy()
    if "Σ" in df_net_for_scale.index:
        df_net_for_scale = df_net_for_scale.drop(index="Σ")

    max_diff = max(
        df_net_for_scale.max().max() if not df_net_for_scale.empty else 0,
        abs(df_net_for_scale.min().min()) if not df_net_for_scale.empty else 0,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            better = df_better.iloc[i, j]
            worse = df_worse.iloc[i, j]
            net = df_net.iloc[i, j]

            # CHANGE 1: All cells with equal numbers (better == worse) get light gray
            if better == worse:
                # Equal numbers (0/0, 1/1, 2/2, etc.): light gray
                colors[i, j, :3] = [0.9, 0.9, 0.9]
            elif better > worse:
                # Green with intensity based on net difference
                intensity = (
                    min(abs(net) / max_diff, 1.0) if max_diff > 0 else 0.5
                )

                # Green
                colors[i, j, :3] = [
                    0.95 - 0.35 * intensity,  # R
                    0.98 - 0.15 * intensity,  # G
                    0.95 - 0.35 * intensity,  # B
                ]

            else:  # worse > better
                # Red with intensity based on net difference
                intensity = (
                    min(abs(net) / max_diff, 1.0) if max_diff > 0 else 0.5
                )

                colors[i, j, :3] = [
                    0.98 - 0.15 * intensity,  # R
                    0.95 - 0.35 * intensity,  # G
                    0.95 - 0.35 * intensity,  # B
                ]

    ax.imshow(colors, aspect="auto")

    # Axis labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(PREFERRED_COLUMN_ORDER, rotation=90, fontsize=18)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_better.index, fontsize=18)

    # Write values in cells
    for i in range(n_rows):
        for j in range(n_cols):
            better = int(df_better.iloc[i, j])
            worse = int(df_worse.iloc[i, j])

            if better == 0 and worse == 0:
                text = "0/0"
            else:
                text = f"{better}/{worse}"

            # CHANGE 2: Always use black text for better readability
            text_color = "black"

            is_sum_row = i == 0
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=18 if is_sum_row else 18,
                fontweight="bold" if is_sum_row else "normal",
            )

    ax.axhline(0.5, color="black", linewidth=2.0)

    # Category separators
    draw_category_separators_and_labels(ax, PREFERRED_COLUMN_ORDER, y_pos=1.04)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# 7) Main program
if __name__ == "__main__":
    print("Starting aggregation of McNemar results over 10 runs...")

    # 1. Collect data from all runs
    all_p_matrices, all_d_matrices = collect_mcnemar_for_all_runs()

    if not all_p_matrices:
        print("No McNemar data found. Terminating script.")
        sys.exit(1)

    print(f"Successfully collected data from {len(all_p_matrices)} runs.")

    # 2. Aggregate results
    df_total, df_better, df_worse = aggregate_mcnemar_results(
        all_p_matrices, all_d_matrices
    )

    # 3. Save aggregated tables
    output_files = []

    # CSV with counts
    df_counts = pd.DataFrame(
        {
            "Classifier": np.repeat(
                classifier_order, len(PREFERRED_COLUMN_ORDER)
            ),
            "Feature": np.tile(PREFERRED_COLUMN_ORDER, len(classifier_order)),
            "Total_Runs": df_total.values.flatten(),
            "Significant_Better": df_better.values.flatten(),
            "Significant_Worse": df_worse.values.flatten(),
        }
    )

    counts_path = os.path.join(ANALYSIS_DIR, "mcnemar_aggregated_counts.csv")
    df_counts.to_csv(counts_path, index=False)
    output_files.append(counts_path)
    print(f"Aggregated counts saved: {counts_path}")

    # Matrix formats for better readability
    df_better_matrix = pd.DataFrame(
        df_better, index=classifier_order, columns=PREFERRED_COLUMN_ORDER
    )
    df_worse_matrix = pd.DataFrame(
        df_worse, index=classifier_order, columns=PREFERRED_COLUMN_ORDER
    )
    df_net_matrix = df_better_matrix - df_worse_matrix

    # Σ-row over all classifiers
    sum_better = df_better_matrix.sum(axis=0)
    sum_worse = df_worse_matrix.sum(axis=0)

    sum_row_better = pd.DataFrame([sum_better], index=["Σ"])
    sum_row_worse = pd.DataFrame([sum_worse], index=["Σ"])

    df_better_matrix = pd.concat([sum_row_better, df_better_matrix], axis=0)
    df_worse_matrix = pd.concat([sum_row_worse, df_worse_matrix], axis=0)
    df_net_matrix = df_better_matrix - df_worse_matrix

    better_path = os.path.join(ANALYSIS_DIR, "mcnemar_counts_better.csv")
    worse_path = os.path.join(ANALYSIS_DIR, "mcnemar_counts_worse.csv")
    net_path = os.path.join(ANALYSIS_DIR, "mcnemar_counts_net.csv")

    df_better_matrix.to_csv(better_path)
    df_worse_matrix.to_csv(worse_path)
    df_net_matrix.to_csv(net_path)

    output_files.extend([better_path, worse_path, net_path])
    print(f"Matrix formats saved.")

    # 4. Create and save heatmaps
    # 4a. Discrete 3-color system
    fig = create_aggregated_heatmap_new(
        df_total, df_better_matrix, df_worse_matrix
    )
    heatmap_path = os.path.join(
        ANALYSIS_DIR, "mcnemar_aggregated_heatmap_3color_discrete.png"
    )
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_files.append(heatmap_path)
    print(f"Heatmap (3-color discrete) saved: {heatmap_path}")

    # 4b. Intensity-based heatmap
    fig = create_aggregated_heatmap_intensity(
        df_total, df_better_matrix, df_worse_matrix
    )
    heatmap_path = os.path.join(
        ANALYSIS_DIR, f"mcnemar_aggregated_heatmap_intensity_{VARIANT}.png"
    )
    heatmap_path_pdf = os.path.join(
        ANALYSIS_DIR, f"mcnemar_aggregated_heatmap_intensity_{VARIANT}.pdf"
    )
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    fig.savefig(heatmap_path_pdf, bbox_inches="tight")
    plt.close(fig)
    output_files.append(heatmap_path)
    print(f"Heatmap (intensity scale) saved: {heatmap_path}")

    # 5. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY OF AGGREGATED McNEMAR RESULTS:")
    print("=" * 60)

    total_cells = df_better_matrix.size
    cells_with_improvement = (df_better_matrix > 0).sum().sum()
    cells_with_worsening = (df_worse_matrix > 0).sum().sum()
    cells_equal = (
        ((df_better_matrix == df_worse_matrix) & (df_better_matrix > 0))
        .sum()
        .sum()
    )
    cells_no_sig = (
        ((df_better_matrix == 0) & (df_worse_matrix == 0)).sum().sum()
    )

    print(
        f"Total number of (Classifier × Feature) combinations: {total_cells}"
    )
    print(
        f"Combinations with significant improvement in ≥1 run: {cells_with_improvement} ({cells_with_improvement/total_cells*100:.1f}%)"
    )
    print(
        f"Combinations with significant worsening in ≥1 run: {cells_with_worsening} ({cells_with_worsening/total_cells*100:.1f}%)"
    )
    print(
        f"Combinations with equal improvements/worsenings: {cells_equal} ({cells_equal/total_cells*100:.1f}%)"
    )
    print(
        f"Combinations without significant results: {cells_no_sig} ({cells_no_sig/total_cells*100:.1f}%)"
    )

    # Best features (by net improvement over all classifiers)
    net_by_feature = df_net_matrix.sum(axis=0)
    top_features = net_by_feature.sort_values(ascending=False).head(10)

    print(f"\nTop 10 features (by net improvements over all classifiers):")
    for feat, net_score in top_features.items():
        if net_score > 0:
            print(f"  {feat}: +{net_score:.0f} (net improvements)")

    # Worst features
    bottom_features = net_by_feature.sort_values(ascending=True).head(10)

    print(f"\nBottom 10 features (by net worsenings over all classifiers):")
    for feat, net_score in bottom_features.items():
        if net_score < 0:
            print(f"  {feat}: {net_score:.0f} (net worsenings)")

    # Features with perfect 10/0 or 0/10
    perfect_improvements = (df_better_matrix == 10).sum(axis=0)
    perfect_worsenings = (df_worse_matrix == 10).sum(axis=0)

    print(f"\nFeatures with perfect 10/0 (always significantly better):")
    for feat in perfect_improvements[perfect_improvements > 0].index:
        count = perfect_improvements[feat]
        if count > 0:
            print(f"  {feat}: {count} classifier(s)")

    print(f"\nFeatures with perfect 0/10 (always significantly worse):")
    for feat in perfect_worsenings[perfect_worsenings > 0].index:
        count = perfect_worsenings[feat]
        if count > 0:
            print(f"  {feat}: {count} classifier(s)")

    print("\n" + "=" * 60)
    print(f"All output files have been saved in: {ANALYSIS_DIR}")
    print("=" * 60)
