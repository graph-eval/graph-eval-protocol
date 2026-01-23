# -*- coding: utf-8 -*-
"""
Aggregation of scores across multiple runs (e.g., _0_1 … _0_10) using trimming:
The mean is computed over the central 8 values, with the minimum and maximum
values removed.

Extension:
- additionally compute the standard deviation over the same central 8 values
- generate an additional heatmap/figure analogous to the mean-intensity figure,
  but visualizing standard deviation values instead
  (intensity encoding: higher standard deviation → darker color)

Output: C:\Experiments\Analysis\Output_{VARIANT}_agg
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm


# 1) Configuration

# Adapt edge removal variant
VARIANT = "0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_BASE = os.path.abspath(
    os.path.join(
        SCRIPT_DIR, "..", "..", "src", "Results_HyPa_fix_Singles", "Results"
    )
)

RUN_SUFFIXES = [f"_{VARIANT}_{i}" for i in range(1, 11)]

# Target folder
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, f"Output_{VARIANT}_agg")

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

METRICS = [
    {
        "name": "logloss",
        "better_is": "lower",
        "key_candidates": ["log_loss"],
        "file_stub": "logloss",
        "pretty_title": "LogLoss",
        "sections": [
            "metrics_test",
            "test_metrics",
            "final_test_metrics_argmax",
            "final_test_metrics_tuned",
        ],
    },
    {
        "name": "f1_fraud",
        "better_is": "higher",
        "key_candidates": ["f1_fraud", "f1", "f1_score_fraud", "f1_score"],
        "file_stub": "f1",
        "pretty_title": "F1-Score",
        "sections": [
            "metrics_test",
            "test_metrics",
            "final_test_metrics_argmax",
            "final_test_metrics_tuned",
            "final_test_metrics",
        ],
    },
    {
        "name": "recall_fraud",
        "better_is": "higher",
        "key_candidates": [
            "recall_fraud",
            "recall",
            "recall_score_fraud",
            "recall_score",
        ],
        "file_stub": "recall",
        "pretty_title": "Recall (Fraud=0)",
        "sections": [
            "metrics_test",
            "test_metrics",
            "final_test_metrics_argmax",
            "final_test_metrics_tuned",
        ],
    },
    {
        "name": "precision_fraud",
        "better_is": "higher",
        "key_candidates": [
            "precision_fraud",
            "precision",
            "precision_score_fraud",
            "precision_score",
        ],
        "file_stub": "precision",
        "pretty_title": "Precision (Fraud=0)",
        "sections": [
            "metrics_test",
            "test_metrics",
            "final_test_metrics_argmax",
            "final_test_metrics_tuned",
        ],
    },
    {
        "name": "roc_auc_fraud",
        "better_is": "higher",
        "key_candidates": ["roc_auc_fraud", "roc_auc"],
        "file_stub": "rocauc",
        "pretty_title": "ROC-AUC (Fraud=0)",
        "sections": [
            "metrics_test",
            "test_metrics",
            "final_test_metrics_argmax",
            "final_test_metrics_tuned",
        ],
    },
    {
        "name": "pr_auc_fraud",
        "better_is": "higher",
        "key_candidates": ["pr_auc_fraud", "pr_auc"],
        "file_stub": "prauc",
        "pretty_title": "PR-AUC (Fraud=0)",
        "sections": [
            "metrics_test",
            "test_metrics",
            "final_test_metrics_argmax",
            "final_test_metrics_tuned",
        ],
    },
]

TOP_K_BASE_CLASSIFIERS = 5

GROUP_COLS = [
    "All GraphIndicators",
    "All Proximity Embeddings",
    "All Structural Embeddings",
    "All",
]


# 2) Mapping / Order
COLUMN_MAPPING = {
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "TRX Only",
    "I1+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "DEGREE_CENTRALITY",
    "I2+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "IN_OUT_DEGREE",
    "I3+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "PAGERANK",
    "I4+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "BETWEENNESS_CENTRALITY",
    "I5+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "EIGENVECTOR_CENTRALITY",
    "I6+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "CLOSENESS_CENTRALITY",
    "I7+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "CLUSTERING_COEFFICIENT",
    "I8+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "CORE_NUMBER",
    "I9+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "TRIANGLES",
    "I10+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "SQUARE_CLUSTERING",
    "I11+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "COMMUNITY_LOUVAIN",
    "I12+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "COMMUNITY_LEIDEN",
    "I13+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "COMMUNITY_INFOMAP",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+DeepWalk": "DeepWalk",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBFS": "node2vec-BFS",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VDFS": "node2vec-DFS",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBal": "node2vec-bal",
    "noInd+1Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Spectral",
    "noInd+0Spectral+1ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "ffstruc2vec",
    "noInd+0Spectral+0ff2Vec+0Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox": "Role2Vec",
    "noInd+0Spectral+0ff2Vec+1Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Graphwave",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+1GCN+0GAT+0GCL+0Prox": "GCN",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+1GAT+0GCL+0Prox": "GAT",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+1GCL+0Prox": "GCL",
    "I1+I2+I3+I4+I5+I6+I7+I8+I9+I10+I11+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "All GraphIndicators",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBal+N2VDFS+N2VBFS+DeepWalk": "All Proximity Embeddings",
    "noInd+1Spectral+0ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox": "All Structural Embeddings",
    "I1+I2+I3+I4+I5+I6+I7+I8+I9+I10+I11+1Spectral+0ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+N2VBal+N2VDFS+N2VBFS+DeepWalk": "All",
}

PREFERRED_COLUMN_ORDER = [
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


# 3) Helper functions
def parse_filename(fname: str):
    parts = fname.split("_")
    if len(parts) < 3:
        return None, None
    classifier = parts[1]

    m = re.search(
        r"base\d+\+(.+?)(?:_(?:edgesVar|edges)_?\d+_splitseed\d+)?_random",
        fname,
    )
    if m:
        return classifier, m.group(1)

    m = re.search(r"base\d+\+(.+?)_random", fname)
    if m:
        return classifier, m.group(1)

    return classifier, "UNKNOWN"


def sort_columns_by_category(columns, preferred_order):
    cols = [c for c in preferred_order if c in columns]
    rest = [c for c in columns if c not in cols]
    return cols + sorted(rest)


def extract_metric(data, key_candidates, sections):
    for sec in sections:
        if sec in data and isinstance(data[sec], dict):
            for key in key_candidates:
                if key in data[sec]:
                    try:
                        return float(data[sec][key])
                    except Exception:
                        pass

    if "final_test_metrics" in data and isinstance(
        data["final_test_metrics"], dict
    ):
        ftm = data["final_test_metrics"]
        for thresh_key in ["at_tau_star", "at_0_5"]:
            if thresh_key in ftm and isinstance(ftm[thresh_key], dict):
                d = ftm[thresh_key]
                for key in key_candidates:
                    if key in d:
                        try:
                            return float(d[key])
                        except Exception:
                            pass

    preferred_result_blocks = [
        "test_results_tuned",
        "test_results_argmax",
        "validation_results_tuned",
        "validation_results_argmax",
    ]
    for block in preferred_result_blocks:
        if block in data and isinstance(data[block], dict):
            d = data[block]
            for key in key_candidates:
                if key in d:
                    try:
                        return float(d[key])
                    except Exception:
                        pass

    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in key_candidates:
                    try:
                        return float(v)
                    except Exception:
                        return None
            for v in obj.values():
                res = _search(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _search(item)
                if res is not None:
                    return res
        return None

    return _search(data)


def draw_category_separators_and_labels(ax, columns):
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
        "Structure": ["ffstruc2vec", "Role2Vec", "Graphwave"],
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
            1.02,
            cat,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=16,
        )


def middle_trim_values(values):
    """
    Removes min & max (if >= 3 values) and returns the central values.
    """
    arr = np.array([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    if arr.size < 3:
        return arr
    arr_sorted = np.sort(arr)
    trimmed = arr_sorted[1:-1]
    if trimmed.size == 0:
        return arr_sorted
    return trimmed


def trimmed_mean_middle8(values):
    arr_mid = middle_trim_values(values)
    if arr_mid.size == 0:
        return np.nan
    return float(np.mean(arr_mid))


def trimmed_std_middle8(values):
    """
    Std dev over the same trimmed (central) values.
    """
    arr_mid = middle_trim_values(values)
    if arr_mid.size == 0:
        return np.nan
    if arr_mid.size < 2:
        return 0.0
    return float(np.std(arr_mid, ddof=1))


# 4) Output-Function
def generate_outputs(metric_matrix, stub, m_pretty, better_is, suffix=""):
    AVG_ROW_GRAY = np.array([0.85, 0.85, 0.85])
    if metric_matrix.empty:
        print(f"Empty matrix for {m_pretty} (Suffix '{suffix}') – skipping.")
        return

    csv_name = f"{stub}_matrix{suffix}.csv"
    csv_path = os.path.join(ANALYSIS_DIR, csv_name)
    metric_matrix.to_csv(csv_path)
    print(f"{csv_name} saved to: {csv_path}")

    base_col = "TRX Only"
    baseline_series = (
        metric_matrix[base_col] if base_col in metric_matrix.columns else None
    )

    # Matrix 1
    real_matrix = metric_matrix.copy()

    # Summary rows (Top-K based on baseline)
    if baseline_series is not None and len(real_matrix.index) > 0:
        valid_baseline = baseline_series.dropna()
        if not valid_baseline.empty:
            ascending = True if better_is == "lower" else False
            top_clfs = (
                valid_baseline.sort_values(ascending=ascending)
                .head(TOP_K_BASE_CLASSIFIERS)
                .index.tolist()
            )
            top_clfs = [c for c in real_matrix.index if c in top_clfs]
            if top_clfs:
                top_matrix = real_matrix.loc[top_clfs]
                avg_abs = top_matrix.mean(axis=0, skipna=True)
                base_for_top = baseline_series.loc[top_clfs]
                avg_delta = (top_matrix.sub(base_for_top, axis=0)).mean(
                    axis=0, skipna=True
                )
                k_used = len(top_clfs)
                avg_abs_name = f"AVG"
                avg_delta_name = f"Δ"
                avg_df = pd.DataFrame(
                    [avg_abs, avg_delta], index=[avg_abs_name, avg_delta_name]
                )
                real_matrix = pd.concat([avg_df, real_matrix], axis=0)
                print(
                    f"Summary rows for {m_pretty}{suffix} based on: {', '.join(top_clfs)}"
                )

    n_rows, n_cols = real_matrix.shape
    color_index = np.full((n_rows, n_cols), np.nan)

    for i, clf in enumerate(real_matrix.index):
        for j, col in enumerate(real_matrix.columns):
            val = real_matrix.iloc[i, j]
            if pd.isna(val):
                continue
            if col == base_col:
                color_index[i, j] = 0
            elif (
                baseline_series is not None
                and clf in baseline_series.index
                and not pd.isna(baseline_series.loc[clf])
            ):
                diff = val - baseline_series.loc[clf]
                if better_is == "lower":
                    color_index[i, j] = 1 if diff <= 0 else 2
                else:
                    color_index[i, j] = 1 if diff >= 0 else 2

    cmap_real = ListedColormap(["#555555", "#00AA00", "#FF0000"])
    cmap_real.set_bad(color="#D3D3D3")
    norm_real = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_real.N)

    fig1, ax1 = plt.subplots(figsize=(22, 10))
    ax1.imshow(color_index, cmap=cmap_real, norm=norm_real, aspect="auto")
    ax1.set_xticks(np.arange(n_cols))
    ax1.set_xticklabels(real_matrix.columns, rotation=90, fontsize=16)
    ax1.set_yticks(np.arange(n_rows))
    ax1.set_yticklabels(real_matrix.index, fontsize=16)

    for i in range(n_rows):
        for j in range(n_cols):
            val = real_matrix.iloc[i, j]
            if pd.isna(val):
                continue
            is_delta_row = i == 1
            ax1.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontweight="normal",
                #                     fontweight="bold" if is_delta_row else "normal",
                color="black",
                fontsize=15,
            )

    ax1.set_title(
        f"{m_pretty} - Dark Gray = Base/neutral, Green = improvement vs TRX-only base, Red = worse/same",
        pad=35,
        fontsize=18,
    )
    draw_category_separators_and_labels(ax1, real_matrix.columns)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    m1_name = f"matrix1_{stub}{suffix}_realvalues.png"
    plt.savefig(os.path.join(ANALYSIS_DIR, m1_name), dpi=300)
    plt.close(fig1)
    print(f"✔ {m1_name} saved.")

    # Matrix 1b: Intensity ~ |Δ|
    if baseline_series is not None:
        diff_eff = np.full((n_rows, n_cols), np.nan, dtype=float)
        for i, clf in enumerate(real_matrix.index):
            if clf not in baseline_series.index or pd.isna(
                baseline_series.loc[clf]
            ):
                continue
            base_val = baseline_series.loc[clf]
            for j, col in enumerate(real_matrix.columns):
                if col == base_col:
                    continue
                val = real_matrix.iloc[i, j]
                if pd.isna(val):
                    continue
                d = val - base_val
                if better_is == "lower":
                    d = -d
                diff_eff[i, j] = d

        max_abs = (
            np.nanmax(np.abs(diff_eff)) if np.any(~np.isnan(diff_eff)) else 0.0
        )

        colors = np.ones((n_rows, n_cols, 4), dtype=float)
        colors[..., :3] = 0.9

        green = np.array([0.50, 0.72, 0.50])
        red = np.array([0.80, 0.55, 0.55])

        for i in range(n_rows):
            for j, col in enumerate(real_matrix.columns):
                if col == base_col:
                    colors[i, j, :3] = np.array([0.33, 0.33, 0.33])
                    continue
                d = diff_eff[i, j]
                if np.isnan(d) or max_abs == 0.0:
                    continue

                strength = min(abs(d) / max_abs, 1.0)

                # Option B: Gamma / sqrt scaling
                gamma = 0.7
                strength = strength**gamma

                strength = 0.2 + 1.1 * strength

                base_col_vec = green if d >= 0 else red
                rgb = 1.0 - strength * (1.0 - base_col_vec)
                colors[i, j, :3] = rgb

        fig1b, ax1b = plt.subplots(figsize=(22, 10))
        colors[0, :, :3] = AVG_ROW_GRAY
        ax1b.imshow(colors, aspect="auto")
        ax1b.set_xticks(np.arange(n_cols))
        ax1b.set_xticklabels(real_matrix.columns, rotation=90, fontsize=18)
        ax1b.set_yticks(np.arange(n_rows))
        ax1b.set_yticklabels(real_matrix.index, fontsize=18)

        for i in range(n_rows):
            for j in range(n_cols):

                val = real_matrix.iloc[i, j]
                if pd.isna(val):
                    continue
                is_delta_row = i == 1
                ax1b.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontweight="normal",
                    color="black",
                    fontsize=17,
                )
        colors[0, :, :3] = AVG_ROW_GRAY

        draw_category_separators_and_labels(ax1b, real_matrix.columns)
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        m1b_name = f"matrix1_{stub}{suffix}_realvalues_intensity_{VARIANT}.png"
        m1b_name_pdf = (
            f"matrix1_{stub}{suffix}_realvalues_intensity_{VARIANT}.pdf"
        )
        plt.savefig(os.path.join(ANALYSIS_DIR, m1b_name), dpi=300)
        plt.savefig(
            os.path.join(ANALYSIS_DIR, m1b_name_pdf), bbox_inches="tight"
        )
        plt.close(fig1b)
        print(f"✔ {m1b_name} saved.")

    # Matrix 2: % Changes
    if baseline_series is not None:
        pct_change = (
            metric_matrix.sub(baseline_series, axis=0) / baseline_series
        ) * 100.0
    else:
        pct_change = metric_matrix.copy() * np.nan

    pct_for_heat = pct_change.copy()
    if base_col in pct_for_heat.columns:
        pct_for_heat[base_col] = np.nan
    mask = pct_for_heat.isna()
    if mask.all().all():
        print(f"Matrix 2{suffix} ({m_pretty}) skipped (all NaN).")
        return

    fig2, ax2 = plt.subplots(figsize=(22, 10))
    cmap2 = plt.get_cmap("RdYlGn_r").copy()
    cmap2.set_bad(color="#D3D3D3")

    sns.heatmap(
        pct_for_heat,
        cmap=cmap2,
        center=0,
        annot=False,
        fmt=".1f",
        xticklabels=pct_change.columns,
        yticklabels=pct_change.index,
        mask=mask,
        ax=ax2,
    )

    for i in range(pct_change.shape[0]):
        for j in range(pct_change.shape[1]):
            val = pct_change.iloc[i, j]
            if pd.isna(val):
                continue
            ax2.text(
                j + 0.5,
                i + 0.5,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=16,
                color="black",
            )

    ax2.set_title(f"Percentage Change vs. Base Performance (%) – {m_pretty}")
    ax2.set_xlabel("Variant")
    ax2.set_ylabel("Classifier")
    draw_category_separators_and_labels(ax2, pct_change.columns)
    plt.tight_layout()

    m2_name = f"matrix2_{stub}{suffix}_percentage_change_heatmap.png"
    plt.savefig(os.path.join(ANALYSIS_DIR, m2_name), dpi=300)
    plt.close(fig2)
    print(f"✔ {m2_name} saved.")


def generate_win_outputs(win_matrix, stub, m_pretty, suffix=""):
    """
    Generates:
    - CSV: <stub>_wins_vs_base_matrix<suffix>.csv
    - Figure: matrix_wins_<stub><suffix>_vs_base.png
    """
    if win_matrix.empty:
        print(
            f"Win matrix empty for {m_pretty} (Suffix '{suffix}') – skipping."
        )
        return

    csv_name = f"{stub}_wins_vs_base_matrix{suffix}.csv"
    csv_path = os.path.join(ANALYSIS_DIR, csv_name)
    win_matrix.to_csv(csv_path)
    print(f"✔ {csv_name} saved to: {csv_path}")

    mat = win_matrix.copy()
    n_rows, n_cols = mat.shape

    values = mat.to_numpy(dtype=float)
    max_val = np.nanmax(values) if np.any(~np.isnan(values)) else 0.0

    # Light gray base; more wins → darker
    colors = np.ones((n_rows, n_cols, 4), dtype=float)
    colors[..., :3] = 0.93
    colors[..., 3] = 1.0

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v) or max_val == 0.0:
                continue
            strength = min(v / max_val, 1.0)
            dark = 0.75 * strength
            rgb = 1.0 - dark
            colors[i, j, :3] = np.array([rgb, rgb, rgb])

    fig, ax = plt.subplots(figsize=(22, 10))
    ax.imshow(colors, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=16)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(mat.index, fontsize=16)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat.iloc[i, j]
            if pd.isna(v):
                continue
            ax.text(
                j,
                i,
                f"{int(v)}",
                ha="center",
                va="center",
                color="black",
                fontsize=16,
            )

    ax.set_title(
        f"{m_pretty} – #Seeds where Feature > Base(TRX-only) (higher = darker)",
        pad=35,
        fontsize=18,
    )
    draw_category_separators_and_labels(ax, mat.columns)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    fig_name = f"matrix_wins_{stub}{suffix}_vs_base.png"
    plt.savefig(os.path.join(ANALYSIS_DIR, fig_name), dpi=300)
    plt.close(fig)
    print(f"✔ {fig_name} saved.")


# 4b) StdDev-Outputs: CSV + "Intensity"-Figure
def generate_std_outputs(std_matrix, stub, m_pretty, suffix=""):
    """
    Generates:
    - CSV: <stub>_std_matrix<suffix>.csv
    - Figure: matrix1_<stub><suffix>_std_middle8_realvalues_intensity.png
    """
    if std_matrix.empty:
        print(
            f"Std matrix empty for {m_pretty} (Suffix '{suffix}') – skipping."
        )
        return

    csv_name = f"{stub}_std_matrix{suffix}.csv"
    csv_path = os.path.join(ANALYSIS_DIR, csv_name)
    std_matrix.to_csv(csv_path)
    print(f"✔ {csv_name} saved to: {csv_path}")

    mat = std_matrix.copy()
    n_rows, n_cols = mat.shape

    values = mat.to_numpy(dtype=float)
    max_val = np.nanmax(values) if np.any(~np.isnan(values)) else 0.0

    # Base color: very light gray
    colors = np.ones((n_rows, n_cols, 4), dtype=float)
    colors[..., :3] = 0.93
    colors[..., 3] = 1.0

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v) or max_val == 0.0:
                continue
            strength = min(v / max_val, 1.0)
            dark = 0.65 * strength
            rgb = 1.0 - dark
            colors[i, j, :3] = np.array([rgb, rgb, rgb])

    fig, ax = plt.subplots(figsize=(22, 10))
    ax.imshow(colors, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=18)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(mat.index, fontsize=18)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat.iloc[i, j]
            if pd.isna(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=17,
            )

    draw_category_separators_and_labels(ax, mat.columns)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # fig_name = f"matrix1_{stub}{suffix}_std_middle8_realvalues_intensity.png"
    fig_name_pdf = (
        f"matrix1_{stub}{suffix}_std_middle8_realvalues_intensity.pdf"
    )
    # plt.savefig(os.path.join(ANALYSIS_DIR, fig_name), dpi=300)
    plt.savefig(os.path.join(ANALYSIS_DIR, fig_name_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"✔ {fig_name_pdf} saved.")


# 5) Collect JSON files across runs
all_records = []  # (run_suffix, classifier, variant, json_data)

for run_suf in RUN_SUFFIXES:
    run_dir = RESULTS_DIR_BASE + run_suf
    if not os.path.exists(run_dir):
        print(f"Run directory not found, skipping: {run_dir}")
        continue

    for root, dirs, files in os.walk(run_dir):
        if os.path.basename(root).lower() == "archive":
            continue
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            clf, var = parse_filename(fname)
            if clf is None or var is None:
                continue

            if clf in {"GCN", "GAT", "GCL"}:
                continue

            all_records.append((run_suf, clf, var, data))

if not all_records:
    raise RuntimeError("No matching JSON files found across runs.")


# 6) Aggregate per metric (trimmed mean + std over central values)
classifier_order = ["MLP", "LR", "NB", "SVC", "RF", "XGB"]

for metric in METRICS:
    m_pretty = metric["pretty_title"]
    better_is = metric["better_is"]
    stub = metric["file_stub"]
    sections = metric["sections"]
    key_candidates = metric["key_candidates"]

    rows = []  # (run, clf, var, value)
    for run_suf, clf, var, data in all_records:
        val = extract_metric(data, key_candidates, sections)
        if val is None:
            continue
        rows.append((run_suf, clf, var, float(val)))

    if not rows:
        print(f"No values found for metric: {m_pretty}")
        continue

    df_raw = pd.DataFrame(
        rows, columns=["Run", "Classifier", "Variant", "Value"]
    )

    df_raw["VariantPretty"] = df_raw["Variant"].map(
        lambda x: COLUMN_MAPPING.get(x, x)
    )

    # EXTRA: Win-Counts vs Baseline
    if metric["file_stub"] == "f1":
        base_col = "TRX Only"

        wide = df_raw.pivot_table(
            index=["Run", "Classifier"],
            columns="VariantPretty",
            values="Value",
            aggfunc="mean",
        )

        if base_col in wide.columns:
            base = wide[base_col]

            diffs = wide.sub(base, axis=0)

            wins_bool = diffs.gt(0)  # strict > 0
            if base_col in wins_bool.columns:
                wins_bool[base_col] = False

            # Sum over seeds: yields #wins (0–10) per classifier × variant.
            win_counts = wins_bool.groupby("Classifier").sum(min_count=1)

            win_counts = win_counts.reindex(classifier_order).dropna(how="all")
            ordered_cols_wins = sort_columns_by_category(
                list(win_counts.columns), PREFERRED_COLUMN_ORDER
            )
            win_counts = win_counts[ordered_cols_wins]

            cols_without_groups_wins = [
                c
                for c in win_counts.columns
                if c not in GROUP_COLS and c != base_col
            ]
            win_counts_main = win_counts[cols_without_groups_wins]

            # Output (Main)
            generate_win_outputs(win_counts_main, stub, m_pretty, suffix="")

            # Output (Groups: Base + GROUP_COLS)
            group_cols_present = [
                c for c in GROUP_COLS if c in win_counts.columns
            ]
            cols_groups = []
            if base_col in win_counts.columns:
                cols_groups.append(base_col)
            cols_groups.extend(group_cols_present)

            if len(cols_groups) >= 2:
                win_counts_groups = win_counts[cols_groups]
                generate_win_outputs(
                    win_counts_groups, stub, m_pretty, suffix="_groups"
                )
        else:
            print(
                f"Baseline column '{base_col}' not found – Win-Counts skipped."
            )

    agg_stats = (
        df_raw.groupby(["Classifier", "VariantPretty"])["Value"]
        .agg(Mean=trimmed_mean_middle8, Std=trimmed_std_middle8)
        .reset_index()
    )

    pivot_mean = agg_stats.pivot(
        index="Classifier", columns="VariantPretty", values="Mean"
    )
    pivot_std = agg_stats.pivot(
        index="Classifier", columns="VariantPretty", values="Std"
    )

    pivot_mean = pivot_mean.reindex(classifier_order).dropna(how="all")
    pivot_std = pivot_std.reindex(classifier_order).dropna(how="all")

    ordered_cols_mean = sort_columns_by_category(
        list(pivot_mean.columns), PREFERRED_COLUMN_ORDER
    )
    ordered_cols_std = sort_columns_by_category(
        list(pivot_std.columns), PREFERRED_COLUMN_ORDER
    )

    pivot_mean = pivot_mean[ordered_cols_mean]
    pivot_std = pivot_std[ordered_cols_std]

    metric_matrix_full_mean = pivot_mean.copy()
    metric_matrix_full_std = pivot_std.copy()

    cols_without_groups_mean = [
        c for c in metric_matrix_full_mean.columns if c not in GROUP_COLS
    ]
    cols_without_groups_std = [
        c for c in metric_matrix_full_std.columns if c not in GROUP_COLS
    ]

    metric_matrix_main_mean = metric_matrix_full_mean[cols_without_groups_mean]
    metric_matrix_main_std = metric_matrix_full_std[cols_without_groups_std]

    # Mean
    generate_outputs(
        metric_matrix_main_mean, stub, m_pretty, better_is, suffix=""
    )

    # Std
    generate_std_outputs(metric_matrix_main_std, stub, m_pretty, suffix="")

    # Groups (Base + GROUP_COLS)
    group_cols_present = [
        c for c in GROUP_COLS if c in metric_matrix_full_mean.columns
    ]
    cols_groups = []
    base_col = "TRX Only"
    if base_col in metric_matrix_full_mean.columns:
        cols_groups.append(base_col)
    cols_groups.extend(group_cols_present)

    if len(cols_groups) >= 2:
        metric_matrix_groups_mean = metric_matrix_full_mean[cols_groups]
        metric_matrix_groups_std = metric_matrix_full_std[cols_groups]

        generate_outputs(
            metric_matrix_groups_mean,
            stub,
            m_pretty,
            better_is,
            suffix="_groups",
        )
        generate_std_outputs(
            metric_matrix_groups_std, stub, m_pretty, suffix="_groups"
        )
    else:
        print(
            f"Not enough group columns available for {m_pretty} — skipping group matrix."
        )

print("Done. (Trimmed mean + standard deviation over the central 8 runs)")
