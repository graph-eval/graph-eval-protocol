# -*- coding: utf-8 -*-
"""
Same evaluation logic as collect_scores_agg_winsorizing.py,
but applied to *category runs* (one separate run & JSON per category).

Example input file:
elliptic_LR_20251230T175330Z_base93+noInd+0Spectral+1ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox_edgesVar_0_splitseed1_random_val20_test20_3694ec6f.json

This script:
- reads Results_<VARIANT>_1 ... Results_<VARIANT>_10
- extracts metrics (LogLoss, F1, Precision, Recall, ROC-AUC, PR-AUC)
- aggregates per (Classifier × Category) using the “middle-8” rule (drop min/max),
  computing mean and standard deviation over the remaining values
- produces CSVs + heatmaps in the same style as the baseline script
- additionally (for F1) creates an intensity matrix showing value ± std-dev per cell
- additionally (F1 only) computes win-counts vs. the base (TRX-only) across seeds

To adapt:
- RESULTS_DIR_BASE
- VARIANT
- CATEGORY_MAPPING / PREFERRED_COLUMN_ORDER (if your categories are named differently)
"""


import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

# 1) Configuration
VARIANT = "0"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_BASE = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "src", "Results_HyPa_fix", "Results")
)

RUN_SUFFIXES = [f"_{VARIANT}_{i}" for i in range(1, 11)]  # _0_1 .. _0_10

ANALYSIS_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, f"Output_{VARIANT}_categories_agg")
)

os.makedirs(ANALYSIS_DIR, exist_ok=True)

CLASSIFIER_ORDER = ["MLP", "LR", "NB", "SVC", "RF", "XGB"]

# 2) Category-Mapping
CATEGORY_MAPPING = {
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "TRX Only",
    "I1+I2+I3+I4+I5+I6+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Centrality",
    "I7+I8+I9+I10+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Cohesion",
    "I11+I12+I13+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Community",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+N2VBal+N2VDFS+N2VBFS+DeepWalk": "Proximity",
    "noInd+1Spectral+0ff2Vec+0Graphwave+0Role2Vec+0GCN+0GAT+0GCL+0Prox": "Spectral",
    "noInd+0Spectral+1ff2Vec+1Graphwave+1Role2Vec+0GCN+0GAT+0GCL+0Prox": "Structure",
    "noInd+0Spectral+0ff2Vec+0Graphwave+0Role2Vec+1GCN+1GAT+1GCL+0Prox": "GNN",
}

PREFERRED_COLUMN_ORDER = [
    "TRX Only",
    "Centrality",
    "Cohesion",
    "Community",
    "Proximity",
    "Spectral",
    "Structure",
    "GNN",
]

# 3) Metrics
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
            "final_test_metrics",
        ],
    },
    {
        "name": "f1_fraud",
        "better_is": "higher",
        "key_candidates": ["f1_fraud", "f1", "f1_score_fraud", "f1_score"],
        "file_stub": "f1",
        "pretty_title": "F1-Scores",
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
            "final_test_metrics",
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
            "final_test_metrics",
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
            "final_test_metrics",
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
            "final_test_metrics",
        ],
    },
]

TOP_K_BASE_CLASSIFIERS = 5


# 4) Helper Functions
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


def sort_columns(columns, preferred_order):
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


def middle_trim_values(values):
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
    arr_mid = middle_trim_values(values)
    if arr_mid.size == 0:
        return np.nan
    if arr_mid.size < 2:
        return 0.0
    return float(np.std(arr_mid, ddof=1))


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

        # Default: draw separators only for multi-column categories
        if start != end:
            ax.axvline(start - 0.5, color="black", linewidth=1.0)
            ax.axvline(end + 0.5, color="black", linewidth=1.0)


def generate_outputs(
    metric_matrix,
    stub,
    m_pretty,
    better_is,
    suffix="",
    std_matrix_for_annot=None,
):
    if metric_matrix.empty:
        print(f"Empty matrix for {m_pretty} (suffix '{suffix}') – skipped.")
        return

    csv_name = f"{stub}_matrix{suffix}_{VARIANT}.csv"
    csv_path = os.path.join(ANALYSIS_DIR, csv_name)
    metric_matrix.to_csv(csv_path)
    print(f"✔ {csv_name} saved to: {csv_path}")

    base_col = "TRX Only"
    baseline_series = (
        metric_matrix[base_col] if base_col in metric_matrix.columns else None
    )

    # Matrix 1
    real_matrix = metric_matrix.copy()

    top_clfs = []

    if len(real_matrix.index) > 0:
        if baseline_series is not None:
            valid_baseline = baseline_series.dropna()
            if not valid_baseline.empty:
                ascending = True if better_is == "lower" else False
                top_clfs = (
                    valid_baseline.sort_values(ascending=ascending)
                    .head(TOP_K_BASE_CLASSIFIERS)
                    .index.tolist()
                )
        else:
            row_mean = real_matrix.mean(axis=1, skipna=True).dropna()
            if not row_mean.empty:
                ascending = True if better_is == "lower" else False
                top_clfs = (
                    row_mean.sort_values(ascending=ascending)
                    .head(TOP_K_BASE_CLASSIFIERS)
                    .index.tolist()
                )

        top_clfs = [c for c in real_matrix.index if c in top_clfs]

        if top_clfs:
            top_matrix = real_matrix.loc[top_clfs]
            avg_abs = top_matrix.mean(axis=0, skipna=True)

            if baseline_series is not None:
                base_for_top = baseline_series.loc[top_clfs]
                avg_delta = (top_matrix.sub(base_for_top, axis=0)).mean(
                    axis=0, skipna=True
                )
                avg_df = pd.DataFrame(
                    [avg_abs, avg_delta],
                    index=[f"AVG (Top-{len(top_clfs)})", "Δ (AVG vs Base)"],
                )
            else:
                avg_df = pd.DataFrame(
                    [avg_abs], index=[f"AVG (Top-{len(top_clfs)})"]
                )

            real_matrix = pd.concat([avg_df, real_matrix], axis=0)
            print(
                f"Added summary row(s) for {m_pretty}{suffix} based on: {', '.join(top_clfs)}"
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

    cmap_real = ListedColormap(["#777777", "#A6DDA6", "#F2B3B3"])
    cmap_real.set_bad(color="#D3D3D3")

    cmap_real.set_bad(color="#D3D3D3")
    norm_real = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_real.N)

    fig1, ax1 = plt.subplots(figsize=(18, 8))
    ax1.imshow(color_index, cmap=cmap_real, norm=norm_real, aspect="auto")
    ax1.set_xticks(np.arange(n_cols))
    ax1.set_xticklabels(real_matrix.columns, rotation=90, fontsize=24)
    ax1.set_yticks(np.arange(n_rows))
    ax1.set_yticklabels(real_matrix.index, fontsize=24)

    for i in range(n_rows):
        for j in range(n_cols):
            val = real_matrix.iloc[i, j]
            if pd.isna(val):
                continue
            ax1.text(
                j,
                i,
                f"{val:.4f}",
                ha="center",
                va="center",
                color="black",
                fontsize=24,
            )

    ax1.set_title(
        f"{m_pretty} - Gray=Base, Green=better/equal vs Base, Red=worse",
        pad=20,
    )
    plt.tight_layout()
    fig1_name = f"matrix1_{stub}{suffix}_realvalues_sign_{VARIANT}.png"
    plt.savefig(os.path.join(ANALYSIS_DIR, fig1_name), dpi=300)
    plt.close(fig1)

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

        AVG_ROW_GRAY = np.array([0.88, 0.88, 0.88])  # leicht heller als normal

        # 0 = AVG, 1 = Δ
        colors[0, :, :3] = AVG_ROW_GRAY
        colors[1, :, :3] = AVG_ROW_GRAY

        green = np.array([0.45, 0.78, 0.45])
        red = np.array([0.92, 0.62, 0.62])

        MIN_STRENGTH = 0.22
        MAX_STRENGTH = 0.75
        GAMMA = 0.6

        for i in range(n_rows):
            for j, col in enumerate(real_matrix.columns):
                if col == base_col:
                    colors[i, j, :3] = np.array([0.60, 0.60, 0.60])
                    continue
                d = diff_eff[i, j]
                if np.isnan(d) or max_abs == 0.0:
                    continue
                strength_raw = min(abs(d) / max_abs, 1.0)
                strength_raw = strength_raw**GAMMA  # boost small changes
                strength = (
                    MIN_STRENGTH + (MAX_STRENGTH - MIN_STRENGTH) * strength_raw
                )

                base_col_vec = green if d >= 0 else red
                rgb = 1.0 - strength * (1.0 - base_col_vec)
                colors[i, j, :3] = rgb

        fig1b, ax1b = plt.subplots(figsize=(20, 13))
        ax1b.imshow(colors, aspect="auto")
        ax1b.set_xticks(np.arange(n_cols))
        ax1b.set_xticklabels(real_matrix.columns, rotation=90, fontsize=24)
        ax1b.set_yticks(np.arange(n_rows))
        ax1b.set_yticklabels(real_matrix.index, fontsize=24)

        for i in range(n_rows):
            for j in range(n_cols):
                val = real_matrix.iloc[i, j]
                if pd.isna(val):
                    continue
                is_delta_row = i == 1

                ax1b.text(
                    j,
                    i,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=24,
                    fontweight="bold" if is_delta_row else "normal",
                )

        ax1b.set_title(
            f"{m_pretty} - Darker colors = larger change vs base (green better, red worse)",
            pad=35,
        )
        draw_category_separators_and_labels(ax1b, real_matrix.columns)
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        m1b_name = f"matrix1_{stub}{suffix}_realvalues_intensity_{VARIANT}.png"
        plt.savefig(os.path.join(ANALYSIS_DIR, m1b_name), dpi=300)
        plt.close(fig1b)
        print(f"{m1b_name} saved.")

        # Matrix 1c (F1 only)
        if std_matrix_for_annot is not None and stub == "f1":
            # Align std matrix to real_matrix (AVG rows have NaN Std)
            std_aligned = pd.DataFrame(
                np.nan,
                index=real_matrix.index,
                columns=real_matrix.columns,
                dtype=float,
            )
            for r in std_matrix_for_annot.index:
                if r in std_aligned.index:
                    for c in std_matrix_for_annot.columns:
                        if c in std_aligned.columns:
                            std_aligned.loc[r, c] = std_matrix_for_annot.loc[
                                r, c
                            ]

            fig1c, ax1c = plt.subplots(figsize=(20, 16))
            ax1c.imshow(colors, aspect="auto")
            ax1c.set_xticks(np.arange(n_cols))
            ax1c.set_xticklabels(real_matrix.columns, rotation=90, fontsize=24)
            ax1c.set_yticks(np.arange(n_rows))
            ax1c.set_yticklabels(real_matrix.index, fontsize=24)

            for i in range(n_rows):
                for j in range(n_cols):
                    val = real_matrix.iloc[i, j]
                    if pd.isna(val):
                        continue
                    s = std_aligned.iloc[i, j]
                    is_delta_row = i == 1
                    ax1c.text(
                        j,
                        i,
                        (
                            f"{val:.3f}\n±{s:.3f}"
                            if not pd.isna(s)
                            else f"{val:.3f}"
                        ),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=24,
                        fontweight="bold" if is_delta_row else "normal",
                    )

            draw_category_separators_and_labels(ax1c, real_matrix.columns)
            plt.tight_layout(rect=[0, 0, 1, 0.90])

            m1c_name = f"matrix1_{stub}{suffix}_realvalues_intensity_with_std_{VARIANT}.png"
            m1c_name_pdf = f"matrix1_{stub}{suffix}_realvalues_intensity_with_std_{VARIANT}.pdf"
            plt.savefig(os.path.join(ANALYSIS_DIR, m1c_name), dpi=600)
            plt.savefig(
                os.path.join(ANALYSIS_DIR, m1c_name_pdf), bbox_inches="tight"
            )
            plt.close(fig1c)
            print(f"✔ {m1c_name} saved.")

    # Matrix 2: % Change
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
    if not mask.all().all():
        fig2, ax2 = plt.subplots(figsize=(18, 8))
        cmap2 = plt.get_cmap("RdYlGn_r").copy()
        cmap2.set_bad(color="#D3D3D3")

        sns.heatmap(
            pct_for_heat,
            cmap=cmap2,
            center=0,
            annot=True,
            fmt=".1f",
            xticklabels=pct_change.columns,
            yticklabels=pct_change.index,
            mask=mask,
            ax=ax2,
            cbar=True,
        )
        ax2.set_title(f"% Change vs Base – {m_pretty}")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Classifier")
        plt.tight_layout()

        fig2_name = (
            f"matrix2_{stub}{suffix}_percentage_change_heatmap_{VARIANT}.png"
        )
        plt.savefig(os.path.join(ANALYSIS_DIR, fig2_name), dpi=300)
        plt.close(fig2)


def generate_std_outputs(std_matrix, stub, m_pretty, suffix=""):
    if std_matrix.empty:
        print(
            f"Std matrix empty for {m_pretty} (Suffix '{suffix}') – skipped."
        )
        return

    csv_name = f"{stub}_std_matrix{suffix}_{VARIANT}.csv"
    csv_path = os.path.join(ANALYSIS_DIR, csv_name)
    std_matrix.to_csv(csv_path)
    print(f"{csv_name} saved to: {csv_path}")

    mat = std_matrix.copy()
    n_rows, n_cols = mat.shape
    values = mat.to_numpy(dtype=float)
    max_val = np.nanmax(values) if np.any(~np.isnan(values)) else 0.0

    colors = np.ones((n_rows, n_cols, 4), dtype=float)
    colors[:, :, :3] = 0.93
    colors[:, :, 3] = 1.0

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v) or max_val == 0.0:
                continue
            strength = min(v / max_val, 1.0)
            dark = 0.65 * strength
            rgb = 1.0 - dark
            colors[i, j, :3] = np.array([rgb, rgb, rgb])

    fig, ax = plt.subplots(figsize=(18, 8))
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
                f"{v:.4f}",
                ha="center",
                va="center",
                color="black",
                fontsize=20,
            )

    ax.set_title(
        f"{m_pretty} – StdDev over middle 8 runs (higher = darker)", pad=20
    )
    plt.tight_layout()
    fig_name = f"matrix1_{stub}{suffix}_std_middle8_realvalues_intensity_{VARIANT}.png"
    plt.savefig(os.path.join(ANALYSIS_DIR, fig_name), dpi=300)
    plt.close(fig)


def generate_win_outputs(win_matrix, stub, m_pretty, suffix=""):
    if win_matrix.empty:
        print(
            f"Win matrix empty for {m_pretty} (Suffix '{suffix}') – skipped."
        )
        return

    csv_name = f"{stub}_wins_vs_base_matrix{suffix}_{VARIANT}.csv"
    csv_path = os.path.join(ANALYSIS_DIR, csv_name)
    win_matrix.to_csv(csv_path)
    print(f"{csv_name} saved to: {csv_path}")

    mat = win_matrix.copy()
    n_rows, n_cols = mat.shape
    values = mat.to_numpy(dtype=float)
    max_val = np.nanmax(values) if np.any(~np.isnan(values)) else 0.0

    colors = np.ones((n_rows, n_cols, 4), dtype=float)
    colors[:, :, :3] = 0.93
    colors[:, :, 3] = 1.0

    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v) or max_val == 0.0:
                continue
            strength = min(v / max_val, 1.0)
            dark = 0.75 * strength
            rgb = 1.0 - dark
            colors[i, j, :3] = np.array([rgb, rgb, rgb])

    fig, ax = plt.subplots(figsize=(18, 8))
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
                fontsize=20,
            )

    ax.set_title(
        f"{m_pretty} – #Seeds where Category > Base (higher = darker)", pad=20
    )
    plt.tight_layout()
    fig_name = f"matrix_wins_{stub}{suffix}_vs_base_{VARIANT}.png"
    plt.savefig(os.path.join(ANALYSIS_DIR, fig_name), dpi=300)
    plt.close(fig)


# 5) Collect JSON files
all_records = []

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

            all_records.append((run_suf, clf, var, data))

if not all_records:
    raise RuntimeError("No matching JSON files found across runs.")

print(f"Loaded JSON files: {len(all_records)}")

# 6) Aggregate per metric + Outputs
for metric in METRICS:
    m_pretty = metric["pretty_title"]
    better_is = metric["better_is"]
    stub = metric["file_stub"]
    sections = metric["sections"]
    key_candidates = metric["key_candidates"]

    rows = []  # (run, clf, cat, value)
    for run_suf, clf, var, data in all_records:
        val = extract_metric(data, key_candidates, sections)
        if val is None:
            continue
        cat = CATEGORY_MAPPING.get(var, var)

        ALLOWED_CATEGORIES = set(PREFERRED_COLUMN_ORDER)  # Base + Categories

        cat = CATEGORY_MAPPING.get(var, var)

        if cat not in ALLOWED_CATEGORIES:
            continue

        rows.append((run_suf, clf, cat, float(val)))

    if not rows:
        print(f"No values found for metric: {m_pretty}")
        continue

    df_raw = pd.DataFrame(
        rows, columns=["Run", "Classifier", "Category", "Value"]
    )

    if stub == "f1":
        base_col = "TRX Only"
        wide = df_raw.pivot_table(
            index=["Run", "Classifier"],
            columns="Category",
            values="Value",
            aggfunc="mean",
        )
        if base_col in wide.columns:
            base = wide[base_col]
            diffs = wide.sub(base, axis=0)
            wins_bool = diffs.gt(0)
            if base_col in wins_bool.columns:
                wins_bool[base_col] = False
            win_counts = wins_bool.groupby("Classifier").sum(min_count=1)
            win_counts = win_counts.reindex(CLASSIFIER_ORDER).dropna(how="all")
            ordered_cols = sort_columns(
                list(win_counts.columns), PREFERRED_COLUMN_ORDER
            )
            win_counts = win_counts[ordered_cols]
            if base_col in win_counts.columns:
                win_counts = win_counts.drop(columns=[base_col])
            generate_win_outputs(win_counts, stub, m_pretty, suffix="")
        else:
            print(
                f"Baseline column '{base_col}' not found – win-counts skipped."
            )

    # Aggregation Mean + Std
    agg_stats = (
        df_raw.groupby(["Classifier", "Category"])["Value"]
        .agg(Mean=trimmed_mean_middle8, Std=trimmed_std_middle8)
        .reset_index()
    )

    pivot_mean = agg_stats.pivot(
        index="Classifier", columns="Category", values="Mean"
    )
    pivot_std = agg_stats.pivot(
        index="Classifier", columns="Category", values="Std"
    )

    pivot_mean = pivot_mean.reindex(CLASSIFIER_ORDER).dropna(how="all")
    pivot_std = pivot_std.reindex(CLASSIFIER_ORDER).dropna(how="all")

    ordered_cols = sort_columns(
        list(pivot_mean.columns), PREFERRED_COLUMN_ORDER
    )
    pivot_mean = pivot_mean[ordered_cols]
    pivot_std = pivot_std[ordered_cols]

    generate_outputs(
        pivot_mean,
        stub,
        m_pretty,
        better_is,
        suffix="",
        std_matrix_for_annot=(pivot_std if stub == "f1" else None),
    )
    generate_std_outputs(pivot_std, stub, m_pretty, suffix="")

print("Done. (Category-level trimmed mean + std over the central 8 runs)")
