"""
Utility functions shared across multiple classifier scripts.

Design goals:
- no dataset-specific naming
- generic, reusable helpers
"""

from __future__ import annotations

import re
import os
import numpy as np
import pandas as pd
import json
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from pathlib import Path
from typing import Any


# File / artifact handling
def has_required_files(path: str) -> bool:
    """Checks whether all required Parquet files for Train/Val/Test are present."""
    req = {
        "X_train.parquet",
        "X_validation.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_validation.parquet",
        "y_test.parquet",
    }
    try:
        return req.issubset(set(os.listdir(path)))
    except FileNotFoundError:
        return False


def _read_parquet_required(path: str) -> pd.DataFrame:
    """Reads a Parquet file and raises a clear error message on failure."""
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Error reading {path}. Ensure that 'pyarrow' or 'fastparquet' "
            f"is installed. Original error: {e}"
        ) from e


def extract_edge_tag(artifact_dir_name: str):
    """
    Supports:
      ..._edgesVar_<variant>_splitseed<seed>_random...
    Returns: (variant, seed) or (None, None)
    """
    m = re.search(
        r"_edgesVar_([^_]+)_splitseed(\d+)_random", artifact_dir_name
    )
    if m:
        variant = m.group(1)
        seed = int(m.group(2))
        return variant, seed
    return None, None


def extract_base_block(
    name: str,
    *,
    base_prefix: str = "base93",
    edges_prefix: str = "_edgesVar_",
    random_prefix: str = "_random",
) -> str:
    """
    Extract the base feature block from an artifact or bundle name.

    Examples:
      base93+noInd+0Spectral_random_splitseed3
        -> base93+noInd+0Spectral_random_splitseed3

      base93+noInd+0Spectral_edgesVar_25_splitseed3_random
        -> base93+noInd+0Spectral
    """
    start = name.find(base_prefix)
    if start == -1:
        raise ValueError(f"'{base_prefix}' not found in name: {name}")

    # 1) edge-drop: stop before edges_prefix
    end = name.find(edges_prefix, start)
    if end != -1:
        return name[start:end]

    # 2) baseline: stop before _random (so base features match edge-drop base block)
    end = name.find(random_prefix, start)
    if end != -1:
        return name[start:end]

    # 3) fallback: stop before __bundle if present
    end = name.find("__bundle", start)
    if end != -1:
        return name[start:end]

    return name[start:]


def find_matching_baseline_bundle(
    *,
    artifact_dir_name: str,
    baseline_results_dir: Path,
    model_tag: str,
    base_prefix: str = "base93",
    edges_prefix: str = "_edgesVar_",
) -> Path:
    """
    Find the UNIQUE baseline joblib bundle matching an edgesVar artifact.

    Matching logic:
      - Extract base block from artifact_dir_name
      - Extract base block from each baseline bundle filename
      - Compare base blocks for equality
      - Expect exactly ONE match

    Raises:
      - FileNotFoundError if no match
      - RuntimeError if multiple matches
    """
    baseline_results_dir = Path(baseline_results_dir)

    base_block_artifact = extract_base_block(
        artifact_dir_name,
        base_prefix=base_prefix,
        edges_prefix=edges_prefix,
    )

    pattern = f"elliptic_{model_tag}_*__bundle.joblib"
    candidates = sorted(baseline_results_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            f"No baseline bundles found using pattern '{pattern}' in {baseline_results_dir}"
        )

    matches = []
    for p in candidates:
        base_block_bundle = extract_base_block(
            p.name,
            base_prefix=base_prefix,
            edges_prefix=edges_prefix,
        )

        if base_block_bundle == base_block_artifact:
            matches.append(p)

    if len(matches) == 0:
        raise FileNotFoundError(
            "No matching baseline bundle found.\n"
            f"  artifact_dir_name = {artifact_dir_name}\n"
            f"  base_block = {base_block_artifact}\n"
            f"  searched in = {baseline_results_dir}\n"
            f"  pattern = {pattern}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple matching baseline bundles found (expected exactly one):\n"
            f"  artifact_dir_name = {artifact_dir_name}\n"
            f"  base_block = {base_block_artifact}\n"
            f"  matches = {[m.name for m in matches]}"
        )

    return matches[0]


def load_best_params_from_baseline_run(baseline_dir: str | Path) -> dict:
    baseline_dir = Path(baseline_dir)

    def _load_json(p: Path) -> dict:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "best_params" not in data:
            raise KeyError(f"'best_params' not found in JSON: {p}")
        return data["best_params"]

    # If a file is passed
    if baseline_dir.is_file():
        if baseline_dir.suffix == ".joblib":
            bundle = joblib.load(baseline_dir)
            if isinstance(bundle, dict) and "best_params" in bundle:
                return bundle["best_params"]

            # Fallback: try matching JSON next to the bundle
            stem = baseline_dir.name
            stem = stem.replace("__bundle.joblib", "")  # typical pattern
            json_candidates = [
                baseline_dir.with_name(stem + ".json"),
                baseline_dir.with_suffix(".json"),
            ]
            for jp in json_candidates:
                if jp.exists():
                    return _load_json(jp)

            # Fallback: search in the same folder for a "closest" json
            folder = baseline_dir.parent
            pref = stem  # already without "__bundle"
            hits = sorted(folder.glob(pref + "*.json"))
            if hits:
                return _load_json(hits[0])

            raise FileNotFoundError(
                f"No best_params in bundle and no matching JSON found next to: {baseline_dir}"
            )

        if baseline_dir.suffix == ".json":
            return _load_json(baseline_dir)

        raise ValueError(f"Unsupported file type: {baseline_dir}")

    # If a directory is passed
    # 1) Try bundle(s)
    for bundle_path in sorted(baseline_dir.glob("*.joblib")):
        try:
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict) and "best_params" in bundle:
                return bundle["best_params"]
        except Exception:
            pass

    # 2) Try JSON(s)
    json_candidates = sorted(baseline_dir.glob("*.json"))
    if not json_candidates:
        raise FileNotFoundError(
            f"No *.joblib with best_params and no *.json found in: {baseline_dir}"
        )
    return _load_json(json_candidates[0])


# JSON / serialization helpers
def _to_jsonable(x):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return str(x)


# Feature grouping & PCA on embeddings
def _split_groups(
    df: pd.DataFrame, gi_prefix: str = "gi_", emb_prefix: str = "emb_"
):
    """Splits columns into base, GI, and embedding groups."""
    gi_cols = [
        c for c in df.columns if isinstance(c, str) and c.startswith(gi_prefix)
    ]
    emb_cols = [
        c
        for c in df.columns
        if isinstance(c, str) and c.startswith(emb_prefix)
    ]
    base_cols = [
        c
        for c in df.columns
        if c not in gi_cols + emb_cols
        and c not in ["txId", "time_step", "class"]
    ]
    return base_cols, gi_cols, emb_cols


def _apply_emb_pca(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    use_emb_pca: bool,
    emb_pca_n,
    scale_before_pca: bool = True,
    emb_prefix: str = "emb_",
):
    """
    Optional: Apply PCA only on embedding columns (emb_*).

    - Fit PCA on TRAIN, apply transform to VAL/TEST
    - Replace emb_* columns with emb_pca_* and return updated DataFrames + meta information
    """
    # Identify embedding columns
    _, _, emb_cols = _split_groups(X_train_df, emb_prefix=emb_prefix)
    if not use_emb_pca or len(emb_cols) == 0:
        return (
            X_train_df,
            X_val_df,
            X_test_df,
            {
                "pca_used": False,
                "k": 0,
                "explained": None,
            },
        )

    # Scale embeddings (train-based), then PCA
    if scale_before_pca:
        mu = X_train_df[emb_cols].mean()
        sd = X_train_df[emb_cols].std(ddof=0).replace(0, 1.0)
        Ztr = ((X_train_df[emb_cols] - mu) / sd).fillna(0.0).to_numpy()
        Zva = ((X_val_df[emb_cols] - mu) / sd).fillna(0.0).to_numpy()
        Zte = ((X_test_df[emb_cols] - mu) / sd).fillna(0.0).to_numpy()
    else:
        mu, sd = None, None
        Ztr = X_train_df[emb_cols].fillna(0.0).to_numpy()
        Zva = X_val_df[emb_cols].fillna(0.0).to_numpy()
        Zte = X_test_df[emb_cols].fillna(0.0).to_numpy()

    # Fit PCA (sklearn.PCA supports both int and float types)
    pca = PCA(n_components=emb_pca_n, random_state=42)

    Etr = pca.fit_transform(Ztr)
    Eva = pca.transform(Zva)
    Ete = pca.transform(Zte)

    k = Etr.shape[1]
    emb_pca_cols = [f"{emb_prefix}pca_{i}" for i in range(k)]

    def _replace(df: pd.DataFrame, E: np.ndarray) -> pd.DataFrame:
        # Keep base + GI columns unchanged, replace embedding columns with PCA columns
        base_cols, gi_cols, _ = _split_groups(
            df, gi_prefix="gi_", emb_prefix=emb_prefix
        )
        out = pd.concat(
            [
                df[base_cols + gi_cols].reset_index(drop=True),
                pd.DataFrame(E, columns=emb_pca_cols, index=df.index),
            ],
            axis=1,
        )
        return out

    Xtr_df = _replace(X_train_df, Etr)
    Xva_df = _replace(X_val_df, Eva)
    Xte_df = _replace(X_test_df, Ete)

    meta = {
        "pca_used": True,
        "k": k,
        "explained": float(np.sum(pca.explained_variance_ratio_)),
        "emb_cols": emb_cols,
        "mu": mu if scale_before_pca else None,
        "sd": sd if scale_before_pca else None,
        "pca_model": pca,
        "scale_before_pca": scale_before_pca,
    }
    return Xtr_df, Xva_df, Xte_df, meta


# Threshold search & F1 scorer
def find_best_threshold(
    y_true: np.ndarray,
    proba_fraud: np.ndarray,
    pos_label: int = 0,
):
    """Searches for a threshold t in [0.01, 0.99] that maximizes the F1 score."""
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        y_pred = np.where(proba_fraud >= t, pos_label, 1 - pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def f1_at_t_scorer_factory(t: float, pos_label: int = 0):
    """Creates a scorer computing F1(y_true, y_hat(t)) based on predict_proba."""

    def _scorer(estimator, X, y_true):
        classes = getattr(estimator, "classes_", np.array([0, 1]))
        # Try to locate the class 'pos_label' explicitly,
        # fallback: index 0 (e.g., normal binary case)
        idx_candidates = np.where(classes == pos_label)[0]
        idx = int(idx_candidates[0]) if len(idx_candidates) > 0 else 0

        proba = estimator.predict_proba(X)[:, idx]
        yhat = np.where(proba >= t, pos_label, 1 - pos_label)
        return f1_score(y_true, yhat, pos_label=pos_label, zero_division=0)

    return _scorer
