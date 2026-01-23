# -*- coding: utf-8 -*-
"""
RF_fix_hp.py
Random Forest variant, consistent with RF.py

- Loads artifacts from:
    <EXPERIMENTS_ROOT>/artifacts/elliptic/<ARTIFACT_DIR_NAME>
- No hyperparameter optimization:
    uses best_params from an existing full-graph JSON (RF.py)
- Replicates the final pipeline from RF.py:
    * no feature scaling (Random Forest does not require scaling)
    * apply_balancing_strategy (none / oversampling / class_weights)
    * RandomForestClassifier
    * determines threshold t∗ on the validation set
    * evaluates on the test set using t∗

Saves results to:
    <EXPERIMENTS_ROOT>/Results_JSON_<drop>_<seed>/RF/
        elliptic_RF_<ARTIFACT_DIR_NAME>.json
        elliptic_RF_<ARTIFACT_DIR_NAME>__testpreds.parquet
"""

import os

# Reproduzierbarkeit wie in RF.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np

import re
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
    find_best_threshold,
    extract_edge_tag,
    find_matching_baseline_bundle,
    load_best_params_from_baseline_run,
)

# Pfade / Konstanten
THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
BASELINE_RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "RF"

BASE_PREFIX = "base93"
EDGES_PREFIX = "_edgesVar"
RANDOM_PREFIX = "_random"


# Balancing – identisch zur Logik in RF.py
def apply_balancing_strategy(X, y, params, drop_seed):
    """
    Wendet die Balancing-Strategie an (wie in RF.py).

    params["balancing_strategy"] ∈ {"none", "oversampling", "class_weights"}
    """
    strategy = params["balancing_strategy"]

    if strategy == "none":
        # kein Balancing
        return X, y, None

    elif strategy == "oversampling":
        # 1:1 Oversampling der Minority-Klasse (Fraud=0)
        X_df = pd.DataFrame(X)
        y_df = pd.Series(y, name="label")
        df_tr = pd.concat([X_df, y_df], axis=1)

        min_df = df_tr[df_tr["label"] == 0]
        maj_df = df_tr[df_tr["label"] == 1]

        if len(min_df) == 0 or len(maj_df) == 0:
            # Sicherheits-Fallback
            X_bal = X
            y_bal = y
        else:
            min_up = resample(
                min_df,
                replace=True,
                n_samples=len(maj_df),
                random_state=drop_seed,
            )
            df_bal = pd.concat([maj_df, min_up]).sample(
                frac=1, random_state=drop_seed
            )
            X_bal = df_bal.drop(columns=["label"]).values
            y_bal = df_bal["label"].values

        return X_bal, y_bal, None

    elif strategy == "class_weights":
        # RF unterstützt class_weight
        return X, y, "balanced"

    else:
        # Fallback: unbekannte Strategie
        return X, y, None


# Kernlogik für EIN Edge-Drop-Artifact
def run_for_artifact_edge(artifact_dir_name: str) -> str:
    artifact_dir = ARTIFACT_ROOT / artifact_dir_name
    print("\n" + "=" * 80)
    print(f"[RF Edge-Drop] Artifact: {artifact_dir_name}")
    print("=" * 80)

    if "edgesVar" not in artifact_dir_name:
        print(
            "  -> Kein Edge-Drop-Artifact (enthält 'edgesVar' nicht). Überspringe."
        )
        return ""

    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Artifact-Ordner nicht gefunden: {artifact_dir}"
        )

    if not has_required_files(artifact_dir):
        raise FileNotFoundError(
            f"Artefakt-Dateien unvollständig: {artifact_dir}"
        )

    variant, seed = extract_edge_tag(artifact_dir_name)
    if variant is None:
        raise ValueError(
            f"Konnte edge-tag nicht extrahieren aus: {artifact_dir_name}"
        )
    edge_tag = f"{variant}_{seed}"

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Ergebnisordner analog SVC/NB/LR/MLP-Edge:
    # <EXPERIMENTS_ROOT>/Results_JSON_<drop>_<seed>/RF
    edge_results_root = (
        EXPERIMENTS_ROOT / "Results_HyPa_fix" / f"Results_{edge_tag}" / "RF"
    )
    edge_results_root.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Ergebnisse werden gespeichert unter: {edge_results_root}")

    # Daten laden (wie in RF.py)
    X_train = _read_parquet_required(artifact_dir / "X_train.parquet")
    X_val = _read_parquet_required(artifact_dir / "X_validation.parquet")
    X_test = _read_parquet_required(artifact_dir / "X_test.parquet")

    y_train = pd.read_parquet(artifact_dir / "y_train.parquet").iloc[:, 0]
    y_val = pd.read_parquet(artifact_dir / "y_validation.parquet").iloc[:, 0]
    y_test = pd.read_parquet(artifact_dir / "y_test.parquet").iloc[:, 0]

    # txId sichern
    if "txId" in X_test.columns:
        txid_test = X_test["txId"].copy()
    else:
        txid_test = pd.Series(np.arange(len(X_test)), name="txId")

    # txId/time_step droppen
    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Labels als int
    y_train = pd.Series(y_train).astype(int)
    y_val = pd.Series(y_val).astype(int)
    y_test = pd.Series(y_test).astype(int)

    print("\n[Dataset shapes (edge-drop)]")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print("\n[Class distribution Train] (0=Fraud,1=No-Fraud)")
    print(y_train.value_counts().sort_index())

    # Full-Graph-JSON + best_params laden
    baseline_bundle_path = find_matching_baseline_bundle(
        artifact_dir_name=artifact_dir_name,
        baseline_results_dir=BASELINE_RESULTS_DIR,
        model_tag="RF",
    )

    best_params = load_best_params_from_baseline_run(baseline_bundle_path)

    print("\n[best_params aus Full-Graph-Run]")
    print(best_params)

    # Arrays für RF (kein Scaling)
    Xtr = X_train.values
    Xva = X_val.values
    Xte = X_test.values

    # TRAIN+VAL zusammen, Balancing wie in RF.py
    Xtv = np.vstack([Xtr, Xva])
    ytv = np.concatenate([y_train.values, y_val.values])

    Xtv_bal, ytv_bal, class_weights_tv = apply_balancing_strategy(
        Xtv, ytv, best_params, seed
    )

    # max_features / max_depth mappen wie in RF.py
    max_features_val = best_params["max_features"]
    if max_features_val == "None":
        max_features_val = None

    if best_params["max_depth"] in [None, "None"]:
        max_depth_val = None
    else:
        max_depth_val = int(best_params["max_depth"])

    # Finalen RF auf TRAIN+VAL fitten
    clf = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=max_depth_val,
        min_samples_split=int(best_params["min_samples_split"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        max_features=max_features_val,
        class_weight=class_weights_tv,
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )

    t0 = time.time()
    clf.fit(Xtv_bal, ytv_bal)
    t1 = time.time()
    print(f"\n[RF Edge-Drop] Fit (TRAIN+VAL) Dauer: {t1 - t0:.2f}s")

    # Threshold t* auf Validation bestimmen
    proba_val_full = clf.predict_proba(Xva)
    fraud_idx_val = int(np.where(clf.classes_ == 0)[0][0])

    t_star, f1_val_star = find_best_threshold(
        y_val.values,
        proba_val_full[:, fraud_idx_val],
        pos_label=0,
    )
    print(f"[VAL] t* = {t_star:.4f} | F1@t* = {f1_val_star:.4f}")

    # Test-Evaluation
    proba_test = clf.predict_proba(Xte)
    fraud_idx_test = int(np.where(clf.classes_ == 0)[0][0])
    proba_fraud_te = proba_test[:, fraud_idx_test]
    pred_test = np.where(proba_fraud_te >= t_star, 0, 1)

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test, labels=clf.classes_)
    acc = accuracy_score(y_test, pred_test)

    roc_auc = roc_auc_score(
        (y_test.values == 0).astype(int),
        proba_test[:, fraud_idx_test],
    )
    pr_auc = average_precision_score(
        y_test,
        proba_test[:, fraud_idx_test],
        pos_label=0,
    )

    print("\n=== Test Results (Fraud=0, Edge-Drop, RF) ===")
    print(
        f"F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | "
        f"BalancedAcc={bacc:.4f} | LogLoss={ll:.4f} | Acc={acc:.4f} | "
        f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f}"
    )
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred_test))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            pred_test,
            target_names=["Fraud", "No Fraud"],
            zero_division=0,
        )
    )

    # Test-Predictions speichern
    proba_fraud = proba_fraud_te
    proba_nofraud = (
        proba_test[:, 1 - fraud_idx_test]
        if proba_test.shape[1] > 1
        else 1.0 - proba_fraud
    )

    df_preds = pd.DataFrame(
        {
            "txId": txid_test.reset_index(drop=True),
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(pred_test).reset_index(drop=True),
            "proba_fraud": proba_fraud,
            "proba_nofraud": proba_nofraud,
        }
    )

    preds_path = (
        edge_results_root
        / f"elliptic_RF_{artifact_dir_name}__testpreds.parquet"
    )
    df_preds.to_parquet(preds_path, index=False)
    print(f"Test-Predictions gespeichert unter: {preds_path}")

    # JSON-Export – Struktur wie in RF.py (ohne Hyperopt-/FI-Details im Edge-Run)
    results_summary = {
        "run_info": {
            "model": "RandomForestClassifier",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": seed,
            "balancing_strategy": best_params.get("balancing_strategy"),
            "scaling": "none (RF)",
            "dim_reduction": "none",
            "final_fit": "train+val (edge-drop)",
            "decision_threshold_fraud": float(t_star),
            "val_f1_at_threshold": float(f1_val_star),
            "edge_drop": {
                "variant": variant,
                "seed": int(seed) if seed is not None else None,
            },
            "edge_tag": edge_tag,
        },
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_validation": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "class_distribution_train": {
            "Fraud(0)": int((y_train == 0).sum()),
            "NoFraud(1)": int((y_train == 1).sum()),
        },
        "best_params": _to_jsonable(best_params),
        "test_metrics": {
            "f1_fraud": float(f1),
            "precision_fraud": float(prec),
            "recall_fraud": float(rec),
            "balanced_accuracy": float(bacc),
            "log_loss": float(ll),
            "accuracy": float(acc),
            "roc_auc_fraud": float(roc_auc),
            "pr_auc_fraud": float(pr_auc),
        },
    }

    json_path = edge_results_root / f"elliptic_RF_{artifact_dir_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"JSON-Results gespeichert unter: {json_path}")

    return str(json_path)


# CLI – Single & Batch
def main():
    print("=" * 80)
    print("RandomForest Edge-Drop Evaluation – konsistent zu RF.py")
    print("=" * 80)

    ap = argparse.ArgumentParser(
        description="RandomForest Elliptic – Edge-Drop Runner (ohne Hyperopt, Single & Batch)"
    )
    ap.add_argument(
        "--artifact",
        help="Name eines Edge-Drop-Artefakt-Unterordners unter artifacts/elliptic",
    )
    ap.add_argument(
        "--folder",
        help="Ordner mit Artefakt-Unterordnern (Default: artifacts/elliptic)",
    )
    ap.add_argument(
        "--pattern",
        help="Substring-Filter für Artefakt-Namen (z.B. 'edges25_42')",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Überspringt Artefakte, für die bereits ein Edge-Drop-JSON existiert",
    )
    args = ap.parse_args()

    global ARTIFACT_ROOT
    if args.folder:
        ARTIFACT_ROOT = Path(args.folder).resolve()
        print(f"[INFO] ARTIFACT_ROOT überschrieben auf: {ARTIFACT_ROOT}")

    # Einzel-Run
    if args.artifact:
        run_for_artifact_edge(args.artifact)
        return

    # Batch-Modus
    root = Path(args.folder) if args.folder else ARTIFACT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Ordner nicht gefunden: {root}")

    candidates = []
    for name in sorted(os.listdir(root)):
        p = root / name
        if not p.is_dir():
            continue

        # Nur Edge-Drop-Artefakte mit korrektem Namensschema
        variant, seed = extract_edge_tag(name)
        if variant is None:
            continue
        edge_tag = f"{variant}_{seed}"

        if args.pattern and args.pattern not in name:
            continue

        if not has_required_files(p):
            continue

        if args.skip_existing:
            res_dir = (
                EXPERIMENTS_ROOT
                / "Results_HyPa_fix"
                / f"Results_{edge_tag}"
                / "RF"
            )
            json_out = res_dir / f"elliptic_RF_{name}.json"
            if json_out.exists():
                continue

        candidates.append(name)

    if not candidates:
        print("Keine passenden Edge-Drop-Artefakte gefunden.")
        return

    print(f"Starte RF-Edge-Drop-Runs für {len(candidates)} Artefakte…")
    for name in candidates:
        try:
            path = run_for_artifact_edge(name)
            print(f"Fertig: {name} -> {path}")
        except Exception as e:
            print(f"[WARN] Fehler bei Artifact {name}: {e}")


if __name__ == "__main__":
    main()
