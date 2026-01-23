# -*- coding: utf-8 -*-
"""
Elliptic – XGBoost (XGBClassifier) variant,
1:1 consistent with the final pipeline from XGB.py

- Loads artifacts from:
    <EXPERIMENTS_ROOT>/artifacts/elliptic/<ARTIFACT_DIR_NAME>
- Locates the corresponding full-graph XGB JSON under:
    <EXPERIMENTS_ROOT>/Results_JSON/XGB/elliptic_XGB_*.json
    (matching is done via the shared base93 block in the filename)
- Uses the stored best_params, but:
    * no Hyperopt is performed,
    * best_iter and the decision threshold t∗ are recomputed on the artifact,
      using exactly the same training logic as in XGB.py:
          1. temporary fit with early stopping on TRAIN (eval_set = VAL) → best_iter, t∗
          2. final fit on TRAIN+VAL with n_estimators = best_iter + 1,
             using the same balancing strategy and sample weights as in XGB.py
- Saves results to:
    <EXPERIMENTS_ROOT>/Results_JSON_<drop>_<seed>/XGB/
        elliptic_XGB_<ARTIFACT_DIR_NAME>.json
        elliptic_XGB_<ARTIFACT_DIR_NAME>__testpreds.parquet
"""

import os

# Reproduzierbarkeit
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np

import re
import json
import time
from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd

from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
    find_best_threshold,
    extract_edge_tag,
    find_matching_baseline_bundle,
    load_best_params_from_baseline_run,
)

# Pfade & Basis-Konfiguration
THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
BASELINE_RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "XGB"
BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_PREFIX = "base93"
EDGES_PREFIX = "_edgesVar"
RANDOM_PREFIX = "_random"


# Balancing (identisch zu XGB.py)
def apply_balancing_strategy(X, y, params, drop_seed):
    """
    Wendet die Balancing-Strategie aus best_params auf (X, y) an.
    Exakt wie in XGB.py: 'none' oder 'oversampling'.
    Rückgabe: (X_bal, y_bal, scale_pos_weight) – letzteres ungenutzt.
    """
    strategy = params["balancing_strategy"]

    if strategy == "none":
        # Kein Balancing - originale Daten
        return X, y, None

    elif strategy == "oversampling":
        # Sichere 1:1 Oversampling-Variante: minority=Label 0 (Fraud)
        X_df = pd.DataFrame(X)
        y_df = pd.Series(y, name="label")
        df_tr = pd.concat([X_df, y_df], axis=1)

        min_df = df_tr[df_tr["label"] == 0]
        maj_df = df_tr[df_tr["label"] == 1]

        if len(min_df) == 0 or len(maj_df) == 0:
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

    else:
        # Fallback: unveränderte Daten
        return X, y, None


def run_for_artifact_edge(artifact_dir_name: str) -> str:
    artifact_dir = ARTIFACT_ROOT / artifact_dir_name

    print("\n" + "=" * 80)
    print(f"[XGB Edge-Drop] Artifact: {artifact_dir_name}")
    print("=" * 80)

    if "edgesVar" not in artifact_dir_name:
        print(
            "  -> Kein Edge-Drop-Artifact (kein 'edgesVar' im Namen). Überspringe."
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
    if not variant:
        raise ValueError(
            f"Konnte edge_tag nicht extrahieren aus: {artifact_dir_name}"
        )
    edge_tag = f"{variant}_{seed}"

    # global seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Ergebnisordner: Results_JSON_<drop>_<seed>/XGB
    edge_results_root = (
        EXPERIMENTS_ROOT / "Results_HyPa_fix" / f"Results_{edge_tag}" / "XGB"
    )
    edge_results_root.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Ergebnisse werden gespeichert unter: {edge_results_root}")

    out_json_path = (
        edge_results_root / f"elliptic_XGB_{artifact_dir_name}.json"
    )
    if out_json_path.exists():
        print(f"  -> JSON existiert bereits, überspringe: {out_json_path}")
        return str(out_json_path)

    # Daten laden – identisch zu XGB.py
    X_train = _read_parquet_required(artifact_dir / "X_train.parquet")
    X_val = _read_parquet_required(artifact_dir / "X_validation.parquet")
    X_test = _read_parquet_required(artifact_dir / "X_test.parquet")

    y_train = pd.read_parquet(artifact_dir / "y_train.parquet").iloc[:, 0]
    y_val = pd.read_parquet(artifact_dir / "y_validation.parquet").iloc[:, 0]
    y_test = pd.read_parquet(artifact_dir / "y_test.parquet").iloc[:, 0]

    # txId für spätere Predictions
    if "txId" in X_test.columns:
        txid_test = X_test["txId"].copy()
    else:
        txid_test = pd.Series(np.arange(len(X_test)), name="txId")

    # txId / time_step entfernen
    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Indizes angleichen
    if y_train.index.equals(pd.RangeIndex(len(y_train))):
        y_train.index = X_train.index
    if y_val.index.equals(pd.RangeIndex(len(y_val))):
        y_val.index = X_val.index
    if y_test.index.equals(pd.RangeIndex(len(y_test))):
        y_test.index = X_test.index

    # Labels als int
    y_train = pd.to_numeric(y_train, errors="coerce").astype(int)
    y_val = pd.to_numeric(y_val, errors="coerce").astype(int)
    y_test = pd.to_numeric(y_test, errors="coerce").astype(int)

    print("\n[Dataset shapes (edge-drop)]")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print("\n[Class distribution Train] (0=Fraud,1=No-Fraud)")
    print(y_train.value_counts().sort_index())

    # Numpy-Arrays für XGB
    Xtr, Xva, Xte = X_train.values, X_val.values, X_test.values

    # Full-Graph-JSON & best_params laden
    baseline_bundle_path = find_matching_baseline_bundle(
        artifact_dir_name=artifact_dir_name,
        baseline_results_dir=BASELINE_RESULTS_DIR,
        model_tag="XGB",
    )

    best_params = load_best_params_from_baseline_run(baseline_bundle_path)

    print("\n[best_params aus Full-Graph-XGB]")
    print(best_params)

    # A) tmp-Fit mit Early Stopping (TRAIN) -> best_iter, t*
    Xtr_bal, ytr_bal, _ = apply_balancing_strategy(
        Xtr, y_train.values, best_params, seed
    )

    ratio_tr = (ytr_bal == 1).sum() / max(1, (ytr_bal == 0).sum())
    sw_tr = np.where(ytr_bal == 0, ratio_tr, 1.0)

    ratio_val = (y_val.values == 1).sum() / max(1, (y_val.values == 0).sum())
    sw_val = np.where(y_val.values == 0, ratio_val, 1.0)

    clf_tmp = XGBClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        learning_rate=float(best_params["learning_rate"]),
        subsample=float(best_params["subsample"]),
        colsample_bytree=float(best_params["colsample_bytree"]),
        min_child_weight=float(best_params["min_child_weight"]),
        gamma=float(best_params["gamma"]),
        reg_alpha=float(best_params["reg_alpha"]),
        reg_lambda=float(best_params["reg_lambda"]),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        scale_pos_weight=1.0,
    )

    t0 = time.time()
    clf_tmp.fit(
        Xtr_bal,
        ytr_bal,
        eval_set=[(Xva, y_val)],
        early_stopping_rounds=50,
        verbose=False,
        sample_weight=sw_tr.astype(np.float32),
        sample_weight_eval_set=[sw_val.astype(np.float32)],
    )
    t1 = time.time()
    print(
        f"[XGB Edge-Drop] tmp-Fit (TRAIN, EarlyStopping) Dauer: {t1 - t0:.2f}s"
    )

    proba_val_full = clf_tmp.predict_proba(Xva)
    fraud_idx_val = int(np.where(clf_tmp.classes_ == 0)[0][0])
    t_star, f1_val_star = find_best_threshold(
        y_val.values,
        proba_val_full[:, fraud_idx_val],
        pos_label=0,
    )
    best_iter = int(
        getattr(clf_tmp, "best_iteration_", int(best_params["n_estimators"]))
    )
    print(
        f"[VAL] t* = {t_star:.4f} | F1@t* = {f1_val_star:.4f} | best_iter={best_iter}"
    )

    # B) Finaler Fit auf TRAIN+VAL ohne Early Stopping
    Xtv = np.concatenate([Xtr, Xva], axis=0)
    ytv = np.concatenate([y_train.values, y_val.values], axis=0)

    Xtv_bal, ytv_bal, _ = apply_balancing_strategy(Xtv, ytv, best_params, seed)

    ratio_tv = (ytv_bal == 1).sum() / max(1, (ytv_bal == 0).sum())
    sw_tv = np.where(ytv_bal == 0, ratio_tv, 1.0)

    clf = XGBClassifier(
        n_estimators=int(best_iter) + 1,
        max_depth=int(best_params["max_depth"]),
        learning_rate=float(best_params["learning_rate"]),
        subsample=float(best_params["subsample"]),
        colsample_bytree=float(best_params["colsample_bytree"]),
        min_child_weight=float(best_params["min_child_weight"]),
        gamma=float(best_params["gamma"]),
        reg_alpha=float(best_params["reg_alpha"]),
        reg_lambda=float(best_params["reg_lambda"]),
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        scale_pos_weight=1.0,
    )

    print("\n[XGB Edge-Drop] Finales Modell wird auf TRAIN+VAL gefittet ...")
    clf.fit(Xtv_bal, ytv_bal, sample_weight=sw_tv)

    # TEST-Evaluation – identische Metriken wie in XGB.py
    proba_test = clf.predict_proba(Xte)
    fraud_idx_test = int(np.where(clf.classes_ == 0)[0][0])
    proba_fraud = proba_test[:, fraud_idx_test]

    pred_test = np.where(proba_fraud >= t_star, 0, 1)

    f1_te = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec_te = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec_te = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc_te = balanced_accuracy_score(y_test, pred_test)
    ll_te = log_loss(y_test, proba_test, labels=clf.classes_)
    acc_te = accuracy_score(y_test, pred_test)

    roc_auc_te = roc_auc_score((y_test.values == 0).astype(int), proba_fraud)
    pr_auc_te = average_precision_score(y_test, proba_fraud, pos_label=0)

    print("\n=== FINAL TEST EVALUATION (XGB Edge-Drop) ===")
    print(
        classification_report(
            y_test, pred_test, target_names=["Fraud(0)", "No-Fraud(1)"]
        )
    )
    print(f"Balanced Accuracy: {bacc_te:.4f}")
    print("########## Test ##########")
    print(f"F1-Score     = {f1_te:.4f}")
    print(f"Precision    = {prec_te:.4f}")
    print(f"Recall       = {rec_te:.4f}")
    print(f"Accuracy     = {acc_te:.4f}")
    print(f"Log Loss     = {ll_te:.4f}")
    print(f"ROC-AUC      = {roc_auc_te:.4f}")
    print(f"PR-AUC       = {pr_auc_te:.4f}\n")

    print("Confusion Matrix (TEST):")
    print(confusion_matrix(y_test, pred_test))

    # Test-Predictions speichern
    try:
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
            / f"elliptic_XGB_{artifact_dir_name}__testpreds.parquet"
        )
        df_preds.to_parquet(preds_path, index=False)
        print(
            f"Test-Predictions (edge-drop, XGB) gespeichert unter: {preds_path}"
        )
    except Exception as e:
        print(f"Konnte Test-Predictions nicht speichern: {e}")

    # JSON-Export – Struktur an XGB.py angelehnt
    out = {
        "artifact_dir": artifact_dir_name,
        "edge_drop": {
            "seed": int(seed) if seed is not None else None,
            "variant": variant,
        },
        "best_params": _to_jsonable(best_params),
        "validation": {
            "best_threshold_t": float(t_star),
            "f1_at_best_t_val": float(f1_val_star),
            "best_iteration": int(best_iter),
        },
        "metrics_test": {
            "f1": float(f1_te),
            "precision": float(prec_te),
            "recall": float(rec_te),
            "balanced_accuracy": float(bacc_te),
            "accuracy": float(acc_te),
            "log_loss": float(ll_te),
            "roc_auc_fraud": float(roc_auc_te),
            "pr_auc_fraud": float(pr_auc_te),
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nEdge-Drop-XGB-Results saved to JSON: {out_json_path}")
    return str(out_json_path)


# CLI – Single & Batch
def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Elliptic – Edge-Drop Runner (ohne Hyperopt, konsistent zu XGB.py)"
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Name eines Edge-Drop-Artefakt-Unterordners unter artifacts/elliptic",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Ordner mit Artefakt-Unterordnern (Default: artifacts/elliptic)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Substring-Filter für Artefakt-Namen (z.B. 'edges25_42' oder 'Role2Vec')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Überspringt Artefakte, für die bereits ein XGB-Edge-Drop-JSON existiert",
    )
    args = parser.parse_args()

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

        variant, seed = extract_edge_tag(name)
        if not variant:
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
                / "XGB"
            )
            json_out = res_dir / f"elliptic_XGB_{name}.json"
            if json_out.exists():
                continue

        candidates.append(name)

    if not candidates:
        print("Keine passenden Edge-Drop-Artefakte gefunden.")
        return

    print(f"Starte XGB-Edge-Drop-Runs für {len(candidates)} Artefakte…")
    for dname in candidates:
        try:
            path = run_for_artifact_edge(dname)
            print(f"Fertig: {dname} -> {path}")
        except Exception as e:
            print(f"Fehler bei {dname}: {e}")


if __name__ == "__main__":
    main()
