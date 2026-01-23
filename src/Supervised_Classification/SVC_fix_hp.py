# -*- coding: utf-8 -*-
"""
SVC_fix_hp.py

For each artifact, the script:
- locates the corresponding full-graph SVC bundle,
- loads its best_params,
- refits the model on TRAIN+VAL using exactly the same pipeline as in SVC.py,
- determines the decision threshold t∗ on the validation set,
- evaluates on the test set,
- saves results and test predictions.

Output structure
- Results are written to:
    <EXPERIMENTS_ROOT>/Results_JSON_<drop>_<seed>/SVC
- The JSON file contains test metrics under "metrics_test" using the same
  key names as in SVC.py ("f1_fraud", "roc_auc_fraud", …), ensuring that
  collect_scores_V.1.9.py can ingest them without any issues.
"""

import os
import re
import gc
import json
import time
import random
from pathlib import Path
from datetime import datetime
import argparse

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    log_loss,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
    _apply_emb_pca,
    find_best_threshold,
    extract_edge_tag,
    find_matching_baseline_bundle,
    load_best_params_from_baseline_run,
)

# Globale Seeds / Threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Pfade
THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
BASELINE_RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "SVC"

BASE_PREFIX = "base93"
EDGES_PREFIX = "_edgesVar"
RANDOM_PREFIX = "_random"


# Hilfsfunktionen
def set_global_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _transform_emb_pca_with_fitted(df, pca_meta, emb_prefix="emb_"):
    """
    Identische Logik wie in SVC.py:
    Wendet eine bereits gefittete PCA-Transformation (pca_meta)
    auf ein neues DataFrame an.
    """
    if not pca_meta.get("pca_used", False):
        return df

    emb_cols = pca_meta["emb_cols"]
    if not emb_cols:
        return df

    mu, sd = pca_meta["mu"], pca_meta["sd"]
    pca = pca_meta["pca_model"]

    Z = ((df[emb_cols] - mu) / sd).fillna(0.0).to_numpy()
    E = pca.transform(Z)
    pca_cols = [f"{emb_prefix}pca_{i+1}" for i in range(E.shape[1])]

    out = df.drop(columns=emb_cols, errors="ignore").copy()
    for i, c in enumerate(pca_cols):
        out[c] = E[:, i].astype("float32")

    return out


def apply_balancing_strategy(X, y, params, seed):
    """
    1:1 Kopie der Balancing-Logik aus SVC.py.

    Erwartet:
      params["balancing_strategy"] in {"none", "oversampling", "class_weights"}
    """
    strategy = params["balancing_strategy"]

    if strategy == "none":
        return (
            X.astype(np.float32, copy=False),
            np.asarray(y, dtype=np.int32),
            None,
        )

    elif strategy == "oversampling":
        X_df = pd.DataFrame(X, dtype=np.float32)
        y_df = pd.Series(y, name="label")
        df_tr = pd.concat([X_df, y_df], axis=1)

        min_df = df_tr[df_tr["label"] == 0]
        maj_df = df_tr[df_tr["label"] == 1]

        df_bal = None
        if len(min_df) == 0 or len(maj_df) == 0:
            X_bal = np.asarray(X, dtype=np.float32)
            y_bal = np.asarray(y, dtype=np.int32)
        else:
            from sklearn.utils import resample

            min_up = resample(
                min_df, replace=True, n_samples=len(maj_df), random_state=seed
            )
            df_bal = pd.concat([maj_df, min_up]).sample(
                frac=1, random_state=seed
            )

            X_bal = df_bal.drop(columns=["label"]).to_numpy(dtype=np.float32)
            y_bal = df_bal["label"].to_numpy(dtype=np.int32)

        del df_tr, X_df, y_df, min_df, maj_df, df_bal
        return X_bal, y_bal, None

    elif strategy == "class_weights":
        # SVC-interne Klassen-Gewichtung
        return (
            X.astype(np.float32, copy=False),
            np.asarray(y, dtype=np.int32),
            "balanced",
        )

    else:
        return (
            X.astype(np.float32, copy=False),
            np.asarray(y, dtype=np.int32),
            None,
        )


# Kernlogik für EIN Artifact
def run_for_artifact(artifact_dir: Path):
    artifact_dir_name = artifact_dir.name
    print("\n" + "=" * 80)
    print(f"[SVC Edge-Drop] Artifact: {artifact_dir_name}")
    print("=" * 80)

    if "edgesVar" not in artifact_dir_name:
        print(
            "  -> Kein Edge-Drop-Artifact (enthält 'edgesVar' nicht). Überspringe."
        )
        return

    variant, seed = extract_edge_tag(artifact_dir_name)

    if variant is None:
        raise ValueError(
            f"Konnte edge-tag nicht extrahieren aus: {artifact_dir_name}"
        )
        return

    edge_tag = f"{variant}_{seed}"

    set_global_seeds(seed)

    # Ergebnisordner analog zum Analyse-Skript
    edge_results_root = (
        EXPERIMENTS_ROOT / "Results_HyPa_fix" / f"Results_{edge_tag}" / "SVC"
    )
    edge_results_root.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Ergebnisse landen in: {edge_results_root}")

    # Daten laden
    if not has_required_files(artifact_dir):
        print(f"  -> Artifact {artifact_dir_name} unvollständig. Überspringe.")
        return

    X_train = _read_parquet_required(artifact_dir / "X_train.parquet")
    X_val = _read_parquet_required(artifact_dir / "X_validation.parquet")
    X_test = _read_parquet_required(artifact_dir / "X_test.parquet")

    y_train = pd.read_parquet(artifact_dir / "y_train.parquet").iloc[:, 0]
    y_val = pd.read_parquet(artifact_dir / "y_validation.parquet").iloc[:, 0]
    y_test = pd.read_parquet(artifact_dir / "y_test.parquet").iloc[:, 0]

    if "txId" in X_test.columns:
        txid_test = X_test["txId"].copy()
    else:
        txid_test = pd.Series(np.arange(len(X_test)), name="txId")

    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    print("\n[Dataset shapes]")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print("\n[Class distribution Train] (0=Fraud, 1=No-Fraud)")
    print(y_train.value_counts().sort_index())

    # Passendes Full-Graph-Bundle + best_params laden
    baseline_bundle_path = find_matching_baseline_bundle(
        artifact_dir_name=artifact_dir_name,
        baseline_results_dir=BASELINE_RESULTS_DIR,
        model_tag="SVC",
    )

    best_params = load_best_params_from_baseline_run(baseline_bundle_path)

    print("\n[best_params aus Full-Graph-Bundle]")
    print(best_params)

    # TRAIN+VAL zusammen, PCA & Scaling – exakt wie in SVC.py
    Xtv_df = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    ytv = np.concatenate([y_train.values, y_val.values])

    Xtv_df, _, Xte_df, pca_meta_final = _apply_emb_pca(
        Xtv_df,
        X_val,  # wird von _apply_emb_pca intern benutzt, aber hier ignoriert
        X_test.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )

    scaler_tv = StandardScaler()
    Xtv = scaler_tv.fit_transform(Xtv_df.values).astype(np.float32)
    Xte_tv = scaler_tv.transform(Xte_df.values).astype(np.float32)

    Xva_tv_df = _transform_emb_pca_with_fitted(X_val.copy(), pca_meta_final)
    Xva_tv = scaler_tv.transform(Xva_tv_df.values).astype(np.float32)

    feature_names = Xtv_df.columns.tolist()

    # Balancing – identisch zu SVC.py
    bal = best_params.get("balancing_strategy", "class_weights")
    Xtv_bal, ytv_bal, class_weights_tv = apply_balancing_strategy(
        Xtv, ytv, {"balancing_strategy": bal}, seed
    )

    # Finaler SVC mit best_params (ohne Hyperopt)
    kernel = best_params.get("kernel", "rbf")
    C_val = float(best_params.get("SVC_C", 1.0))
    gamma_val = None
    if kernel == "rbf":
        gamma_val = float(best_params.get("gamma", 1.0))

    common_svc_kwargs = dict(
        class_weight=class_weights_tv,
        probability=True,
        random_state=seed,
        max_iter=5000,
        tol=1e-3,
        shrinking=True,
        cache_size=2000,
    )

    if kernel == "rbf":
        clf = SVC(
            C=C_val,
            kernel="rbf",
            gamma=gamma_val,
            **common_svc_kwargs,
        )
    else:
        clf = SVC(
            C=C_val,
            kernel="linear",
            **common_svc_kwargs,
        )

    t0 = time.time()
    clf.fit(Xtv_bal, ytv_bal)
    t1 = time.time()
    print(f"\n[Fit] Dauer: {t1 - t0:.2f}s")

    # Schwelle t* auf Validation bestimmen
    proba_val = clf.predict_proba(Xva_tv)
    fraud_idx = int(np.where(clf.classes_ == 0)[0][0])
    proba_val_fraud = proba_val[:, fraud_idx]

    t_star, f1_val_star = find_best_threshold(
        y_val.values, proba_val_fraud, pos_label=0
    )
    print(f"[VAL] t* = {t_star:.4f} | F1@t* = {f1_val_star:.4f}")

    # Test-Evaluation
    proba_test = clf.predict_proba(Xte_tv)
    proba_test_fraud = proba_test[:, fraud_idx]
    pred_test = np.where(proba_test_fraud >= t_star, 0, 1)

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test, labels=clf.classes_)
    acc = accuracy_score(y_test, pred_test)

    y_test_fraud = (y_test.values == 0).astype(int)
    roc_auc = roc_auc_score(y_test_fraud, proba_test_fraud)
    pr_auc = average_precision_score(y_test_fraud, proba_test_fraud)

    print("\n=== Test Results (Fraud=0, Edge-Drop, SVC) ===")
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
    proba_nofraud = (
        proba_test[:, 1 - fraud_idx]
        if proba_test.shape[1] > 1
        else 1.0 - proba_test_fraud
    )

    df_preds = pd.DataFrame(
        {
            "txId": txid_test.reset_index(drop=True),
            "y_true": y_test.reset_index(drop=True),
            "y_pred": pd.Series(pred_test).reset_index(drop=True),
            "proba_fraud": proba_test_fraud,
            "proba_nofraud": proba_nofraud,
        }
    )

    preds_path = (
        edge_results_root
        / f"elliptic_SVC_{artifact_dir_name}__testpreds.parquet"
    )
    df_preds.to_parquet(preds_path, index=False)
    print(f"Test-Predictions gespeichert unter: {preds_path}")

    # JSON-Export – Metriken wie in SVC.py
    results_summary = {
        "run_info": {
            "model": "SVC",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": seed,
            "edge_results_root": str(edge_results_root),
        },
        "best_params": _to_jsonable(best_params),
        "metrics_test": {
            "f1_fraud": float(f1),
            "precision_fraud": float(prec),
            "recall_fraud": float(rec),
            "balanced_accuracy": float(bacc),
            "log_loss": float(ll),
            "accuracy": float(acc),
            "roc_auc_fraud": float(roc_auc),
            "pr_auc_fraud": float(pr_auc),
            "t_star_val": float(t_star),
            "f1_val_at_t_star": float(f1_val_star),
        },
    }

    json_path = edge_results_root / f"elliptic_SVC_{artifact_dir_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"✔ JSON-Results gespeichert unter: {json_path}")

    del clf, Xtv, Xte_tv, Xva_tv, Xtv_bal, ytv_bal
    gc.collect()


# main – Single & Batch
def main():
    print("=" * 80)
    print("SVC Edge-Drop Evaluation – konsistent zu SVC.py")
    print("=" * 80)

    parser = argparse.ArgumentParser(
        description="SVC Elliptic – Edge-Drop Runner (ohne Hyperopt, Single & Batch)"
    )
    parser.add_argument(
        "--artifact",
        help="Name eines Edge-Drop-Artefakt-Unterordners oder voller Pfad",
    )
    parser.add_argument(
        "--folder",
        help="Ordner mit Artefakt-Unterordnern (Default: artifacts/elliptic)",
    )
    parser.add_argument(
        "--pattern",
        help="Substring-Filter für Artefakt-Namen (z.B. 'edges25_42')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Überspringt Artefakte, für die bereits ein Edge-Drop-JSON existiert",
    )
    args = parser.parse_args()

    global ARTIFACT_ROOT
    if args.folder:
        ARTIFACT_ROOT = Path(args.folder).resolve()
        print(f"[INFO] ARTIFACT_ROOT überschrieben auf: {ARTIFACT_ROOT}")

    # Einzel-Run
    if args.artifact:
        root = Path(args.folder) if args.folder else ARTIFACT_ROOT
        art_path = Path(args.artifact)
        if not art_path.is_absolute():
            art_path = root / args.artifact
        if not art_path.is_dir():
            raise FileNotFoundError(
                f"Artifact-Ordner nicht gefunden: {art_path}"
            )
        run_for_artifact(art_path)
        return

    # Batch-Modus
    root = Path(args.folder) if args.folder else ARTIFACT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Ordner nicht gefunden: {root}")

    artifact_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    candidates = []
    for art_dir in artifact_dirs:
        name = art_dir.name
        if "edgesVar" not in name:
            continue
        if args.pattern and args.pattern not in name:
            continue
        if not has_required_files(art_dir):
            continue

        if args.skip_existing:
            variant, seed = extract_edge_tag(name)

            if variant is None:
                print(f"Konnte edge-tag nicht extrahieren aus: {name}")
                continue

            edge_tag = f"{variant}_{seed}"

            res_dir = (
                EXPERIMENTS_ROOT
                / "Results_HyPa_fix"
                / f"Results_{edge_tag}"
                / "SVC"
            )
            json_out = res_dir / f"elliptic_SVC_{name}.json"
            if json_out.exists():
                continue

        candidates.append(art_dir)

    if not candidates:
        print("Keine passenden Edge-Drop-Artefakte gefunden.")
        return

    print(f"Starte Edge-Drop-Runs für {len(candidates)} Artefakte…")
    for art_dir in candidates:
        try:
            run_for_artifact(art_dir)
        except Exception as e:
            print(f"[WARN] Fehler bei Artifact {art_dir.name}: {e}")


if __name__ == "__main__":
    main()
