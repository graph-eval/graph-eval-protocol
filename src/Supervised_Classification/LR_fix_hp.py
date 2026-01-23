# -*- coding: utf-8 -*-
"""
LR_fix_hp.py
Logistic Regression – variant, consistent with LR.py

- Loads edge-drop artifacts from artifacts/elliptic/<ARTIFACT_DIR_NAME>
- No hyperparameter optimization
- Uses best_params from the corresponding full-graph LR bundle
- Replicates the final pipeline from LR.py:
    * Optional embedding PCA via _apply_emb_pca
    * StandardScaler
    * Balancing (none / oversampling / class_weights)
    * LogisticRegression(penalty, C, solver, class_weight)
    * Threshold t* determined on the validation set using a train-only pipeline
- Fits the final model on TRAIN + VAL of the edge-drop artifact
- Evaluates on TEST
- Saves outputs to:
    C:/Experiments/Results_JSON_<drop>_<seed>/LR/
        elliptic_LR_<ARTIFACT_DIR_NAME>.json
        elliptic_LR_<ARTIFACT_DIR_NAME>__testpreds.parquet
"""

import os

# Reproduzierbarkeit wie in LR.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import numpy as np

import re
import time
import json
import joblib
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


# Pfade / Konstanten
THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
BASELINE_RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "LR"

BASE_PREFIX = "base93"
EDGES_PREFIX = "_edgesVar"
RANDOM_PREFIX = "_random"


# Wrapper & Balancing – 1:1 aus LR.py übernommen
class PreFittedLRWrapper:
    """
    Wendet (optional) Embedding-PCA, anschließend globales Scaling,
    dann die bereits gefittete LogisticRegression an.
    Erwartet Rohfeatures (Original-DF/ndarray).
    """

    def __init__(
        self, emb_cols, mu, sd, pca_model, scale_before_pca, scaler, clf
    ):
        self.emb_cols = emb_cols
        self.mu = mu
        self.sd = sd
        self.pca = pca_model
        self.scale_before_pca = scale_before_pca
        self.scaler = scaler
        self.clf = clf
        self.classes_ = getattr(clf, "classes_", None)
        self.final_feature_names_ = None  # nur für FI relevant

    def fit(self, X, y=None):
        # Wrapper ist bereits „pre-fitted“
        return self

    def _apply_emb_pca_to_array(self, X_df_like):
        # akzeptiert np.array oder pd.DataFrame
        if isinstance(X_df_like, np.ndarray):
            X_df = pd.DataFrame(X_df_like)
        else:
            X_df = X_df_like.copy()

        if (
            self.pca is None
            or self.emb_cols is None
            or len(self.emb_cols) == 0
        ):
            out = X_df.values
            if self.final_feature_names_ is None:
                self.final_feature_names_ = (
                    list(X_df.columns)
                    if hasattr(X_df, "columns")
                    else [str(i) for i in range(X_df.shape[1])]
                )
            return out

        # Embedding-Block z-normieren (falls gewünscht)
        if (
            self.scale_before_pca
            and self.mu is not None
            and self.sd is not None
        ):
            Z = (
                ((X_df[self.emb_cols] - self.mu) / self.sd)
                .fillna(0.0)
                .to_numpy()
            )
        else:
            Z = X_df[self.emb_cols].fillna(0.0).to_numpy()

        E = self.pca.transform(Z)
        pca_cols = [f"emb_pca_{i+1}" for i in range(E.shape[1])]
        out_df = X_df.drop(columns=self.emb_cols, errors="ignore").copy()
        for i, c in enumerate(pca_cols):
            out_df[c] = E[:, i].astype("float32")

        if self.final_feature_names_ is None:
            self.final_feature_names_ = list(out_df.columns)
        return out_df.values

    def _transform(self, X):
        Xp = self._apply_emb_pca_to_array(X)
        return self.scaler.transform(Xp)

    def predict(self, X):
        return self.clf.predict(self._transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._transform(X))


def apply_balancing_strategy(X, y, params, drop_seed):
    """
    Balancing-Strategie *identisch* zu LR.py.

    params["balancing_strategy"] ∈ {"none", "oversampling", "class_weights"}
    """
    strategy = params["balancing_strategy"]

    if strategy == "none":
        return X, y, None

    elif strategy == "oversampling":
        Xtr_df = pd.DataFrame(X)
        ytr_df = pd.Series(y, name="label")
        df_tr = pd.concat([Xtr_df, ytr_df], axis=1)

        min_df = df_tr[df_tr["label"] == 0]
        maj_df = df_tr[df_tr["label"] == 1]

        if len(min_df) == 0 or len(maj_df) == 0:
            Xtr_bal = X
            ytr_bal = y
        else:
            from sklearn.utils import resample

            min_up = resample(
                min_df,
                replace=True,
                n_samples=len(maj_df),
                random_state=drop_seed,
            )
            df_bal = pd.concat([maj_df, min_up]).sample(
                frac=1, random_state=drop_seed
            )

            Xtr_bal = df_bal.drop(columns=["label"]).values
            ytr_bal = df_bal["label"].values

        return Xtr_bal, ytr_bal, None

    elif strategy == "class_weights":
        # SVC/LR-interne Klassen-Gewichtung
        return X, y, "balanced"

    else:
        # Fallback
        return X, y, None


# Kernlogik für EIN Edge-Drop-Artifact
def run_for_artifact_edge(artifact_dir_name: str) -> str:
    artifact_dir = ARTIFACT_ROOT / artifact_dir_name
    print("\n" + "=" * 80)
    print(f"[LR Edge-Drop] Artifact: {artifact_dir_name}")
    print("=" * 80)

    if "_edgesVar" not in artifact_dir_name:
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

    # Ergebnisordner analog SVC/NB-Edge:
    # C:/Experiments/Results_JSON_<drop>_<seed>/LR
    edge_results_root = (
        EXPERIMENTS_ROOT / "Results_HyPa_fix" / f"Results_{edge_tag}" / "LR"
    )
    edge_results_root.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Ergebnisse werden gespeichert unter: {edge_results_root}")

    # Daten laden (wie in LR.py)
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

    y_train = pd.Series(y_train).astype(int)
    y_val = pd.Series(y_val).astype(int)
    y_test = pd.Series(y_test).astype(int)

    print("\n[Dataset shapes (edge-drop)]")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print("\n[Class distribution Train] (0=Fraud,1=No-Fraud)")
    print(y_train.value_counts().sort_index())

    # Full-Graph-Bundle + best_params laden
    baseline_bundle_path = find_matching_baseline_bundle(
        artifact_dir_name=artifact_dir_name,
        baseline_results_dir=BASELINE_RESULTS_DIR,
        model_tag="LR",
    )

    best_params = load_best_params_from_baseline_run(baseline_bundle_path)

    print("\n[best_params aus Full-Graph-Bundle]")
    print(best_params)

    # TRAIN-only-Pipeline für t*: PCA + Scaling + Balancing
    Xtr_df, Xva_df, _dummy, pca_meta_tr = _apply_emb_pca(
        X_train.copy(),
        X_val.copy(),
        X_val.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )

    scaler_tr = StandardScaler()
    Xtr = scaler_tr.fit_transform(Xtr_df.values)
    Xva = scaler_tr.transform(Xva_df.values)

    Xtr_bal, ytr_bal, class_weights_tr = apply_balancing_strategy(
        Xtr, y_train.values, best_params, seed
    )

    clf_tmp = LogisticRegression(
        penalty=best_params["penalty"],
        C=float(best_params["C"]),
        solver=best_params["solver"],
        max_iter=3000,
        class_weight=class_weights_tr,
        random_state=seed,
    ).fit(Xtr_bal, ytr_bal)

    wrapped_tmp = PreFittedLRWrapper(
        emb_cols=pca_meta_tr.get("emb_cols"),
        mu=pca_meta_tr.get("mu"),
        sd=pca_meta_tr.get("sd"),
        pca_model=pca_meta_tr.get("pca_model"),
        scale_before_pca=pca_meta_tr.get("scale_before_pca"),
        scaler=scaler_tr,
        clf=clf_tmp,
    )

    fraud_col_tmp = int(np.where(wrapped_tmp.classes_ == 0)[0][0])
    proba_val_tmp = wrapped_tmp.predict_proba(X_val)[:, fraud_col_tmp]
    t_star, f1_val_star = find_best_threshold(
        y_val.values,
        proba_val_tmp,
        pos_label=0,
    )
    print(f"[VAL] t* = {t_star:.4f} | F1@t* = {f1_val_star:.4f}")

    # TRAIN+VAL zusammen, PCA+Scaler neu fitten, finales Modell
    Xtv_df = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    ytv = np.concatenate([y_train.values, y_val.values])

    # PCA auf TRAIN+VAL fitten, TEST transformieren
    Xtv_df, _dummy, Xte_df, pca_meta_final = _apply_emb_pca(
        Xtv_df,
        X_val.copy(),
        X_test.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )

    scaler_tv = StandardScaler()
    Xtv = scaler_tv.fit_transform(Xtv_df.values)
    Xte = scaler_tv.transform(Xte_df.values)

    # Val für Debug/evtl. spätere Auswertungen ebenfalls durch finalen PCA+Scaler
    Xva_tv_df, _dummy, _dummy2, _ = _apply_emb_pca(
        X_val.copy(),
        X_val.copy(),
        X_val.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )
    Xva_tv = scaler_tv.transform(Xva_tv_df.values)

    Xtv_bal, ytv_bal, class_weights = apply_balancing_strategy(
        Xtv, ytv, best_params, seed
    )

    clf = LogisticRegression(
        penalty=best_params["penalty"],
        C=float(best_params["C"]),
        solver=best_params["solver"],
        max_iter=3000,
        class_weight=class_weights,
        random_state=seed,
    )
    t0 = time.time()
    clf.fit(Xtv_bal, ytv_bal)
    t1 = time.time()
    print(f"[Fit TRAIN+VAL] Dauer: {t1 - t0:.2f}s")

    wrapped = PreFittedLRWrapper(
        emb_cols=pca_meta_final.get("emb_cols"),
        mu=pca_meta_final.get("mu"),
        sd=pca_meta_final.get("sd"),
        pca_model=pca_meta_final.get("pca_model"),
        scale_before_pca=pca_meta_final.get("scale_before_pca"),
        scaler=scaler_tv,
        clf=clf,
    )

    # Test-Evaluation mit t*
    fraud_col_final = int(np.where(wrapped.classes_ == 0)[0][0])
    proba_test_full = wrapped.predict_proba(X_test)
    proba_test = proba_test_full[:, fraud_col_final]
    pred_test = np.where(proba_test >= t_star, 0, 1)

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test_full, labels=wrapped.classes_)
    acc = accuracy_score(y_test, pred_test)

    y_test_fraud = (y_test.values == 0).astype(int)
    roc_auc = roc_auc_score(y_test_fraud, proba_test)
    pr_auc = average_precision_score(y_test_fraud, proba_test)

    print("\n=== Test Results (Fraud=0, Edge-Drop, LR) ===")
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
    proba_fraud = proba_test
    proba_nofraud = (
        proba_test_full[:, 1 - fraud_col_final]
        if proba_test_full.shape[1] > 1
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
        / f"elliptic_LR_{artifact_dir_name}__testpreds.parquet"
    )
    df_preds.to_parquet(preds_path, index=False)
    print(f"Test-Predictions gespeichert unter: {preds_path}")

    # JSON-Export – Struktur wie LR.py (ohne Hyperopt & Feature Importance)
    results_summary = {
        "run_info": {
            "model": "LogisticRegression",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": seed,
            "balancing_strategy": best_params.get("balancing_strategy"),
            "scaling": "StandardScaler (FINAL: fit on TRAIN+VAL edge-drop)",
            "dim_reduction": (
                "emb_PCA" if best_params.get("use_emb_pca", False) else "none"
            ),
            "final_fit": "train+val (edge-drop)",
            "decision_threshold_fraud": float(t_star),
            "val_f1_at_threshold": float(f1_val_star),
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

    json_path = edge_results_root / f"elliptic_LR_{artifact_dir_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"JSON-Results gespeichert unter: {json_path}")

    return str(json_path)


# CLI – Single & Batch
def main():
    print("=" * 80)
    print("LogisticRegression Edge-Drop Evaluation – konsistent zu LR.py")
    print("=" * 80)

    ap = argparse.ArgumentParser(
        description="LogisticRegression Elliptic – Edge-Drop Runner (ohne Hyperopt, Single & Batch)"
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
        help="Substring-Filter für Artefakt-Namen (z.B. 'edges25_42' oder 'Var_pagerankTop10_42')",
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

        if "edges" not in name:
            continue

        if args.pattern and args.pattern not in name:
            continue

        if not has_required_files(p):
            continue

        variant, seed = extract_edge_tag(name)
        if variant is None:
            print(f"Konnte edge-tag nicht extrahieren aus: {name}")
            continue
        edge_tag = f"{variant}_{seed}"

        if args.skip_existing:
            res_dir = (
                EXPERIMENTS_ROOT
                / "Results_HyPa_fix"
                / f"Results_{edge_tag}"
                / "LR"
            )
            json_out = res_dir / f"elliptic_LR_{name}.json"
            if json_out.exists():
                continue

        candidates.append(name)

    if not candidates:
        print("Keine passenden Edge-Drop-Artefakte gefunden.")
        return

    print(f"Starte Edge-Drop-Runs für {len(candidates)} Artefakte…")
    for name in candidates:
        try:
            path = run_for_artifact_edge(name)
            print(f"Fertig: {name} -> {path}")
        except Exception as e:
            print(f"[WARN] Fehler bei Artifact {name}: {e}")


if __name__ == "__main__":
    main()
