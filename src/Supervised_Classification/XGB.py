# -*- coding: utf-8 -*-
"""
Conventional Supervised Classification – XGBoost (Elliptic)

- Loads preprocessed artifacts (X_train / X_validation / X_test, y_*)
- Uses Hyperopt for tuning on the validation set (no test leakage)
- Early stopping is handled exclusively via Hyperopt; XGBoost uses eval_metric='logloss'
- Class imbalance handling options: 'none' | 'oversampling' | 'scale_pos_weight'
- Final refit on the training set and evaluation on the test set
- Metrics: LogLoss (minimized), Accuracy, Balanced Accuracy, F1 / Precision / Recall (Fraud = 0),
           ROC-AUC and PR-AUC (fraud-based)
- Feature importance: permutation importance and intrinsic importance (XGBoost gain)

IMPORTANT:
    Label encoding is consistent with the rest of the project:
        0 = Fraud (minority class)
        1 = No-Fraud (majority class)
"""

import os

# reproducability
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random

random.seed(42)
import numpy as np

np.random.seed(42)
import time
import json
import joblib
import pandas as pd
import argparse

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
    find_best_threshold,
    f1_at_t_scorer_factory,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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
    make_scorer,
)
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from pathlib import Path
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from xgboost import XGBClassifier


# KONFIGURATION
ARTIFACT_DIR_NAME_DEFAULT = None

# Schalter: Feature Importance rechnen oder dummy-Werte schreiben
COMPUTE_FEATURE_IMPORTANCE = False

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "XGB"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_nan_count(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.number):
        return np.isnan(arr).sum()
    return pd.isna(arr).sum()


# Class-Imbalance
def apply_balancing_strategy(X, y, params):
    """Wendet die gewählte Balancing-Strategie auf (X, y) an und liefert ggf. scale_pos_weight."""
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
                min_df, replace=True, n_samples=len(maj_df), random_state=42
            )
            df_bal = pd.concat([maj_df, min_up]).sample(
                frac=1, random_state=42
            )
            X_bal = df_bal.drop(columns=["label"]).values
            y_bal = df_bal["label"].values

        return X_bal, y_bal, None

    else:
        # Fallback
        return X, y, None


def calculate_xgb_intrinsic_importance(clf, X, y):
    """XGB Feature-Importance (Gain)."""
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    print(
        "No feature_importances_ available, using uniform importance as fallback"
    )
    return np.ones(X.shape[1]) / X.shape[1]


# Hyperopt Space
params_space = {
    "n_estimators": hp.quniform("n_estimators", 50, 500, 25),
    "max_depth": hp.quniform("max_depth", 2, 8, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample": hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "min_child_weight": hp.loguniform(
        "min_child_weight", np.log(1e-1), np.log(10.0)
    ),
    "gamma": hp.loguniform("gamma", np.log(1e-3), np.log(1.0)),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-6), np.log(1.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10.0)),
    "balancing_strategy": hp.choice(
        "balancing_strategy",
        [
            "none",
            "oversampling",
        ],
    ),
}


# Run-Funktion für 1 Artefakt
def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
) -> str:
    artifact_dir = os.path.join(artifact_root, artifact_dir_name)
    print(f"Using artifacts from: {artifact_dir}")

    # Daten laden
    X_train = _read_parquet_required(
        os.path.join(artifact_dir, "X_train.parquet")
    )
    X_val = _read_parquet_required(
        os.path.join(artifact_dir, "X_validation.parquet")
    )
    X_test = _read_parquet_required(
        os.path.join(artifact_dir, "X_test.parquet")
    )

    # Labels als Parquet (erste Spalte nehmen)
    y_train = pd.read_parquet(
        os.path.join(artifact_dir, "y_train.parquet")
    ).iloc[:, 0]
    y_val = pd.read_parquet(
        os.path.join(artifact_dir, "y_validation.parquet")
    ).iloc[:, 0]
    y_test = pd.read_parquet(
        os.path.join(artifact_dir, "y_test.parquet")
    ).iloc[:, 0]

    # txIds vom Test-Set für spätere Prediction-Speicherung sichern
    if "txId" in X_test.columns:
        txid_test = X_test["txId"].copy()
    else:
        txid_test = pd.Series(np.arange(len(X_test)), name="txId")

    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Index-Absicherung: gleiche Indizes wie Features
    if y_train.index.equals(pd.RangeIndex(len(y_train))):
        y_train.index = X_train.index
    if y_val.index.equals(pd.RangeIndex(len(y_val))):
        y_val.index = X_val.index
    if y_test.index.equals(pd.RangeIndex(len(y_test))):
        y_test.index = X_test.index

    # Numerisch & 0/1
    y_train = pd.to_numeric(y_train, errors="coerce").astype(int)
    y_val = pd.to_numeric(y_val, errors="coerce").astype(int)
    y_test = pd.to_numeric(y_test, errors="coerce").astype(int)

    # In numpy für XGB
    Xtr, Xva, Xte = X_train.values, X_val.values, X_test.values

    # HYPEROPT – OBJECTIVE
    def objective(params):
        """
        Trainiert XGBClassifier und wertet auf Validation aus.
        Minimiert LogLoss.
        """
        # Balancing anwenden
        Xtr_bal, ytr_bal, _ = apply_balancing_strategy(
            Xtr, y_train.values, params
        )

        ratio_tr = (ytr_bal == 1).sum() / max(1, (ytr_bal == 0).sum())
        sw_tr = np.where(ytr_bal == 0, ratio_tr, 1.0)

        ratio_val = (y_val.values == 1).sum() / max(
            1, (y_val.values == 0).sum()
        )
        sw_val = np.where(y_val.values == 0, ratio_val, 1.0)

        # Modell
        clf = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params["gamma"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
        )

        t0 = time.time()
        clf.fit(
            Xtr_bal,
            ytr_bal,
            eval_set=[(Xva, y_val)],
            early_stopping_rounds=50,
            verbose=False,
            sample_weight=sw_tr.astype(np.float32),
            sample_weight_eval_set=[sw_val.astype(np.float32)],
        )

        t1 = time.time()

        pred = clf.predict(Xva)
        proba = clf.predict_proba(Xva)

        # === Metriken (Fraud=0) ===
        f1 = f1_score(y_val, pred, pos_label=0, zero_division=0)
        prec = precision_score(y_val, pred, pos_label=0, zero_division=0)
        rec = recall_score(y_val, pred, pos_label=0, zero_division=0)
        bacc = balanced_accuracy_score(y_val, pred)
        ll = log_loss(y_val, proba, labels=clf.classes_)
        acc = accuracy_score(y_val, pred)

        # nach predict_proba
        fraud_idx_val = int(np.where(clf.classes_ == 0)[0][0])
        roc_auc = roc_auc_score(
            (y_val.values == 0).astype(int), proba[:, fraud_idx_val]
        )
        pr_auc = average_precision_score(
            y_val, proba[:, fraud_idx_val], pos_label=0
        )

        print(
            f"Fit {t1 - t0:.2f}s | "
            f"F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
            f"BAcc={bacc:.4f} LogLoss={ll:.4f} Acc={acc:.4f} "
            f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
            f"| params={params}"
        )

        # Minimierungsziel
        return {"loss": ll, "status": STATUS_OK}

    trials = Trials()
    rng = np.random.default_rng(42)

    print("\nStarting hyperparameter optimization (100 trials).")
    best = fmin(
        fn=objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=rng,
    )
    best_params = space_eval(params_space, best)
    print(f"\nBest params: {best_params}")

    # REFIT AUF TRAIN & EVAL AUF TEST
    print("\nRefitting best model and evaluating on test set.")

    print(
        "\nRefitting best model (Best-Epoch-Transfer) and evaluating on test set."
    )

    # A) Erstes Fit nur, um best_iteration_ zu ermitteln (wie gehabt mit Val als eval_set)
    Xtr_bal, ytr_bal, _ = apply_balancing_strategy(
        Xtr, y_train.values, best_params
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
        random_state=42,
        n_jobs=1,
        scale_pos_weight=1.0,
    )

    clf_tmp.fit(
        Xtr_bal,
        ytr_bal,
        eval_set=[(Xva, y_val)],
        early_stopping_rounds=50,
        verbose=False,
        sample_weight=sw_tr.astype(np.float32),
        sample_weight_eval_set=[sw_val.astype(np.float32)],
    )

    # Schwellenwert auf VALIDATION bestimmen (Fraud = Klasse 0)
    proba_val_full = clf_tmp.predict_proba(Xva)
    fraud_idx_val = int(np.where(clf_tmp.classes_ == 0)[0][0])  # robust
    t_star, f1_val_star = find_best_threshold(
        y_val.values, proba_val_full[:, fraud_idx_val], pos_label=0
    )
    print(
        f"Best threshold (VAL) for Fraud=0: t*={t_star:.3f} | F1@t*={f1_val_star:.4f}"
    )

    best_iter = getattr(
        clf_tmp, "best_iteration_", int(best_params["n_estimators"])
    )
    print(f"Best iteration from validation: {best_iter}")

    # B) Train+Val zusammenbauen und finales Modell ohne ES trainieren
    Xtv = np.concatenate([Xtr, Xva], axis=0)
    ytv = np.concatenate([y_train.values, y_val.values], axis=0)
    Xtv_bal, ytv_bal, _ = apply_balancing_strategy(Xtv, ytv, best_params)

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
        random_state=42,
        n_jobs=1,
        scale_pos_weight=1.0,
    )

    clf.fit(Xtv_bal, ytv_bal, sample_weight=sw_tv)

    # FEATURE IMPORTANCE: optional berechnen oder dummy
    if COMPUTE_FEATURE_IMPORTANCE:

        print("Calculating permutation importance.")

        scorer = f1_at_t_scorer_factory(t_star, pos_label=0)
        perm_importance = permutation_importance(
            clf,
            Xva,
            y_val,
            n_repeats=10,
            random_state=42,
            n_jobs=1,
            scoring=scorer,
        )
        feature_importance_perm = perm_importance.importances_mean
        calculation_method_perm = (
            "permutation_importance@F1(t*=VAL, pos=Fraud=0)"
        )

        print("Calculating intrinsic XGBoost importance.")
        feature_importance_intrinsic = calculate_xgb_intrinsic_importance(
            clf, Xtv_bal, ytv_bal
        )
        calculation_method_intrinsic = "xgb_gain_importance"

        feature_names = X_train.columns.tolist()

        df_perm = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "perm_importance": feature_importance_perm,
                }
            )
            .sort_values("perm_importance", ascending=False)
            .reset_index(drop=True)
        )
        df_perm["perm_rank"] = np.arange(1, len(df_perm) + 1)

        df_intr = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "intrinsic_importance": feature_importance_intrinsic,
                }
            )
            .sort_values("intrinsic_importance", ascending=False)
            .reset_index(drop=True)
        )
        df_intr["intrinsic_rank"] = np.arange(1, len(df_intr) + 1)

        print("\n=== Ranking – Permutation (Top 10) ===")
        print(df_perm.head(10))
        print("\n=== Ranking – Intrinsic (Top 10) ===")
        print(df_intr.head(10))

    else:

        # KEINE Feature Importance berechnen, stattdessen Nullen/Dummies schreiben
        print(
            "Skipping feature importance computation (COMPUTE_FEATURE_IMPORTANCE=False)."
        )
        feature_names = X_train.columns.tolist()
        if feature_names is None:
            # Fallback: einfache Index-Namen
            feature_names = [f"f{i}" for i in range(Xtv.shape[1])]

        n_features = len(feature_names)
        feature_importance_perm = np.zeros(n_features, dtype=float)
        calculation_method_perm = "skipped (COMPUTE_FEATURE_IMPORTANCE=False)"

        feature_importance_intrinsic = np.zeros(n_features, dtype=float)
        calculation_method_intrinsic = (
            "skipped (COMPUTE_FEATURE_IMPORTANCE=False)"
        )

        df_perm = pd.DataFrame(
            {
                "feature": feature_names,
                "perm_importance": feature_importance_perm,
            }
        )
        df_perm["perm_rank"] = np.arange(1, len(df_perm) + 1)

        df_intr = pd.DataFrame(
            {
                "feature": feature_names,
                "intrinsic_importance": feature_importance_intrinsic,
            }
        )
        df_intr["intrinsic_rank"] = np.arange(1, len(df_intr) + 1)

    # TEST-Evaluation
    proba_test = clf.predict_proba(Xte)
    fraud_idx_test = int(np.where(clf.classes_ == 0)[0][0])
    pred_test = np.where(proba_test[:, fraud_idx_test] >= t_star, 0, 1)

    f1_te = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec_te = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec_te = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc_te = balanced_accuracy_score(y_test, pred_test)
    ll_te = log_loss(y_test, proba_test, labels=clf.classes_)
    acc_te = accuracy_score(y_test, pred_test)

    roc_auc_te = roc_auc_score(
        (y_test.values == 0).astype(int), proba_test[:, fraud_idx_test]
    )
    pr_auc_te = average_precision_score(
        y_test, proba_test[:, fraud_idx_test], pos_label=0
    )

    print("\n=== FINAL TEST EVALUATION ===")
    print(
        classification_report(
            y_test, pred_test, target_names=["Fraud(0)", "No-Fraud(1)"]
        )
    )
    print(f"Balanced Accuracy: {bacc_te:.4f}")

    print("########## Test ##########")
    pred_fraud = int((pred_test == 0).sum())
    pred_nofrd = int((pred_test == 1).sum())
    lbl_fraud = int((y_test.values == 0).sum())
    lbl_nofrd = int((y_test.values == 1).sum())
    print(f"Pred Fraud = {pred_fraud}")
    print(f"Pred No fraud = {pred_nofrd}")
    print(f"Label Fraud = {lbl_fraud}")
    print(f"Label No fraud = {lbl_nofrd}")
    print(f"F1-Score = {f1_te:.4f}")
    print(f"precision = {prec_te:.4f}")
    print(f"recall = {rec_te:.4f}")
    print(f"acc_Score = {acc_te:.4f}")
    print(f"Log_Loss = {ll_te:.4f}")
    print(f"ROC-AUC = {roc_auc_te:.4f}")
    print(f"PR-AUC = {pr_auc_te:.4f}\n")

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    print("Confusion Matrix (rows=true, cols=pred):")
    print("               Predicted")
    print("               Fraud    No Fraud")
    print(f"Actual Fraud   {cm[0,0]:7d} {cm[0,1]:10d}")
    print(f"Actual NoFraud {cm[1,0]:7d} {cm[1,1]:10d}")

    # Test-Predictions speichern
    try:
        # Fraud-Wahrscheinlichkeit (Klasse 0) herausziehen
        proba_fraud = proba_test[:, fraud_idx_test]
        # Falls du auch die Gegenklasse willst:
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

        os.makedirs(results_dir, exist_ok=True)
        preds_path = os.path.join(
            results_dir, f"elliptic_XGB_{artifact_dir_name}__testpreds.parquet"
        )
        df_preds.to_parquet(preds_path, index=False)
        print(f"✔ Test-Predictions gespeichert unter: {preds_path}")
    except Exception as e:
        print(f"Konnte Test-Predictions nicht speichern: {e}")

    # Ergebnisse als JSON speichern
    out = {
        "artifact_dir": artifact_dir_name,
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
        "hyperopt": {
            "trials": len(trials.trials),
            "best_loss": float(
                min(
                    [
                        t["result"]["loss"]
                        for t in trials.trials
                        if "result" in t
                    ]
                    or [float("inf")]
                )
            ),
        },
        "feature_importance": {
            "permutation_importance": {
                "scores": _to_jsonable(feature_importance_perm),
                "calculation_method": calculation_method_perm,
            },
            "intrinsic_importance": {
                "scores": _to_jsonable(feature_importance_intrinsic),
                "calculation_method": calculation_method_intrinsic,
            },
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(
        results_dir, f"elliptic_XGB_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {json_path}")

    artifact_tag = artifact_dir_name
    out_base = Path(results_dir)

    # 1) DataFrames robust speichern
    df_perm_path = out_base / f"elliptic_XGB_{artifact_tag}__perm.parquet"
    df_intr_path = out_base / f"elliptic_XGB_{artifact_tag}__intr.parquet"
    df_perm.to_parquet(df_perm_path, index=False)
    df_intr.to_parquet(df_intr_path, index=False)

    # 2) Hyperopt Trials speichern (Pickle via joblib)
    trials_path = out_base / f"elliptic_XGB_{artifact_tag}__trials.joblib"
    joblib.dump(trials, trials_path, compress=3)

    # 3) „Reloadables“ (Modell + Scaler + Schwelle)
    bundle = {
        "model": clf,
        "threshold_t_star": float(t_star),
        "fraud_proba_index": int(fraud_idx_test),
        "feature_names": feature_names,  # für sauberes Re-Mapping
    }
    bundle_path = out_base / f"elliptic_XGB_{artifact_tag}__bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    print(
        f"Saved extras:\n- {df_perm_path}\n- {df_intr_path}\n- {trials_path}\n- {bundle_path}"
    )

    return json_path


# CLI / Batch Runner
def main():
    ap = argparse.ArgumentParser(
        description="XGBoost Elliptic – Single & Batch Runner"
    )
    ap.add_argument(
        "--artifact",
        help="Name eines Artefakt-Unterordners unter artifacts/elliptic",
    )
    ap.add_argument(
        "--folder",
        help="Ordner mit Artefakt-Unterordnern (Default: ../artifacts/elliptic)",
    )
    ap.add_argument("--pattern", help="Substring-Filter für Artefakt-Namen")
    ap.add_argument(
        "--jobs", type=int, default=1, help="Parallelität (Prozesse)"
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Überspringt Artefakte mit vorhandener JSON",
    )
    args = ap.parse_args()

    # Single run hat Vorrang
    if args.artifact or ARTIFACT_DIR_NAME_DEFAULT:
        name = args.artifact or ARTIFACT_DIR_NAME_DEFAULT
        run_for_artifact(name, ARTIFACT_ROOT, RESULTS_DIR)
        return

    # Batch
    root = args.folder or ARTIFACT_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Ordner nicht gefunden: {root}")

    candidates = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if args.pattern and args.pattern not in name:
            continue
        if not has_required_files(p):
            continue
        if args.skip_existing:
            out = os.path.join(RESULTS_DIR, f"elliptic_XGB_{name}.json")
            if os.path.exists(out):
                continue
        candidates.append(name)

    if not candidates:
        print("Keine passenden Artefakte gefunden.")
        return

    print(f"Starte Runs für {len(candidates)} Artefakte…")
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {
                ex.submit(
                    run_for_artifact, name, ARTIFACT_ROOT, RESULTS_DIR
                ): name
                for name in candidates
            }
            for f in as_completed(futs):
                name = futs[f]
                try:
                    path = f.result()
                    print(f"Fertig: {name} -> {path}")
                except Exception as e:
                    print(f"Fehler bei {name}: {e}")
    else:
        for name in candidates:
            try:
                path = run_for_artifact(name, ARTIFACT_ROOT, RESULTS_DIR)
                print(f"Fertig: {name} -> {path}")
            except Exception as e:
                print(f"Fehler bei {name}: {e}")


if __name__ == "__main__":
    main()
