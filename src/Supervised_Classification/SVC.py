# -*- coding: utf-8 -*-
"""
Elliptic node classification with SVC (refactored)

- Loads preprocessed artifacts (Parquet) from ./artifacts/elliptic/<ARTIFACT_DIR_NAME>
- Applies StandardScaler (fit on training data only, transform on validation and test data)
- No feature selection
- Oversamples the minority class (Fraud = 0) to a 1:1 class ratio
- Uses deterministic Hyperopt via np.random.default_rng
- Saves results as JSON to ./Results_JSON/elliptic_SVC_<ARTIFACT_DIR_NAME>.json
"""

import os

# reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random

random.seed(42)
import numpy as np

np.random.seed(42)
import json
import time
import joblib
import pandas as pd
import gc
import argparse

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
    _apply_emb_pca,
    find_best_threshold,
    f1_at_t_scorer_factory,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from sklearn.preprocessing import StandardScaler
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
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from pathlib import Path
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, space_eval

# KONFIGURATION
USE_KERNEL = "rbf"

ARTIFACT_DIR_NAME_DEFAULT = None

# Schalter: Feature Importance rechnen oder dummy-Werte schreiben
COMPUTE_FEATURE_IMPORTANCE = (
    False  # auf False setzen, um Importance-Berechnung zu deaktivieren
)

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "SVC"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _transform_emb_pca_with_fitted(df, pca_meta, emb_prefix="emb_"):
    if not pca_meta.get("pca_used", False):
        return df
    emb_cols = pca_meta["emb_cols"]
    mu, sd = pca_meta["mu"], pca_meta["sd"]
    pca = pca_meta["pca_model"]
    # Standardisieren wie beim Fit
    Z = ((df[emb_cols] - mu) / sd).fillna(0.0).to_numpy()
    E = pca.transform(Z)
    pca_cols = [f"{emb_prefix}pca_{i+1}" for i in range(E.shape[1])]
    out = df.drop(columns=emb_cols, errors="ignore").copy()
    for i, c in enumerate(pca_cols):
        out[c] = E[:, i].astype("float32")
    return out


def apply_balancing_strategy(X, y, params):
    strategy = params["balancing_strategy"]

    if strategy == "none":
        return (
            X.astype(np.float32, copy=False),
            np.asarray(y, dtype=np.int32),
            None,
        )

    elif strategy == "oversampling":
        # DataFrame mit float32 erzwingen
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
            min_up = resample(
                min_df, replace=True, n_samples=len(maj_df), random_state=42
            )
            df_bal = pd.concat([maj_df, min_up]).sample(
                frac=1, random_state=42
            )

            # WICHTIG: float32 erzwingen
            X_bal = df_bal.drop(columns=["label"]).to_numpy(dtype=np.float32)
            y_bal = df_bal["label"].to_numpy(dtype=np.int32)

        del df_tr, X_df, y_df, min_df, maj_df, df_bal

        return X_bal, y_bal, None

    elif strategy == "class_weights":
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


def calculate_svc_intrinsic_importance(clf, X, y):
    """
    Berechnet intrinsische Feature Importance für SVC.
    """
    if hasattr(clf, "coef_") and clf.kernel == "linear":
        # Absolute Werte der Koeffizienten
        importance = np.abs(clf.coef_[0])
        print("Using linear SVC coefficients for intrinsic importance")
    else:
        # Für RBF-Kernel: Keine sinnvolle intrinsische Importance möglich
        print("No meaningful intrinsic importance available for RBF-SVC")
        print("Consider using only permutation importance for RBF kernel")

        # Setze alle auf 0, um zu zeigen dass es nicht verfügbar ist
        importance = np.zeros(X.shape[1])

    return importance


# Kernels abhängig vom globalen Schalter
if USE_KERNEL == "linear":
    params_space = {
        "kernel": "linear",
        "SVC_C": hp.loguniform("C_linear", np.log(1e-3), np.log(10.0)),
        "balancing_strategy": "class_weights",
        "use_emb_pca": False,
        "emb_pca_n": 32,
        "scale_before_pca": True,
    }

elif USE_KERNEL == "rbf":
    params_space = {
        "kernel": "rbf",
        "SVC_C": hp.loguniform("C_rbf", np.log(1e-3), np.log(1e2)),
        "gamma": hp.loguniform("gamma_rbf", np.log(1e-4), np.log(1e1)),
        "balancing_strategy": "class_weights",
        "use_emb_pca": False,
        "emb_pca_n": 32,
        "scale_before_pca": True,
    }

else:  # USE_KERNEL == "both"
    params_space = hp.choice(
        "svc_config",
        [
            {
                "kernel": "linear",
                "SVC_C": hp.loguniform("C_linear", np.log(1e-3), np.log(10.0)),
                "balancing_strategy": "class_weights",
                "use_emb_pca": False,
                "emb_pca_n": 32,
                "scale_before_pca": True,
            },
            {
                "kernel": "rbf",
                "SVC_C": hp.loguniform("C_rbf", np.log(1e-1), np.log(5.0)),
                "gamma": hp.loguniform("gamma_rbf", np.log(1e-3), np.log(1.0)),
                "balancing_strategy": "class_weights",
                "use_emb_pca": False,
                "emb_pca_n": 32,
                "scale_before_pca": True,
            },
        ],
    )


# Single run
def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
) -> str:

    artifact_dir = os.path.join(artifact_root, artifact_dir_name)
    if not has_required_files(artifact_dir):
        raise FileNotFoundError(f"Artefakt unvollständig: {artifact_dir}")

    print(f"\nUsing artifacts from: {artifact_dir}")

    X_train = _read_parquet_required(
        os.path.join(artifact_dir, "X_train.parquet")
    )
    X_val = _read_parquet_required(
        os.path.join(artifact_dir, "X_validation.parquet")
    )
    X_test = _read_parquet_required(
        os.path.join(artifact_dir, "X_test.parquet")
    )

    # Labels (Parquet-Serien)
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

    # Spalten 0/1 sind txId und time_step -> per Name (falls vorhanden) entfernen
    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # in int konvertieren (0/1)
    y_train = pd.Series(y_train).astype(int)
    y_val = pd.Series(y_val).astype(int)
    y_test = pd.Series(y_test).astype(int)

    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print("\nClass distribution in training set (0=Fraud,1=No-Fraud):")
    print(y_train.value_counts().sort_index())

    # Optionale PCA nur auf emb_* (Preview mit Defaults, Trials setzen es in objective)
    params_defaults = {
        "use_emb_pca": False,
        "emb_pca_n": 32,
        "scale_before_pca": True,
    }
    X_train_pca, X_val_pca, X_test_pca, pca_meta_init = _apply_emb_pca(
        X_train,
        X_val,
        X_test,
        use_emb_pca=params_defaults["use_emb_pca"],
        emb_pca_n=params_defaults["emb_pca_n"],
        scale_before_pca=params_defaults["scale_before_pca"],
    )

    # HYPEROPT – OBJECTIVE
    def objective(params):
        """
        Trainiert SVC und wertet auf Validation aus.
        Minimiert LogLoss.
        -> Beschleunigte Version: probability=False, decision_function,
           kein X_test im Tuning, Oversampling -> class_weights.
        """

        # 1) PCA NUR falls angefordert (auf Train/Val), KEIN Test im Tuning
        Xtr_df = X_train
        Xva_df = X_val
        if params.get("use_emb_pca", False):
            Xtr_df, Xva_df, _, _ = _apply_emb_pca(
                X_train.copy(),
                X_val.copy(),
                X_test,  # X_test wird ignoriert
                use_emb_pca=True,
                emb_pca_n=params.get("emb_pca_n", 32),
                scale_before_pca=params.get("scale_before_pca", True),
            )

        # 2) Einmaliges Skalieren (nur Train/Val) – kein Test
        scaler_local = StandardScaler()
        Xtr_local = scaler_local.fit_transform(Xtr_df.values).astype(
            np.float32
        )
        Xva_local = scaler_local.transform(Xva_df.values).astype(np.float32)

        bal = params.get("balancing_strategy", "class_weights")
        Xtr_bal, ytr_bal, class_weights = apply_balancing_strategy(
            Xtr_local, y_train.values, {"balancing_strategy": bal}
        )

        common = dict(
            class_weight=class_weights,
            probability=False,  # <<< großer Speed-Gewinn im Tuning
            random_state=42,
            max_iter=5000,
            tol=5e-3,
            shrinking=True,  # <<< fest auf True
            cache_size=2000,  # <<< größerer Cache (MB)
        )
        if params["kernel"] == "rbf":
            clf = SVC(
                C=float(params["SVC_C"]),
                kernel="rbf",
                gamma=float(params["gamma"]),
                **common,
            )
        else:
            clf = SVC(C=float(params["SVC_C"]), kernel="linear", **common)

        t0 = time.time()
        clf.fit(Xtr_bal, ytr_bal)
        t1 = time.time()

        # 5) Scoring mit decision_function (schnell & probabilitätsfrei)
        scores = clf.decision_function(Xva_local)
        # Wir richten die Vorhersage so aus, dass Klasse 0 (Fraud) als "positiv" behandelt wird.
        if scores.ndim == 1:
            pred = (scores < 0).astype(
                int
            )  # score<0 -> Klasse 0 (Fraud), sonst 1
            # Für AUC/PR können die rohen scores verwendet werden – Vorzeichen ist egal, solange konsistent:
            scores_fraud = -scores  # größer = eher Fraud
        else:
            # bei multi-class nicht relevant hier; fallback auf predict
            pred = clf.predict(Xva_local)
            scores_fraud = None

        # Metriken
        f1 = f1_score(y_val, pred, pos_label=0, zero_division=0)
        prec = precision_score(y_val, pred, pos_label=0, zero_division=0)
        rec = recall_score(y_val, pred, pos_label=0, zero_division=0)
        bacc = balanced_accuracy_score(y_val, pred)
        acc = accuracy_score(y_val, pred)

        # Für LogLoss/PR-AUC/ROC-AUC ohne probability verwenden wir decision scores:
        y_val_fraud = (y_val.values == 0).astype(int)
        if scores_fraud is not None:
            roc_auc = roc_auc_score(y_val_fraud, scores_fraud)
            pr_auc = average_precision_score(y_val_fraud, scores_fraud)
            loss = 1.0 - roc_auc  # surrogate, monoton mit Qualität
            ll = np.nan  # nicht berechnet im Tuning
        else:
            roc_auc, pr_auc, ll = np.nan, np.nan, np.nan
            loss = 1.0 - bacc

        print(
            f"Fit {t1 - t0:.2f}s | "
            f"F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
            f"BAcc={bacc:.4f} Acc={acc:.4f} "
            f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
            f"| params={params}"
        )

        del clf
        gc.collect()
        return {"loss": float(loss), "status": STATUS_OK}

    trials = Trials()
    rng = np.random.default_rng(42)  # deterministisch, hat .integers

    print("\nStarting hyperparameter optimization (50 trials)...")
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

    # REFIT AUF TRAIN+VAL & EVAL AUF TEST
    print("\nRefitting best model on TRAIN+VAL and evaluating on test...")

    # TRAIN+VAL zusammenbauen (DFs)
    Xtv_df = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    ytv = np.concatenate([y_train.values, y_val.values])

    # PCA mit best_params auf emb_* (fit auf TRAIN+VAL)
    Xtv_df, _, Xte_df, pca_meta_final = _apply_emb_pca(
        Xtv_df,
        X_val,
        X_test.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )

    # Scaler fitten auf TRAIN+VAL (post-PCA)
    scaler_tv = StandardScaler()
    Xtv = scaler_tv.fit_transform(Xtv_df.values).astype(np.float32)
    Xte_tv = scaler_tv.transform(Xte_df.values).astype(np.float32)

    Xva_tv_df = _transform_emb_pca_with_fitted(X_val.copy(), pca_meta_final)
    Xva_tv = scaler_tv.transform(Xva_tv_df.values).astype(np.float32)
    feature_names = Xtv_df.columns.tolist()

    # Balancing auf TRAIN+VAL
    # Xtv_bal, ytv_bal, class_weights_tv = apply_balancing_strategy(Xtv, ytv, best_params)
    bal = best_params.get("balancing_strategy", "class_weights")
    Xtv_bal, ytv_bal, class_weights_tv = apply_balancing_strategy(
        Xtv, ytv, {"balancing_strategy": bal}
    )

    if best_params["kernel"] == "rbf":
        clf = SVC(
            C=float(best_params["SVC_C"]),
            kernel="rbf",
            gamma=float(best_params["gamma"]),
            class_weight=class_weights_tv,
            probability=True,
            random_state=42,
            max_iter=5000,
            tol=1e-3,
            shrinking=True,
            cache_size=2000,
        )
    else:  # linear
        clf = SVC(
            C=float(best_params["SVC_C"]),
            kernel="linear",
            class_weight=class_weights_tv,
            probability=True,
            random_state=42,
            max_iter=5000,
            tol=1e-3,
            shrinking=True,
            cache_size=2000,
        )

    clf.fit(Xtv_bal, ytv_bal)

    proba_val_final = clf.predict_proba(Xva_tv)
    fraud_idx = int(np.where(clf.classes_ == 0)[0][0])
    t_star, f1_val_star = find_best_threshold(
        y_val.values, proba_val_final[:, fraud_idx], pos_label=0
    )
    print(f"Best threshold (VAL): t*={t_star:.3f} | F1@t*={f1_val_star:.4f}")

    # Test-Preds mit t*
    proba_test = clf.predict_proba(Xte_tv)
    proba_fraud_te = proba_test[:, fraud_idx]
    pred_test = np.where(proba_fraud_te >= t_star, 0, 1)

    # FEATURE IMPORTANCE: optional berechnen oder dummy
    if COMPUTE_FEATURE_IMPORTANCE:

        # Permutation Importance auf VALIDATION, aber mit scaler_tv transformiert
        print("Calculating permutation importance...")
        scorer = f1_at_t_scorer_factory(t_star, pos_label=0)
        perm_importance = permutation_importance(
            clf,
            Xva_tv,
            y_val,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring=scorer,
        )

        feature_importance_perm = perm_importance.importances_mean
        calculation_method_perm = (
            "permutation_importance@F1(t*=VAL, pos=Fraud=0)"
        )

        print("Calculating intrinsic SVC importance...")
        # Intrinsic importance (nur linear sinnvoll) – nimm Xtv_bal/ytv_bal als Referenz
        feature_importance_intrinsic = calculate_svc_intrinsic_importance(
            clf, Xtv_bal, ytv_bal
        )
        calculation_method_intrinsic = (
            f"svc_{best_params['kernel']}_coefficient_abs"
        )

        # Gemeinsamer Feature-Index
        feature_names = Xtv_df.columns.tolist()

        # 1) Permutation-Importance DataFrame
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

        # 2) Intrinsische Importance (|coef|) – bereits berechnet oben als feature_importance_intrinsic
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

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test, labels=clf.classes_)
    acc = accuracy_score(y_test, pred_test)

    y_test_fraud = (y_test.values == 0).astype(int)
    roc_auc = roc_auc_score(y_test_fraud, proba_test[:, fraud_idx])
    pr_auc = average_precision_score(
        y_test, proba_test[:, fraud_idx], pos_label=0
    )

    print("\n=== Test Results (Fraud=0) ===")
    print(
        f"F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | "
        f"BalancedAcc={bacc:.4f} | LogLoss={ll:.4f} | Acc={acc:.4f} | "
        f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
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
    try:
        # Fraud-Wahrscheinlichkeit (Klasse 0) herausziehen
        proba_fraud = proba_test[:, fraud_idx]
        # Falls du auch die Gegenklasse willst:
        proba_nofraud = (
            proba_test[:, 1 - fraud_idx]
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
            results_dir, f"elliptic_SVC_{artifact_dir_name}__testpreds.parquet"
        )
        df_preds.to_parquet(preds_path, index=False)
        print(f"Test-Predictions gespeichert unter: {preds_path}")
    except Exception as e:
        print(f"Konnte Test-Predictions nicht speichern: {e}")

    # JSON-EXPORT
    results_summary = {
        "run_info": {
            "model": "SVC",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": 42,
            "scaling": "StandardScaler (FINAL: fit on TRAIN+VAL)",
            "balancing_strategy": best_params["balancing_strategy"],
        },
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "X_validation": list(X_val.shape),
        },
        "validation": {
            "best_threshold_t": float(t_star),
            "f1_at_best_t_val": float(f1_val_star),
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
                "sorted_features": df_perm["feature"].tolist(),
                "sorted_importances": df_perm["perm_importance"].tolist(),
            },
            "intrinsic_importance": {
                "scores": _to_jsonable(feature_importance_intrinsic),
                "calculation_method": calculation_method_intrinsic,
                "sorted_features": df_intr["feature"].tolist(),
                "sorted_importances": df_intr["intrinsic_importance"].tolist(),
                "coefficients_available": hasattr(clf, "coef_")
                and clf.kernel == "linear",
                "kernel_type": best_params["kernel"],
            },
        },
    }

    json_path = os.path.join(
        RESULTS_DIR, f"elliptic_SVC_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to JSON: {json_path}")

    artifact_tag = artifact_dir_name
    out_base = Path(results_dir)

    # 1) DataFrames robust speichern
    df_perm_path = out_base / f"elliptic_SVC_{artifact_tag}__perm.parquet"
    df_intr_path = out_base / f"elliptic_SVC_{artifact_tag}__intr.parquet"
    df_perm.to_parquet(df_perm_path, index=False)
    df_intr.to_parquet(df_intr_path, index=False)

    # 2) Hyperopt Trials speichern (Pickle via joblib)
    trials_path = out_base / f"elliptic_SVC_{artifact_tag}__trials.joblib"
    joblib.dump(trials, trials_path, compress=3)

    # 3) „Reloadables“ (Modell + Scaler + Schwelle)
    bundle = {
        "model": clf,
        "threshold_t_star": float(t_star),
        "fraud_proba_index": int(
            fraud_idx
        ),  # <- robust ggü. Klassenreihenfolge
        "feature_names": feature_names,  # nach evtl. PCA
        "scaler": scaler_tv,  # <- für Re-Use
        "pca_meta": {  # <- nur wenn verwendet
            "pca_used": bool(pca_meta_final.get("pca_used", False)),
            "k": int(pca_meta_final.get("k", 0)),
            "explained": float(pca_meta_final.get("explained") or 0.0),
            "emb_cols": pca_meta_final.get("emb_cols", []),
            "mu": pca_meta_final.get("mu"),
            "sd": pca_meta_final.get("sd"),
            "pca_model": pca_meta_final.get("pca_model"),
            "scale_before_pca": bool(
                pca_meta_final.get("scale_before_pca", True)
            ),
        },
        "best_params": best_params,
    }
    bundle_path = out_base / f"elliptic_SVC_{artifact_tag}__bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    print(
        f"Saved extras:\n- {df_perm_path}\n- {df_intr_path}\n- {trials_path}\n- {bundle_path}"
    )

    return json_path


# Batch main
def main():
    ap = argparse.ArgumentParser(
        description="SVC Elliptic – Single & Batch Runner"
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

    if args.artifact:
        run_for_artifact(args.artifact, ARTIFACT_ROOT, RESULTS_DIR)
        return

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
            out = os.path.join(RESULTS_DIR, f"elliptic_SVC_{name}.json")
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
