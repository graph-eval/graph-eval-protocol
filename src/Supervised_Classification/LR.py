# -*- coding: utf-8 -*-
"""
Elliptic node classification with Logistic Regression (refactored)

- Loads preprocessed artifacts (Parquet) from ../artifacts/elliptic/<ARTIFACT_DIR_NAME>
- Applies StandardScaler (fit on training data only, transform on validation and test data)
- No feature selection
- Oversamples the minority class (Fraud = 0) to a 1:1 class ratio
- Uses deterministic Hyperopt via np.random.default_rng
- Saves results as JSON to ../Results_JSON/elliptic_LR_<ARTIFACT_DIR_NAME>.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
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
from sklearn.inspection import permutation_importance
from pathlib import Path
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, space_eval


# Configuration
ARTIFACT_DIR_NAME_DEFAULT = None

# Switch: compute feature importance or write dummy values
COMPUTE_FEATURE_IMPORTANCE = False

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "LR"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PreFittedLRWrapper:
    """
    Applies (optional) embedding PCA, followed by global scaling, and then the
    pre-fitted LogisticRegression. Expects raw features (original DataFrame/ndarray).
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
        # Store final feature order after PCA replacement
        self.final_feature_names_ = None

    def fit(self, X, y=None):
        return self

    def _apply_emb_pca_to_array(self, X_df_like):
        # Accept np.array or pd.DataFrame
        if isinstance(X_df_like, np.ndarray):
            X_df = pd.DataFrame(X_df_like)
        else:
            X_df = X_df_like.copy()

        if (
            self.pca is None
            or self.emb_cols is None
            or len(self.emb_cols) == 0
        ):
            # No embedding PCA: return input as-is
            out = X_df.values
            if self.final_feature_names_ is None:
                self.final_feature_names_ = (
                    list(X_df.columns)
                    if hasattr(X_df, "columns")
                    else [str(i) for i in range(X_df.shape[1])]
                )
            return out

        # Z-normalize embedding block if requested
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


def apply_balancing_strategy(X, y, params):
    """Applies the optimized balancing strategy"""
    strategy = params["balancing_strategy"]

    if strategy == "none":
        # No balancing – use original data
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
            min_up = resample(
                min_df, replace=True, n_samples=len(maj_df), random_state=42
            )
            df_bal = pd.concat([maj_df, min_up]).sample(
                frac=1, random_state=42
            )

            Xtr_bal = df_bal.drop(columns=["label"]).values
            ytr_bal = df_bal["label"].values

        return Xtr_bal, ytr_bal, None

    elif strategy == "class_weights":
        # Automatic class weights
        return X, y, "balanced"

    else:
        # Fallback: no balancing
        return X, y, None


# Hyperparameter space
params_space = {
    "penalty": hp.choice("penalty", ["l2", None]),
    "C": hp.loguniform("C", np.log(0.001), np.log(10)),
    "solver": hp.choice("solver", ["lbfgs", "saga"]),
    "balancing_strategy": hp.choice(
        "balancing_strategy", ["none", "oversampling", "class_weights"]
    ),
    # PCA applied only to emb_* features
    "use_emb_pca": hp.choice("use_emb_pca", [False, True]),
    "emb_pca_n": hp.choice(
        "emb_pca_n", [16, 32, 48, 64, 0.95]
    ),  # compact search space
    "scale_before_pca": hp.choice("scale_before_pca", [True]),
}


def run_for_artifact(
    artifact_dir_name: str,
    artifact_root=ARTIFACT_ROOT,
    results_dir=RESULTS_DIR,
) -> str:
    artifact_dir = os.path.join(artifact_root, artifact_dir_name)
    print(f"Using artifacts from: {artifact_dir}")

    X_train = _read_parquet_required(
        os.path.join(artifact_dir, "X_train.parquet")
    )
    X_val = _read_parquet_required(
        os.path.join(artifact_dir, "X_validation.parquet")
    )
    X_test = _read_parquet_required(
        os.path.join(artifact_dir, "X_test.parquet")
    )

    y_train = pd.read_parquet(
        os.path.join(artifact_dir, "y_train.parquet")
    ).iloc[:, 0]
    y_val = pd.read_parquet(
        os.path.join(artifact_dir, "y_validation.parquet")
    ).iloc[:, 0]
    y_test = pd.read_parquet(
        os.path.join(artifact_dir, "y_test.parquet")
    ).iloc[:, 0]

    # Preserve txIds from the test set for later prediction storage
    if "txId" in X_test.columns:
        txid_test = X_test["txId"].copy()
    else:
        txid_test = pd.Series(np.arange(len(X_test)), name="txId")

    # Columns txId and time_step are removed by name if present
    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    y_train = pd.Series(y_train).astype(int)
    y_val = pd.Series(y_val).astype(int)
    y_test = pd.Series(y_test).astype(int)

    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print("\nClass distribution in training set (0=Fraud,1=No-Fraud):")
    print(y_train.value_counts().sort_index())

    # Optional PCA on emb_* features before global scaling
    # Default values in case Hyperopt has not set them yet (initial runs)
    params_defaults = {
        "use_emb_pca": False,
        "emb_pca_n": 32,
        "scale_before_pca": True,
    }

    p_use = params_defaults["use_emb_pca"]
    p_k = params_defaults["emb_pca_n"]
    p_scale = params_defaults["scale_before_pca"]

    X_train_pca, X_val_pca, X_test_pca, pca_meta_init = _apply_emb_pca(
        X_train,
        X_val,
        X_test,
        use_emb_pca=p_use,
        emb_pca_n=p_k,
        scale_before_pca=p_scale,
    )

    # HYPEROPT – OBJECTIVE
    def objective(params):
        """
        Trains LogisticRegression and evaluates on the validation set.
        Minimizes log loss.
        """

        # 1) PCA applied only to emb_* features: fit on train, transform on val/test
        Xtr_df = X_train.copy()
        Xva_df = X_val.copy()
        Xte_df = X_test.copy()

        Xtr_df, Xva_df, Xte_df, pca_meta = _apply_emb_pca(
            Xtr_df,
            Xva_df,
            Xte_df,
            use_emb_pca=params.get("use_emb_pca", False),
            emb_pca_n=params.get("emb_pca_n", 32),
            scale_before_pca=params.get("scale_before_pca", True),
        )

        # 2) Global StandardScaler (fit on training data)
        scaler_local = StandardScaler()
        Xtr_local = scaler_local.fit_transform(Xtr_df.values)
        Xva_local = scaler_local.transform(Xva_df.values)
        Xte_local = scaler_local.transform(Xte_df.values)

        # 3) Balancing
        Xtr_bal, ytr_bal, class_weights = apply_balancing_strategy(
            Xtr_local, y_train.values, params
        )

        # 4) Model
        clf = LogisticRegression(
            penalty=params["penalty"],
            C=float(params["C"]),
            solver=params["solver"],
            max_iter=3000,
            class_weight=class_weights,
            random_state=42,
        )
        t0 = time.time()
        clf.fit(Xtr_bal, ytr_bal)
        t1 = time.time()

        pred = clf.predict(Xva_local)
        proba = clf.predict_proba(Xva_local)

        f1 = f1_score(y_val, pred, pos_label=0, zero_division=0)
        prec = precision_score(y_val, pred, pos_label=0, zero_division=0)
        rec = recall_score(y_val, pred, pos_label=0, zero_division=0)
        bacc = balanced_accuracy_score(y_val, pred)
        ll = log_loss(y_val, proba, labels=clf.classes_)
        acc = accuracy_score(y_val, pred)

        proba = clf.predict_proba(Xva_local)
        fraud_col_local = int(np.where(clf.classes_ == 0)[0][0])
        y_val_fraud = (y_val.values == 0).astype(int)
        roc_auc = roc_auc_score(y_val_fraud, proba[:, fraud_col_local])
        pr_auc = average_precision_score(
            y_val, proba[:, fraud_col_local], pos_label=0
        )

        print(
            f"Fit {t1 - t0:.2f}s | "
            f"F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
            f"BAcc={bacc:.4f} LogLoss={ll:.4f} Acc={acc:.4f} "
            f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} "
            f"| params={params}"
        )

        return {"loss": ll, "status": STATUS_OK}

    trials = Trials()
    rng = np.random.default_rng(42)

    print("\nStarting hyperparameter optimization (100 trials)...")
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

    # Refit on TRAIN+VAL and evaluate on TEST
    print("\nRefitting best model on TRAIN+VAL and evaluating on test set...")

    # TRAIN-only pipeline for t_star (no leakage)
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
        Xtr, y_train.values, best_params
    )

    clf_tmp = LogisticRegression(
        penalty=best_params["penalty"],
        C=float(best_params["C"]),
        solver=best_params["solver"],
        max_iter=3000,
        class_weight=class_weights_tr,
        random_state=42,
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
        y_val.values, proba_val_tmp, pos_label=0
    )

    # Combine TRAIN+VAL (original feature space)
    Xtv_df = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    ytv = np.concatenate([y_train.values, y_val.values])

    # PCA applied only to emb_* features (fit on TRAIN+VAL; transform on TEST)
    Xtv_df, _dummy, Xte_df, pca_meta_final = _apply_emb_pca(
        Xtv_df,
        X_val.copy(),
        X_test.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )

    # Global scaler (fit on TRAIN+VAL; transform on TEST & VAL for PI)
    scaler_tv = StandardScaler()
    Xtv = scaler_tv.fit_transform(Xtv_df.values)
    Xte = scaler_tv.transform(Xte_df.values)

    Xva_tv_df, _dummy, _dummy2, _ = _apply_emb_pca(
        X_val.copy(),
        X_val.copy(),
        X_val.copy(),
        use_emb_pca=best_params.get("use_emb_pca", False),
        emb_pca_n=best_params.get("emb_pca_n", 32),
        scale_before_pca=best_params.get("scale_before_pca", True),
    )
    Xva_tv = scaler_tv.transform(Xva_tv_df.values)

    # Apply balancing on TRAIN+VAL
    Xtv_bal, ytv_bal, class_weights = apply_balancing_strategy(
        Xtv, ytv, best_params
    )

    # Fit final model on TRAIN+VAL
    clf = LogisticRegression(
        penalty=best_params["penalty"],
        C=float(best_params["C"]),
        solver=best_params["solver"],
        max_iter=3000,
        class_weight=class_weights,
        random_state=42,
    )
    clf.fit(Xtv_bal, ytv_bal)

    # Pipeline-Wrapper (Emb-PCA + Scaler + LR)
    wrapped = PreFittedLRWrapper(
        emb_cols=pca_meta_final.get("emb_cols"),
        mu=pca_meta_final.get("mu"),
        sd=pca_meta_final.get("sd"),
        pca_model=pca_meta_final.get("pca_model"),
        scale_before_pca=pca_meta_final.get("scale_before_pca"),
        scaler=scaler_tv,
        clf=clf,
    )

    # Test evaluation (using the same t_star)
    fraud_col_final = int(np.where(wrapped.classes_ == 0)[0][0])
    proba_test_full = wrapped.predict_proba(X_test)
    proba_test = proba_test_full[:, fraud_col_final]
    # proba_test corresponds to the probability of class 0 (Fraud)
    pred_test = np.where(proba_test >= t_star, 0, 1)

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test_full, labels=wrapped.classes_)
    acc = accuracy_score(y_test, pred_test)

    y_test_fraud = (y_test.values == 0).astype(int)
    roc_auc = roc_auc_score(y_test_fraud, proba_test)
    pr_auc = average_precision_score(y_test, proba_test, pos_label=0)

    # FEATURE IMPORTANCE: optionally compute or write dummy values
    if COMPUTE_FEATURE_IMPORTANCE:

        # Permutation importance on validation set using F1@t* (t* from TRAIN-only stage)
        scorer = f1_at_t_scorer_factory(t_star, pos_label=0)
        perm = permutation_importance(
            estimator=wrapped,
            X=X_val,
            y=y_val,
            scoring=scorer,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        feature_importance_perm = perm.importances_mean
        calculation_method_perm = (
            "permutation_importance@F1(t*=VAL, pos=Fraud=0)"
        )

        # Intrinsic importance (|coef|) after scaling / PCA replacement
        print("Calculating intrinsic importance...")
        feature_importance_intrinsic = (
            np.abs(clf.coef_[0])
            if hasattr(clf, "coef_")
            else np.zeros(Xtv.shape[1])
        )
        calculation_method_intrinsic = (
            "lr_coefficient_abs (after emb-PCA replacement + global scaling)"
        )

        # Feature names from wrapper (set after a predict_proba call)
        feature_names = wrapped.final_feature_names_

        # Rankings as DataFrames
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
        # Do NOT compute feature importance; write zeros/dummy values instead
        print(
            "Skipping feature importance computation (COMPUTE_FEATURE_IMPORTANCE=False)."
        )
        feature_names = wrapped.final_feature_names_
        if feature_names is None:
            # Fallback: simple index-based feature names
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

    # Save test predictions
    try:
        # proba_test already represents the Fraud probability (class 0)
        proba_fraud = proba_test
        # No-Fraud probability from the other column
        if proba_test_full.shape[1] > 1:
            proba_nofraud = proba_test_full[:, 1 - fraud_col_final]
        else:
            proba_nofraud = 1.0 - proba_fraud

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
            results_dir, f"elliptic_LR_{artifact_dir_name}__testpreds.parquet"
        )
        df_preds.to_parquet(preds_path, index=False)
        print(f"Test predictions saved to: {preds_path}")
    except Exception as e:
        print(f"Failed to save test predictions: {e}")

    # JSON export
    results_summary = {
        "run_info": {
            "model": "LogisticRegression",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": 42,
            "balancing_strategy": best_params["balancing_strategy"],
            "scaling": "StandardScaler (FINAL: fit on TRAIN+VAL)",
            "final_fit": "train+val",
            "decision_threshold_fraud": float(t_star),
            "val_f1_at_threshold": float(f1_val_star),
        },
        "data_shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "X_validation": list(X_val.shape),
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
                "coefficients": (
                    _to_jsonable(clf.coef_[0].tolist())
                    if hasattr(clf, "coef_")
                    else None
                ),
            },
        },
    }

    json_path = os.path.join(
        results_dir, f"elliptic_LR_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to JSON: {json_path}")

    artifact_tag = artifact_dir_name
    out_base = Path(results_dir)

    # 1) Persist DataFrames robustly
    df_perm_path = out_base / f"elliptic_LR_{artifact_tag}__perm.parquet"
    df_intr_path = out_base / f"elliptic_LR_{artifact_tag}__intr.parquet"
    df_perm.to_parquet(df_perm_path, index=False)
    df_intr.to_parquet(df_intr_path, index=False)

    # 2) Persist Hyperopt trials (pickle via joblib)
    trials_path = out_base / f"elliptic_LR_{artifact_tag}__trials.joblib"
    joblib.dump(trials, trials_path, compress=3)

    # 3) Reloadables (model + scaler + threshold)
    fraud_col_final = int(np.where(wrapped.classes_ == 0)[0][0])
    bundle = {
        "model": clf,
        "threshold_t_star": float(t_star),
        "fraud_proba_index": int(fraud_col_final),
        "feature_names": feature_names,
        "scaler": scaler_tv,
        "pca_meta": {
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

    bundle_path = out_base / f"elliptic_LR_{artifact_tag}__bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    print(
        f"Saved extras:\n- {df_perm_path}\n- {df_intr_path}\n- {trials_path}\n- {bundle_path}"
    )

    return json_path


def main():
    ap = argparse.ArgumentParser(
        description="LogisticRegression Elliptic – Single & Batch Runner"
    )
    ap.add_argument(
        "--artifact",
        help="Name of an artifact subdirectory under artifacts/elliptic",
    )
    ap.add_argument(
        "--folder",
        help="Folder containing artifact subdirectories (Default: ../artifacts/elliptic)",
    )
    ap.add_argument("--pattern", help="Substring filter for artifact names")
    ap.add_argument(
        "--jobs", type=int, default=1, help="Parallelism (processes)"
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip artifacts with existing JSON results",
    )
    args = ap.parse_args()

    if args.artifact or ARTIFACT_DIR_NAME_DEFAULT:
        name = args.artifact or ARTIFACT_DIR_NAME_DEFAULT
        run_for_artifact(name)
        return

    # Batch
    root = args.folder or ARTIFACT_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

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
            out = os.path.join(RESULTS_DIR, f"elliptic_LR_{name}.json")
            if os.path.exists(out):
                continue
        candidates.append(name)

    if not candidates:
        print("No matching artifacts found.")
        return

    print(f"Starting runs for {len(candidates)} artifacts...")
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {
                ex.submit(run_for_artifact, name): name for name in candidates
            }
            for f in as_completed(futs):
                name = futs[f]
                try:
                    path = f.result()
                    print(f"Finished: {name} -> {path}")
                except Exception as e:
                    print(f"Error for {name}: {e}")
    else:
        for name in candidates:
            try:
                path = run_for_artifact(name)
                print(f"Finished: {name} -> {path}")
            except Exception as e:
                print(f"Error for {name}: {e}")


if __name__ == "__main__":
    main()
