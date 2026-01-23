# -*- coding: utf-8 -*-
"""
Elliptic node classification with Gaussian Naive Bayes (refactored)

- Lädt vorbereitete Artefakte (Parquet) aus ../artifacts/elliptic/<ARTIFACT_DIR_NAME>
- Keine Feature Selection
- Mit Scaling (StandardScaler oder MinMaxScaler)
- Oversampling der Minoritätsklasse bis max. 1:1
- Deterministische Hyperopt (np.random.default_rng)
- Ergebnisse als JSON gespeichert unter ../Results_JSON/elliptic_NB_<ARTIFACT_DIR_NAME>.json
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
    find_best_threshold,
    f1_at_t_scorer_factory,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from datetime import datetime

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
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, space_eval


# KONFIGURATION
ARTIFACT_DIR_NAME_DEFAULT = None  # bisheriger fixer Name entfällt

# Schalter: Feature Importance rechnen oder dummy-Werte schreiben
COMPUTE_FEATURE_IMPORTANCE = (
    False  # auf False setzen, um Importance-Berechnung zu deaktivieren
)

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "NB"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class PreFittedNBWrapper:
    """Wrapper: nur Scaler + fertig trainierter NB, keine PCA mehr."""

    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf = clf
        self.classes_ = getattr(clf, "classes_", None)

    def fit(self, X, y=None):
        # Wrapper ist bereits „pre-fitted“
        return self

    def _transform(self, X):
        return self.scaler.transform(X)

    def predict(self, X):
        return self.clf.predict(self._transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._transform(X))


def apply_balancing_strategy(X, y, params):
    strategy = params["balancing_strategy"]
    if strategy == "none":
        return X, y
    elif strategy == "oversampling":
        ratio = float(params.get("oversample_ratio", 1.0))
        X_df = pd.DataFrame(X)
        y_df = pd.Series(y, name="label")
        df = pd.concat([X_df, y_df], axis=1)
        min_df = df[df["label"] == 0]
        maj_df = df[df["label"] == 1]
        if len(min_df) == 0 or len(maj_df) == 0:
            return X, y
        # Anzahl Minoritäts-Samples nach Oversampling
        target_min = int(np.clip(ratio * len(maj_df), 1, 10**9))
        min_up = resample(
            min_df, replace=True, n_samples=target_min, random_state=42
        )
        df_bal = pd.concat([maj_df, min_up]).sample(frac=1, random_state=42)
        return df_bal.drop(columns=["label"]).values, df_bal["label"].values
    elif strategy == "class_prior_":
        return X, y
    else:
        return X, y


def safe_predict_proba(clf, X, epsilon=1e-10):
    """Sichere Wahrscheinlichkeiten mit Clipping"""
    proba = clf.predict_proba(X)
    proba = np.clip(proba, epsilon, 1.0 - epsilon)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


def scale_train_val_test(X_train, X_val, X_test, method="standard"):
    """
    Skaliert Trainings-, Validierungs- und Testdaten basierend auf der angegebenen Methode.

    method:
      - "standard": StandardScaler (z-Transformation)
      - "minmax":   MinMaxScaler ([0,1]-Skalierung)

    Rückgabe:
      (X_train_scaled, X_val_scaled, X_test_scaled)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unbekannte Skalierungs-Methode: {method}")

    # Nur auf Train fitten, dann auf Val/Test anwenden
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


# HYPERPARAMETER-SPACE
params_space = {
    # Glättungsparameter (Naive Bayes)
    "var_smoothing": hp.loguniform(
        "var_smoothing", np.log(1e-6), np.log(1e-1)
    ),
    # Balancing-Strategie
    "balancing_strategy": hp.choice(
        "balancing_strategy", ["none", "oversampling", "class_prior_"]
    ),
    # Oversampling-Ratio nur relevant bei balancing_strategy="oversampling"
    "oversample_ratio": hp.uniform("oversample_ratio", 0.5, 1.0),
    # Skalierungsmethode (StandardScaler oder MinMaxScaler)
    "scaler": hp.choice("scaler", ["standard", "minmax"]),
}


def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
) -> str:
    """
    Führt den kompletten NB-Workflow für EIN Artefaktordner aus.
    Gibt den Pfad zur erzeugten JSON-Datei zurück.
    """
    import pandas as pd
    from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        balanced_accuracy_score,
        log_loss,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.naive_bayes import GaussianNB

    # Artefakt-Pfade
    artifact_dir = os.path.join(artifact_root, artifact_dir_name)
    print(f"\n=== Using artifacts from: {artifact_dir} ===")
    if not has_required_files(artifact_dir):
        raise FileNotFoundError(
            f"Artefakt-Dateien unvollständig unter: {artifact_dir}"
        )

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

    # Spalten 0/1 sind txId und time_step -> per Name entfernen (falls vorhanden)
    drop_cols = [c for c in ["txId", "time_step"] if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Labels als int (0/1), Fraud=0
    y_train = pd.Series(y_train).astype(int)
    y_val = pd.Series(y_val).astype(int)
    y_test = pd.Series(y_test).astype(int)

    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # in NumPy
    Xtr = X_train.to_numpy(dtype=np.float64)
    Xva = X_val.to_numpy(dtype=np.float64)
    Xte = X_test.to_numpy(dtype=np.float64)

    # HYPEROPT – OBJECTIVE
    def objective(params):
        # 1) Scaling (fit auf Train, transform auf Val)
        Xtr_s, Xva_s, _ = scale_train_val_test(
            Xtr, Xva, Xte, method=params["scaler"]
        )

        # 2) Balancing-Strategie anwenden (genau wie bisher bei dir)
        Xtr_bal, ytr_bal = apply_balancing_strategy(
            Xtr_s, y_train.values, params
        )

        # 3) Priors nur, wenn Strategie "class_prior_" gewählt ist
        priors = None
        if params["balancing_strategy"] == "class_prior_":
            class_counts = np.bincount(ytr_bal)
            priors = class_counts / class_counts.sum()

        # 4) Modell trainieren
        clf = GaussianNB(
            var_smoothing=float(params["var_smoothing"]), priors=priors
        )
        t0 = time.time()
        clf.fit(Xtr_bal, ytr_bal)
        t1 = time.time()

        # 5) Validierung (Klasse 0 = Fraud)
        pred = clf.predict(Xva_s)
        proba = safe_predict_proba(clf, Xva_s)

        f1 = f1_score(y_val, pred, pos_label=0, zero_division=0)
        prec = precision_score(y_val, pred, pos_label=0, zero_division=0)
        rec = recall_score(y_val, pred, pos_label=0, zero_division=0)
        bacc = balanced_accuracy_score(y_val, pred)
        ll = log_loss(y_val, proba, labels=clf.classes_)
        acc = accuracy_score(y_val, pred)

        fraud_idx = int(np.where(clf.classes_ == 0)[0][0])
        roc_auc = roc_auc_score(
            (y_val.values == 0).astype(int), proba[:, fraud_idx]
        )
        pr_auc = average_precision_score(
            y_val, proba[:, fraud_idx], pos_label=0
        )

        print(
            f"Fit {t1 - t0:.2f}s | F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
            f"BAcc={bacc:.4f} LogLoss={ll:.4f} Acc={acc:.4f} "
            f"ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} | params={params}"
        )

        # Minimierungsziel: LogLoss
        return {"loss": ll, "status": STATUS_OK}

    # Hyperopt-Suche
    trials = Trials()
    best = fmin(
        fn=objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    best_params = space_eval(params_space, best)
    print("\nBest params:", best_params)

    # REFIT AUF TRAIN+VAL & EVAL AUF TEST
    print("\nRefitting best model on TRAIN+VAL and evaluating on test set...")

    # A) TRAIN+VAL im Originalraum zusammenbauen
    Xtv_raw = np.vstack([X_train.values, X_val.values])
    ytv = np.concatenate([y_train.values, y_val.values])

    # B) Skalierung mit dem gewählten Scaler (ohne PCA)
    if best_params["scaler"] == "standard":
        scaler_tv = StandardScaler()
    elif best_params["scaler"] == "minmax":
        scaler_tv = MinMaxScaler()
    else:
        raise ValueError(
            f"Unbekannte Skalierungsmethode: {best_params['scaler']}"
        )

    Xtv_s = scaler_tv.fit_transform(Xtv_raw)
    Xva_tv_s = scaler_tv.transform(X_val.values)
    Xte_tv_s = scaler_tv.transform(X_test.values)

    # C) Balancing-Strategie auf TRAIN+VAL anwenden
    Xtv_bal, ytv_bal = apply_balancing_strategy(Xtv_s, ytv, best_params)

    # D) Priors für finalen Fit (falls gewählt) aus TRAIN+VAL ableiten
    priors = None
    if best_params["balancing_strategy"] == "class_prior_":
        counts = np.bincount(ytv_bal)
        priors = counts / counts.sum()

    # E) Finales NB auf skaliertem TRAIN+VAL fitten
    clf = GaussianNB(
        var_smoothing=float(best_params["var_smoothing"]), priors=priors
    )
    clf.fit(Xtv_bal, ytv_bal)

    # F) Wrapper mit neuem Scaler für Permutation & Test benutzen

    wrapped = PreFittedNBWrapper(scaler_tv, clf)

    # Best-Threshold auf VALIDATION (robust):
    fraud_idx_val = int(np.where(wrapped.classes_ == 0)[0][0])
    proba_val = wrapped.predict_proba(X_val.values)[:, fraud_idx_val]
    t_star, f1_val_star = find_best_threshold(
        y_val.values, proba_val, pos_label=0
    )

    # G) Permutation Importance auf VALIDATION (Original-Features!)
    # WICHTIG: scorer muss (estimator, X, y) -> float sein, mit festem Schwellenwert t_star

    # FEATURE IMPORTANCE: optional berechnen oder dummy
    if COMPUTE_FEATURE_IMPORTANCE:

        scorer = f1_at_t_scorer_factory(t_star, pos_label=0)

        perm = permutation_importance(
            estimator=wrapped,  # PreFitted Wrapper (Scaler + NB)
            X=X_val.values,  # konsistent mit t_star vom VAL-Set
            y=y_val.values,
            scoring=scorer,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )

        feature_importance_perm = perm.importances_mean
        perm_importances = perm.importances_mean
        perm_ranks = (-perm_importances).argsort()
        calculation_method_perm = (
            "permutation_importance@F1(t*=VAL, pos=Fraud=0)"
        )

        # Gemeinsamer Feature-Index
        feature_names = X_train.columns.tolist()

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

        print("\n=== Ranking – Permutation (Top 10) ===")
        print(df_perm.head(10))

    else:

        # KEINE Feature Importance berechnen, stattdessen Nullen/Dummies schreiben
        print(
            "Skipping feature importance computation (COMPUTE_FEATURE_IMPORTANCE=False)."
        )
        feature_names = X_train.columns.tolist()
        if feature_names is None:
            # Fallback: einfache Index-Namen
            feature_names = [f"f{i}" for i in range(Xtv_s.shape[1])]

        n_features = len(feature_names)
        feature_importance_perm = np.zeros(n_features, dtype=float)
        calculation_method_perm = "skipped (COMPUTE_FEATURE_IMPORTANCE=False)"

        df_perm = pd.DataFrame(
            {
                "feature": feature_names,
                "perm_importance": feature_importance_perm,
            }
        )
        df_perm["perm_rank"] = np.arange(1, len(df_perm) + 1)

    # H) Test-Evaluation über den Wrapper (Original-Features)
    proba_test_full = safe_predict_proba(wrapped, X_test.values)
    fraud_idx = int(np.where(wrapped.classes_ == 0)[0][0])
    proba_test = proba_test_full[:, fraud_idx]
    pred_test = np.where(proba_test >= t_star, 0, 1)

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test_full, labels=wrapped.classes_)
    acc = accuracy_score(y_test, pred_test)

    # Fraud als positive Klasse (1) für AUC
    y_test_fraud = (y_test.values == 0).astype(int)
    roc_auc = roc_auc_score(y_test_fraud, proba_test)
    pr_auc = average_precision_score(y_test, proba_test, pos_label=0)

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
        # Fraud- und No-Fraud-Wahrscheinlichkeiten aus der vollen Matrix
        proba_fraud = proba_test_full[:, fraud_idx]
        proba_nofraud = (
            proba_test_full[:, 1 - fraud_idx]
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

        os.makedirs(results_dir, exist_ok=True)
        preds_path = os.path.join(
            results_dir, f"elliptic_NB_{artifact_dir_name}__testpreds.parquet"
        )
        df_preds.to_parquet(preds_path, index=False)
        print(f"✔ Test-Predictions gespeichert unter: {preds_path}")
    except Exception as e:
        print(f"Konnte Test-Predictions nicht speichern: {e}")

    # JSON-EXPORT
    results_summary = {
        "run_info": {
            "model": "GaussianNB",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": 42,
            "balancing_strategy": best_params["balancing_strategy"],
            "scaling": f"{best_params['scaler']} (FINAL: fit on TRAIN+VAL)",
            "dim_reduction": "none",
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
        },
    }

    json_path = os.path.join(
        results_dir, f"elliptic_NB_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to JSON: {json_path}")

    artifact_tag = artifact_dir_name
    out_base = Path(results_dir)

    # 1) DataFrames robust speichern
    df_perm_path = out_base / f"elliptic_NB_{artifact_tag}__perm.parquet"
    df_perm.to_parquet(df_perm_path, index=False)

    # 2) Hyperopt Trials speichern (Pickle via joblib)
    trials_path = out_base / f"elliptic_NB_{artifact_tag}__trials.joblib"
    joblib.dump(trials, trials_path, compress=3)

    # 3) „Reloadables“ (Modell + Scaler + Schwelle)
    bundle = {
        "model": clf,
        "threshold_t_star": float(t_star),
        "fraud_proba_index": int(fraud_idx),
        "feature_names": feature_names,
        "scaler": scaler_tv,
        "best_params": best_params,
    }

    bundle_path = out_base / f"elliptic_NB_{artifact_tag}__bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    print(f"Saved extras:\n- {df_perm_path}\n- {trials_path}\n- {bundle_path}")

    return json_path


def main():
    ap = argparse.ArgumentParser(
        description="GaussianNB Elliptic – Single & Batch Runner"
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

    # Einzelrun (artifact) hat Vorrang
    if args.artifact or ARTIFACT_DIR_NAME_DEFAULT:
        name = args.artifact or ARTIFACT_DIR_NAME_DEFAULT
        run_for_artifact(name)
        return

    # Batch über Ordner
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
            out = os.path.join(RESULTS_DIR, f"elliptic_NB_{name}.json")
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
                ex.submit(run_for_artifact, name): name for name in candidates
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
                path = run_for_artifact(name)
                print(f"Fertig: {name} -> {path}")
            except Exception as e:
                print(f"Fehler bei {name}: {e}")


if __name__ == "__main__":
    main()
