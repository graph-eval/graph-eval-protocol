# -*- coding: utf-8 -*-
"""
Elliptic node classification with MLP (refactored; StandardScaler; no SelectKBest)

- Loads preprocessed artifacts (Parquet) from ../artifacts/elliptic/<ARTIFACT_DIR_NAME>
- Consistent labels: Fraud = 0, No-Fraud = 1
- Feature scaling via StandardScaler (fit on TRAIN, transform on VAL and TEST)
- Oversamples the minority class to a 1:1 class ratio
- Uses deterministic Hyperopt via np.random.default_rng
- Saves results as JSON to ../Results_JSON/elliptic_MLP_<ARTIFACT_DIR_NAME>.json
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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    make_scorer,
    roc_auc_score,
    average_precision_score,
)
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from pathlib import Path
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK, space_eval

# KONFIGURATION
ARTIFACT_DIR_NAME_DEFAULT = None

# Schalter: Feature Importance rechnen oder dummy-Werte schreiben
COMPUTE_FEATURE_IMPORTANCE = (
    False  # auf False setzen, um Importance-Berechnung zu deaktivieren
)

THIS_FILE = Path(__file__).resolve()
EXPERIMENTS_ROOT = THIS_FILE.parents[1]

ARTIFACT_ROOT = EXPERIMENTS_ROOT / "artifacts" / "elliptic"
RESULTS_DIR = EXPERIMENTS_ROOT / "Results_HyPa" / "MLP"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_balancing_strategy(X, y, params):
    """Wendet die optimierte Balancing-Strategie an"""
    strategy = params["balancing_strategy"]

    if strategy == "none":
        # Kein Balancing - originale Daten
        return X, y

    elif strategy == "oversampling":
        # Sicheres 1:1 Oversampling
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

        return X_bal, y_bal

    else:
        # Fallback: Kein Balancing
        return X, y


def calculate_mlp_intrinsic_importance(clf, X, y):
    """
    Berechnet intrinsische Feature Importance für MLP basierend auf
    den absoluten Gewichten der ersten Schicht.
    """
    # MLP hat coefs_ für jede Schicht, wir nehmen die erste Schicht (Input → Hidden)
    if hasattr(clf, "coefs_") and len(clf.coefs_) > 0:
        # Absolute Werte der Gewichte der ersten Schicht
        # coefs_[0] hat Shape (n_features, n_neurons_first_layer)
        importance = np.abs(clf.coefs_[0]).mean(axis=1)
    else:
        # Fallback: Gleichverteilung
        print(
            "⚠️  No coefs_ available for MLP, using uniform importance as fallback"
        )
        importance = np.ones(X.shape[1]) / X.shape[1]

    return importance


# HYPERPARAMETER-SPACE
params_space = {
    # DISKRETE ARCHITEKTUR-BUCKETS (viel effizienter!)
    "hidden_layer_sizes": hp.choice(
        "hidden_layer_sizes",
        [
            (64,),  # 1 Layer, 64 Neuronen
            (128,),  # 1 Layer, 128 Neuronen
            (256,),  # 1 Layer, 256 Neuronen
            (64, 32),  # 2 Layer: 64 -> 32
            (128, 64),  # 2 Layer: 128 -> 64
            (128, 128),  # 2 Layer: 128 -> 128
            (256, 128),  # 2 Layer: 256 -> 128
        ],
    ),
    # STÄRKERE REGULARISIERUNG (breiterer Bereich)
    "alpha": hp.loguniform("alpha", np.log(1e-6), np.log(1e-2)),
    # LEARNING RATE
    "lr_init": hp.loguniform("lr_init", np.log(1e-4), np.log(5e-2)),
    # AKTIVIERUNGSFUNKTION
    "activation": hp.choice("activation", ["relu", "tanh"]),
    # BATCH SIZE
    "batch_size": hp.choice("batch_size", [64, 128, 256]),
    "balancing_strategy": hp.choice(
        "balancing_strategy", ["none", "oversampling"]
    ),
}


def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
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

    # Labels als **Series** (0/1), Fraud=0
    y_train = (
        pd.read_parquet(os.path.join(artifact_dir, "y_train.parquet"))
        .iloc[:, 0]
        .astype(int)
    )
    y_val = (
        pd.read_parquet(os.path.join(artifact_dir, "y_validation.parquet"))
        .iloc[:, 0]
        .astype(int)
    )
    y_test = (
        pd.read_parquet(os.path.join(artifact_dir, "y_test.parquet"))
        .iloc[:, 0]
        .astype(int)
    )

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

    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print("\nClass distribution in training set (0=Fraud,1=No-Fraud):")
    print(y_train.value_counts().sort_index())

    # StandardScaler (fit nur auf Train)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)

    # HYPEROPT – OBJECTIVE
    def objective(params):
        """
        MLP mit Eval auf Validation. Keine Feature Selection.
        """
        # ARCHITEKTUR direkt aus Choice übernehmen
        hidden_layer = params["hidden_layer_sizes"]

        # Balancing-Strategie anwenden
        Xtr_bal, ytr_bal = apply_balancing_strategy(
            Xtr, y_train.values, params
        )

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer,
            activation=params["activation"],
            batch_size=params["batch_size"],
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["lr_init"]),
        )

        t0 = time.time()
        clf.fit(Xtr_bal, ytr_bal)
        t1 = time.time()

        pred = clf.predict(Xva)
        proba = clf.predict_proba(Xva)

        # Fraud = positive class (0)
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
            f"Train {t1 - t0:.2f}s | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | BAcc={bacc:.4f} | "
            f"LogLoss={ll:.4f} | Acc={acc:.4f} | "
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
        trials=trials,
        algo=tpe.suggest,
        max_evals=50,
        rstate=rng,
    )
    best_params = space_eval(params_space, best)
    print(f"\nBest params: {best_params}")

    # REFIT AUF TRAIN & EVAL AUF TEST
    print(
        "\nRefitting best model (Best-Epoch-Transfer) on TRAIN+VAL and evaluating on test..."
    )

    # A) beste Epoche auf TRAIN ermitteln (ES nur hier)
    Xtr_bal, ytr_bal = apply_balancing_strategy(
        Xtr, y_train.values, best_params
    )
    probe = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        batch_size=best_params["batch_size"],
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
        alpha=float(best_params["alpha"]),
        learning_rate_init=float(best_params["lr_init"]),
    )
    probe.fit(Xtr_bal, ytr_bal)

    proba_val_full = probe.predict_proba(
        Xva
    )  # wichtig: Xva wurde mit dem ersten Scaler skaliert
    fraud_idx_val = int(np.where(probe.classes_ == 0)[0][0])
    t_star, f1_val_star = find_best_threshold(
        y_val.values, proba_val_full[:, fraud_idx_val], pos_label=0
    )
    print(f"Best threshold (VAL): t*={t_star:.3f} | F1@t*={f1_val_star:.4f}")

    best_n_iter = int(getattr(probe, "n_iter_", 200))
    print(f"Best n_iter on TRAIN via ES: {best_n_iter}")

    # B) Train+Val zusammenbauen + Scaler auf Train+Val
    Xtv_raw = np.vstack([X_train.values, X_val.values])
    ytv = np.concatenate([y_train.values, y_val.values])
    scaler_tv = StandardScaler()
    Xtv = scaler_tv.fit_transform(Xtv_raw)
    Xte_tv = scaler_tv.transform(X_test.values)

    # C) Balancing Train+Val
    Xtv_bal, ytv_bal = apply_balancing_strategy(Xtv, ytv, best_params)

    # D) Finales Modell OHNE ES, mit fester Epoche
    clf = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        batch_size=best_params["batch_size"],
        random_state=42,
        max_iter=best_n_iter,  # Best-Epoch-Transfer
        early_stopping=False,  # wichtig: aus
        tol=1e-4,
        alpha=float(best_params["alpha"]),
        learning_rate_init=float(best_params["lr_init"]),
    )
    clf.fit(Xtv_bal, ytv_bal)

    # FEATURE IMPORTANCE: optional berechnen oder dummy
    if COMPUTE_FEATURE_IMPORTANCE:

        # Intrinsische Feature Importance für MLP
        print("Calculating intrinsic MLP importance...")
        feature_importance_intrinsic = calculate_mlp_intrinsic_importance(
            clf, Xtv_bal, ytv_bal
        )
        calculation_method_intrinsic = "mlp_first_layer_weights_abs"

        # Permutation Importance berechnen
        print("Calculating permutation importance...")
        Xva_tv = scaler_tv.transform(X_val.values)

        scorer = f1_at_t_scorer_factory(t_star, pos_label=0)
        perm_importance = permutation_importance(
            clf,
            Xva_tv,
            y_val,
            n_repeats=10,
            random_state=42,
            n_jobs=1,
            scoring=scorer,
        )

        feature_importance_perm = perm_importance.importances_mean
        calculation_method_perm = "permutation_importance_f1_fraud"

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

    # E) Test mit Xte_tv
    proba_test = clf.predict_proba(Xte_tv)
    fraud_idx_test = int(np.where(clf.classes_ == 0)[0][0])
    proba_fraud_te = proba_test[:, fraud_idx_test]

    # Fraud=0 bei hoher Fraud-Wahrscheinlichkeit:
    pred_test = np.where(proba_fraud_te >= t_star, 0, 1)

    f1 = f1_score(y_test, pred_test, pos_label=0, zero_division=0)
    prec = precision_score(y_test, pred_test, pos_label=0, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=0, zero_division=0)
    bacc = balanced_accuracy_score(y_test, pred_test)
    ll = log_loss(y_test, proba_test, labels=clf.classes_)
    acc = accuracy_score(y_test, pred_test)

    # Für AUC wird Fraud (Label 0) nach 1 gemappt.
    roc_auc = roc_auc_score((y_test.values == 0).astype(int), proba_fraud_te)
    pr_auc = average_precision_score(y_test, proba_fraud_te, pos_label=0)

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
            results_dir, f"elliptic_MLP_{artifact_dir_name}__testpreds.parquet"
        )
        df_preds.to_parquet(preds_path, index=False)
        print(f"✔ Test-Predictions gespeichert unter: {preds_path}")
    except Exception as e:
        print(f"Konnte Test-Predictions nicht speichern: {e}")

    # Save results as JSON
    results_summary = {
        "run_info": {
            "model": "MLPClassifier",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": 42,
            "scaler": "StandardScaler (HP: fit on TRAIN; FINAL: fit on TRAIN+VAL)",
            "feature_selection": "none",
            "balancing_strategy": best_params["balancing_strategy"],
        },
        "validation": {
            "best_threshold_t": float(t_star),
            "f1_at_best_t_val": float(f1_val_star),
            "best_n_iter_train_es": int(best_n_iter),
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
                "importance_type": "first_layer_weights_abs",
                "weights_available": hasattr(clf, "coefs_")
                and len(clf.coefs_) > 0,
            },
        },
    }

    json_path = os.path.join(
        results_dir, f"elliptic_MLP_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to JSON: {json_path}")

    artifact_tag = artifact_dir_name
    out_base = Path(results_dir)

    # 1) DataFrames robust speichern
    df_perm_path = out_base / f"elliptic_MLP_{artifact_tag}__perm.parquet"
    df_intr_path = out_base / f"elliptic_MLP_{artifact_tag}__intr.parquet"
    df_perm.to_parquet(df_perm_path, index=False)
    df_intr.to_parquet(df_intr_path, index=False)

    # 2) Hyperopt Trials speichern (Pickle via joblib)
    trials_path = out_base / f"elliptic_MLP_{artifact_tag}__trials.joblib"
    joblib.dump(trials, trials_path, compress=3)

    # 3) Optionale „Reloadables“ (Modell + Scaler + Schwelle)
    bundle = {
        "model": clf,
        "scaler_tv": scaler_tv,
        "threshold_t_star": float(t_star),
        "fraud_proba_index": int(fraud_idx_test),
        "feature_names": feature_names,  # für sauberes Re-Mapping
    }
    bundle_path = out_base / f"elliptic_MLP_{artifact_tag}__bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    print(
        f"Saved extras:\n- {df_perm_path}\n- {df_intr_path}\n- {trials_path}\n- {bundle_path}"
    )

    return json_path


def main():
    ap = argparse.ArgumentParser(
        description="MLP Elliptic – Single & Batch Runner"
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
            out = os.path.join(RESULTS_DIR, f"elliptic_MLP_{name}.json")
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
