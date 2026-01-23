# -*- coding: utf-8 -*-
"""
Graph Convolutional Network - EndToEnd

- Threshold tuning on the validation set (Balanced Accuracy by default)
- Best-epoch transfer during final training (prevents AUC drop)
- Safe Hyperopt search space (no division-based objective functions)
- Scaling fix retained (scaling applied only to labeled training rows)

Patched (NaN / overflow hardening):
- Numerically robust evaluate(): nan_to_num, row normalization, clipping, fallback log loss
- _safe_log_loss() as a robust wrapper around sklearn.log_loss
- Training guards: skip/abort on non-finite loss, gradient clipping
- Softmax and logits are sanitized before and after the softmax operation
"""

import os

# Reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import random

SEED = 42
random.seed(SEED)
import numpy as np

np.random.seed(SEED)
import dgl
import dgl.function as fn
import torch as th

th.manual_seed(SEED)
dgl.random.seed(SEED)
th.use_deterministic_algorithms(True)
import json
import time
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import joblib
import argparse

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
)
from pathlib import Path
from datetime import datetime
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK, STATUS_FAIL, space_eval
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

# Device configuration
device = th.device("cpu")  # für strikte Reproduzierbarkeit CPU erzwingen
print(f"Using device: {device}")

# Config
ARTIFACT_DIR_NAME_DEFAULT = None
ARTIFACT_ROOT = os.path.join(".", "artifacts", "elliptic")
RESULTS_DIR = os.path.join(".", "Results_JSON", "GCN")
os.makedirs(RESULTS_DIR, exist_ok=True)

number_of_epochs = 200
number_of_hyperopt = 40
add_unlabeled_nodes = False
random_split = False

# Threshold-Tuning Zielmetrik: 'balanced_accuracy' | 'f1' | 'accuracy'
THRESH_METRIC = "balanced_accuracy"

# metrics accumulator (Validation während Hyperopt)
test_metrics = {
    k: []
    for k in [
        "indices_fraud",
        "indices_no_fraud",
        "labels_fraud",
        "labels_no_fraud",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "roc_auc",
        "pr_auc",
    ]
}


def _ensure_txid_column(df: pd.DataFrame) -> pd.DataFrame:
    if "txId" in df.columns:
        df["txId"] = df["txId"].astype(str)
        return df
    idx_name = (df.index.name or "").lower()
    if idx_name in {"txid", "tx_id", "txhash", "id", "node_id"}:
        df = df.reset_index().rename(columns={df.index.name: "txId"})
        df["txId"] = df["txId"].astype(str)
        return df
    for alt in ["txid", "tx_id", "TXID", "TxId", "id", "node_id", "txhash"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "txId"})
            df["txId"] = df["txId"].astype(str)
            return df
    df = df.reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": "txId"})
    elif df.columns[0] != "txId":
        df = df.rename(columns={df.columns[0]: "txId"})
    df["txId"] = df["txId"].astype(str)
    return df


# Model
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, params, use_dropout=True):
        super().__init__()
        # message/reduce (keine Divisionen)
        if params["message_function"] == "copy_u":
            self.msg = fn.copy_u("h", "m")
        else:
            self.msg = getattr(fn, str(params["message_function"]))(
                "h", "h", out="m"
            )
        self.red = getattr(fn, str(params["reduce_function"]))(
            msg="m", out="h"
        )
        # linear + dropout
        self.linear = nn.Linear(in_feats, out_feats, bias=params["bias"])
        self.dropout = (
            nn.Dropout(params.get("dropout_rate", 0.0))
            if use_dropout
            else nn.Identity()
        )

    def forward(self, g, x):
        with g.local_scope():
            g.ndata["h"] = x
            g.update_all(self.msg, self.red)
            h = g.ndata["h"]
            h = self.linear(h)
            return self.dropout(h)


class Net1(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.layer1 = GCNLayer(in_feats, 2, params, use_dropout=False)

    def forward(self, g, x):
        return self.layer1(g, x)


class Net2(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params
        h2 = int(params["layer2_input_neuron"])
        self.layer1 = GCNLayer(in_feats, h2, params, use_dropout=True)
        self.layer2 = GCNLayer(h2, 2, params, use_dropout=False)

    def forward(self, g, x):
        act = getattr(F, str(self.params["activation_function"]))
        x = act(self.layer1(g, x))
        x = self.layer2(g, x)
        return x


class Net3(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params
        h2 = int(params["layer2_input_neuron"])
        h3 = int(params["layer3_input_neuron"])
        self.layer1 = GCNLayer(in_feats, h2, params, use_dropout=True)
        self.layer2 = GCNLayer(h2, h3, params, use_dropout=True)
        self.layer3 = GCNLayer(h3, 2, params, use_dropout=False)

    def forward(self, g, x):
        act = getattr(F, str(self.params["activation_function"]))
        x = act(self.layer1(g, x))
        x = act(self.layer2(g, x))
        x = self.layer3(g, x)
        return x


class Net4(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params
        h2 = int(params["layer2_input_neuron"])
        h3 = int(params["layer3_input_neuron"])
        h4 = int(params["layer4_input_neuron"])
        self.layer1 = GCNLayer(in_feats, h2, params, use_dropout=True)
        self.layer2 = GCNLayer(h2, h3, params, use_dropout=True)
        self.layer3 = GCNLayer(h3, h4, params, use_dropout=True)
        self.layer4 = GCNLayer(h4, 2, params, use_dropout=False)

    def forward(self, g, x):
        act = getattr(F, str(self.params["activation_function"]))
        x = act(self.layer1(g, x))
        x = act(self.layer2(g, x))
        x = act(self.layer3(g, x))
        x = self.layer4(g, x)
        return x


# Data loading
def robust_load_elliptic_dataset(params, artifact_dir):
    def _rp(fname):
        return os.path.join(artifact_dir, fname)

    req = [
        "X_train.parquet",
        "X_test.parquet",
        "X_validation.parquet",
        "y_train.parquet",
        "y_test.parquet",
        "y_validation.parquet",
    ]
    for r in req:
        if not os.path.exists(_rp(r)):
            raise FileNotFoundError(
                f"Artefakt fehlt: {r} unter {artifact_dir}"
            )

    try:
        X_tr = _read_parquet_required(_rp("X_train.parquet"))
        X_te = _read_parquet_required(_rp("X_test.parquet"))
        X_va = _read_parquet_required(_rp("X_validation.parquet"))

        X_tr = _ensure_txid_column(X_tr)
        X_te = _ensure_txid_column(X_te)
        X_va = _ensure_txid_column(X_va)

        y_tr = _read_parquet_required(_rp("y_train.parquet")).iloc[:, 0]
        y_te = _read_parquet_required(_rp("y_test.parquet")).iloc[:, 0]
        y_va = _read_parquet_required(_rp("y_validation.parquet")).iloc[:, 0]

        for df in (X_tr, X_te, X_va):
            df["txId"] = df["txId"].astype(str)
            df.set_index("txId", inplace=True)

        if y_tr.index.equals(pd.RangeIndex(len(y_tr))):
            y_tr.index = X_tr.index
        if y_te.index.equals(pd.RangeIndex(len(y_te))):
            y_te.index = X_te.index
        if y_va.index.equals(pd.RangeIndex(len(y_va))):
            y_va.index = X_va.index

        y_tr = pd.to_numeric(y_tr, errors="coerce")
        y_te = pd.to_numeric(y_te, errors="coerce")
        y_va = pd.to_numeric(y_va, errors="coerce")

        # Stack with split
        X_tr["_split"] = "train"
        X_te["_split"] = "test"
        X_va["_split"] = "val"
        DF = pd.concat([X_tr, X_te, X_va], axis=0)
        y_all = pd.concat([y_tr, y_te, y_va], axis=0).reindex(DF.index)

        # valid labels {0,1}
        valid_mask = y_all.isin([0, 1])
        split = DF.pop("_split")

        train_mask_np = (split == "train").to_numpy() & valid_mask.to_numpy()
        val_mask_np = (split == "val").to_numpy() & valid_mask.to_numpy()
        test_mask_np = (split == "test").to_numpy() & valid_mask.to_numpy()

        if train_mask_np.sum() == 0:
            raise RuntimeError(
                "Keine gelabelten Train-Knoten gefunden (split=='train' & gültige Labels)."
            )

        y_all_clean = y_all.fillna(0).astype(int)

        # Build graph
        edge_path = os.path.join(
            ".", "EllipticDataSet", "elliptic_txs_edgelist.csv"
        )
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"Edgelist fehlt unter {edge_path}")
        edges = pd.read_csv(edge_path, usecols=["txId1", "txId2"], dtype=str)
        idx_map = {tx: i for i, tx in enumerate(DF.index)}
        mask_both = edges["txId1"].isin(idx_map) & edges["txId2"].isin(idx_map)
        E = edges.loc[mask_both]
        src = E["txId1"].map(idx_map).to_numpy(np.int64)
        dst = E["txId2"].map(idx_map).to_numpy(np.int64)

        g = dgl.graph((th.tensor(src), th.tensor(dst)), num_nodes=len(DF))
        g = dgl.to_simple(dgl.add_reverse_edges(g))
        g = dgl.add_self_loop(g)

        # Features
        drop_cols = [c for c in ["time_step"] if c in DF.columns]
        FEATURES = DF.drop(columns=drop_cols, errors="ignore").apply(
            pd.to_numeric, errors="coerce"
        )

        # Scaling (nur gelabelte Train-Rows)
        labeled_train_rows = DF.index[train_mask_np]
        tr = FEATURES.loc[labeled_train_rows]
        mu = tr.mean()
        sd = tr.std(ddof=0)
        sd = sd.replace(0, 1.0).fillna(1.0)
        mu = mu.fillna(0.0)

        FEATURES = (FEATURES - mu) / sd
        FEATURES = FEATURES.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Diagnostics
        tr_scaled = FEATURES.loc[labeled_train_rows]
        print("\n=== NaN ANALYSIS ===")
        print(f"NaN in X_train: {pd.isna(tr_scaled).to_numpy().sum()}")
        print(
            f"NaN in X_val:   {pd.isna(FEATURES.loc[DF.index[(split=='val')]]).to_numpy().sum()}"
        )
        print(
            f"NaN in X_test:  {pd.isna(FEATURES.loc[DF.index[(split=='test')]]).to_numpy().sum()}"
        )
        print(f"NaN in y_train: {y_tr.isna().sum()}")
        print(f"NaN in y_val:   {y_va.isna().sum()}")
        print(f"NaN in y_test:  {y_te.isna().sum()}")

        feats = th.tensor(FEATURES.values, dtype=th.float32)

        labels = th.tensor(
            y_all_clean.values.astype(int), dtype=th.long, device=device
        )
        train_mask = th.from_numpy(train_mask_np).to(th.bool).to(device)
        val_mask = th.from_numpy(val_mask_np).to(th.bool).to(device)
        test_mask = th.from_numpy(test_mask_np).to(th.bool).to(device)

        g = g.to(device)
        feats = feats.to(device)

        # Graph stats
        n = g.number_of_nodes()
        e_tot = g.number_of_edges()
        e_wo_self = e_tot - n
        density = (e_wo_self / (n * (n - 1))) if n > 1 else float("nan")
        deg = (g.in_degrees() + g.out_degrees()).cpu()
        near_iso = int((deg <= 2).sum().item())
        print("\nGraph statistics:")
        print(
            f"  Nodes: {n}, Edges (with self-loops): {e_tot}, Edges (no self): {e_wo_self}"
        )
        print(
            f"  Train nodes (labeled): {train_mask.sum().item()}, Val nodes: {val_mask.sum().item()}, Test nodes: {test_mask.sum().item()}"
        )
        print(f"  Isolated-or-near-isolated (<=2 incl. self+rev): {near_iso}")
        print(f"  Features: {feats.shape[1]}, Labels: {len(labels)}")
        print(f"  Graph density (no self-loops): {density:.6f}")

        df_edge = pd.DataFrame({"node_id1": src, "node_id2": dst})
        return g, feats, labels, train_mask, test_mask, val_mask, df_edge

    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        raise RuntimeError(f"Datenladung fehlgeschlagen: {e}")


# Analysis helpers
def analyze_graph_data(g, features, labels, train_mask, test_mask, val_mask):
    print("\n" + "=" * 50)
    print("GRAPH DATA ANALYSIS")
    print("=" * 50)
    print(f"NaN in features: {th.isnan(features).sum().item()}")
    print(f"Inf in features: {th.isinf(features).sum().item()}")
    print(
        f"Feature range: [{features.min().item():.4f}, {features.max().item():.4f}]"
    )
    print(
        f"Feature mean: {features.mean().item():.4f} ± {features.std().item():.4f}"
    )

    n = g.number_of_nodes()
    e_total = g.number_of_edges()
    e_wo_self = e_total - n
    density = (e_wo_self / (n * (n - 1))) if n > 1 else float("nan")
    print(f"\nGraph nodes: {n}")
    print(f"Graph edges (with self-loops): {e_total} | (no self): {e_wo_self}")
    print(f"Graph density (no self-loops): {density:.6f}")

    masks = {"Train": train_mask, "Validation": val_mask, "Test": test_mask}
    for name, m in masks.items():
        y = labels[m]
        if y.numel():
            fraud = (y == 0).sum().item()
            nofraud = (y == 1).sum().item()
            tot = fraud + nofraud
            ratio = fraud / tot if tot else 0.0
            print(
                f"{name:12} - Fraud: {fraud:4d}, No-Fraud: {nofraud:4d}, Ratio: {ratio:.3f}, Total: {tot:4d}"
            )


def apply_gcn_balancing_strategy(labels_train, params):
    strat = params.get("balancing_strategy", "class_weights")
    print(f"Applying balancing strategy: {strat}")
    if strat == "none":
        return None
    counts = th.bincount(labels_train).clamp(min=1).float()
    K = len(counts)
    return counts.sum() / (K * counts)


# Threshold tuning
def _metric_from_preds(y_true, y_pred, metric="balanced_accuracy"):
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    return balanced_accuracy_score(y_true, y_pred)


def tune_threshold(y_true, proba_pos_class0, metric="balanced_accuracy"):
    ts = np.linspace(0.0, 1.0, 1001)
    best_t, best_s = 0.5, -1.0
    for t in ts:
        y_pred_labels = np.where(proba_pos_class0 >= t, 0, 1)
        s = _metric_from_preds(y_true, y_pred_labels, metric=metric)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


# Safe log loss
def _safe_log_loss(y_true, y_proba_2col, eps=1e-12):
    y = np.asarray(y_true)
    P = np.asarray(y_proba_2col, dtype=np.float64)
    P = np.where(np.isfinite(P), P, 0.0)
    P = np.clip(P, 0.0, 1.0)
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
    P = P / row_sum
    P = np.clip(P, eps, 1.0 - eps)
    P = P / P.sum(axis=1, keepdims=True)
    try:
        return float(log_loss(y, P, labels=[0, 1]))
    except Exception:
        y_bin = (y == 0).astype(np.float64)
        p0 = P[:, 0]
        p1 = 1.0 - p0
        p0 = np.clip(p0, eps, 1 - eps)
        p1 = np.clip(p1, eps, 1 - eps)
        ll = -(y_bin * np.log(p0) + (1 - y_bin) * np.log(p1))
        return float(np.mean(ll))


def calculate_gcn_feature_importance(
    model, g, features, labels, mask, n_repeats=3
):
    print("Calculating GCN feature importance via permutation...")
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        base_pred = th.argmax(logits[mask], dim=1)
        base_acc = (base_pred == labels[mask]).float().mean().item()
    imp = th.zeros(features.shape[1], device=device)
    for j in range(features.shape[1]):
        diffs = []
        for _ in range(n_repeats):
            f_perm = features.clone()
            perm_idx = th.randperm(features.shape[0], device=device)
            f_perm[:, j] = features[perm_idx, j]
            with th.no_grad():
                p_logits = model(g, f_perm)
                p_pred = th.argmax(p_logits[mask], dim=1)
                p_acc = (p_pred == labels[mask]).float().mean().item()
            diffs.append(base_acc - p_acc)
        imp[j] = th.tensor(diffs, device=device).mean()
    return imp.cpu().numpy()


# Evaluation
def evaluate(
    model, g, features, labels, mask, split_name="Test", tuned_threshold=None
):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        logits = th.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        logp = F.log_softmax(logits, 1)
        proba = F.softmax(logits, 1)
        proba = th.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
        proba = proba / (proba.sum(dim=1, keepdim=True) + 1e-12)

        labels_split = labels[mask]
        if tuned_threshold is None:
            _, indices = th.max(logits, dim=1)
            y_pred = indices.cpu().numpy()
        else:
            p0 = proba[:, 0].cpu().numpy()
            y_pred = np.where(p0 >= tuned_threshold, 0, 1)

        y_true = labels_split.cpu().numpy()
        y_proba = proba.cpu().numpy()

        pred_fraud = int((np.array(y_pred) == 0).sum())
        pred_nofraud = int((np.array(y_pred) == 1).sum())
        lbl_fraud = int((y_true == 0).sum())
        lbl_nofraud = int((y_true == 1).sum())

        f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        nll = _safe_log_loss(y_true, y_proba)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        y_true_pos = (y_true == 0).astype(int)
        try:
            roc = roc_auc_score(y_true_pos, y_proba[:, 0])
        except ValueError:
            roc = float("nan")
        try:
            pr = average_precision_score(y_true_pos, y_proba[:, 0])
        except ValueError:
            pr = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        c00, c01, c10, c11 = cm.ravel()

        head = f"{split_name} (τ={'argmax' if tuned_threshold is None else f'{tuned_threshold:.3f}'})"
        print(f"\nDetailed {head} Classification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["Fraud", "No Fraud"],
                zero_division=0,
            )
        )
        print(f"Balanced Accuracy: {bal_acc:.4f}")

        print(f"########## {head} ##########")
        print(f"Pred Fraud = {pred_fraud}")
        print(f"Pred No fraud = {pred_nofraud}")
        print(f"Label Fraud = {lbl_fraud}")
        print(f"Label No fraud = {lbl_nofraud}")
        print(f"F1-Score = {f1:.4f}")
        print(f"precision = {prec:.4f}")
        print(f"recall = {rec:.4f}")
        print(f"acc_Score = {acc:.4f}")
        print(f"Log_Loss = {nll:.4f}")
        print(f"ROC-AUC = {roc:.4f}")
        print(f"PR-AUC = {pr:.4f}")

        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("               Predicted")
        print("               Fraud    No Fraud")
        print(f"Actual Fraud   {c00:6d}   {c01:6d}")
        print(f"Actual NoFraud {c10:6d}   {c11:6d}")

        return (
            roc,
            pr,
            {
                "indices_fraud": pred_fraud,
                "indices_no_fraud": pred_nofraud,
                "labels_fraud": lbl_fraud,
                "labels_no_fraud": lbl_nofraud,
                "f1_score": f1,
                "precision": prec,
                "recall": rec,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "log_loss": float(nll),
                "roc_auc": roc,
                "pr_auc": pr,
                "confusion_matrix": cm.tolist(),
                "cm_c00": int(c00),
                "cm_c01": int(c01),
                "cm_c10": int(c10),
                "cm_c11": int(c11),
            },
        )


def _nan_to_none(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    return obj


def _name_funcs_in_params(params):
    out = {}
    for k, v in params.items():
        out[k] = v.__name__ if callable(v) else v
    return out


def build_oversampled_train_index(
    labels, train_mask, oversample_ratio: float = 1.0
):
    # labels ∈ {0,1}; 0 = Fraud (Minorität)
    idx = th.nonzero(train_mask, as_tuple=False).view(-1).cpu()
    y = labels[idx].cpu()
    idx_min = idx[(y == 0)]
    idx_maj = idx[(y == 1)]
    if len(idx_min) == 0 or len(idx_maj) == 0:
        return idx.to(device)  # kein Oversampling möglich
    target_min = min(int(len(idx_maj) * oversample_ratio), len(idx_maj))
    # mit Replacement auf target_min hochziehen
    if len(idx_min) >= target_min:
        extra = idx_min[th.randperm(len(idx_min))[:target_min]]
    else:
        extra = idx_min[th.randint(0, len(idx_min), (target_min,))]
    pool = th.cat([idx_maj, extra], dim=0)
    perm = th.randperm(len(pool))
    return pool[perm].to(device)


# Hyperopt Space (sichere msg-funcs)
SAFE_MSG_FUNCS = ["copy_u", "u_add_v", "v_add_u", "v_sub_u", "v_mul_u"]
params_space = {
    "optimizer": hp.choice(
        "optimizer",
        [
            "Adam",
            "SGD",
            "Adadelta",
            "Adagrad",
            "AdamW",
            "Adamax",
            "ASGD",
            "RMSprop",
            "Rprop",
        ],
    ),
    "reduce_function": hp.choice("reduce_function", ["sum", "max", "mean"]),
    "message_function": hp.choice("message_function", SAFE_MSG_FUNCS),
    "activation_function": hp.choice(
        "activation_function", ["relu", "elu", "gelu", "leaky_relu", "silu"]
    ),
    "bias": hp.choice("bias", [True, False]),
    "layer2_input_neuron": hp.choice(
        "layer2_input_neuron", np.arange(8, 128, dtype=int)
    ),
    "layer3_input_neuron": hp.choice(
        "layer3_input_neuron", np.arange(8, 64, dtype=int)
    ),
    "layer4_input_neuron": hp.choice(
        "layer4_input_neuron", np.arange(8, 32, dtype=int)
    ),
    "number_of_layers": hp.choice(
        "number_of_layers", np.arange(1, 5, dtype=int)
    ),
    "balancing_strategy": hp.choice(
        "balancing_strategy", ["class_weights", "oversampling", "none"]
    ),
    "oversample_ratio": hp.uniform("oversample_ratio", 0.1, 1.0),
    "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.3),
    "learning_rate": hp.loguniform(
        "learning_rate", np.log(1e-4), np.log(5e-2)
    ),
}


def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
) -> str:
    artifact_dir = os.path.join(artifact_root, artifact_dir_name)

    # Hyperopt
    def objective(params):
        print("Hyperparameters:", params)

        if (
            params["number_of_layers"] >= 3
            and params["layer3_input_neuron"] >= params["layer2_input_neuron"]
        ):
            return {"status": STATUS_FAIL}
        if (
            params["number_of_layers"] >= 4
            and params["layer4_input_neuron"] >= params["layer3_input_neuron"]
        ):
            return {"status": STATUS_FAIL}

        try:
            g, features, labels, train_mask, test_mask, val_mask, df_edge = (
                robust_load_elliptic_dataset(params, artifact_dir=artifact_dir)
            )
        except Exception as e:
            print(f"Datenladung fehlgeschlagen: {e}")
            return {"status": STATUS_FAIL}

        analyze_graph_data(
            g, features, labels, train_mask, test_mask, val_mask
        )

        in_feats = features.size(1)
        if params["number_of_layers"] == 1:
            net = Net1(params, in_feats)
        elif params["number_of_layers"] == 2:
            net = Net2(params, in_feats)
        elif params["number_of_layers"] == 3:
            net = Net3(params, in_feats)
        else:
            net = Net4(params, in_feats)
        net = net.to(device)
        print("Model architecture:", net)

        lr = float(params.get("learning_rate", 1e-2))
        if params["optimizer"] == "SGD":
            optimizer = th.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = getattr(th.optim, str(params["optimizer"]))(
                net.parameters(), lr=lr
            )

        labels_train = labels[train_mask]
        class_weights = apply_gcn_balancing_strategy(labels_train, params)

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        best_val = float("inf")
        patience = 10
        no_impr = 0
        best_state = None
        dur = []

        for epoch in range(number_of_epochs):
            if epoch >= 3:
                t0 = time.time()

            net.train()
            x = features.float()
            logits = net(g, x)
            logits = th.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            logp = F.log_softmax(logits, 1)
            # loss = F.nll_loss(logp[train_mask], labels_train, weight=class_weights)

            strat = params.get("balancing_strategy", "class_weights")
            if strat == "oversampling":
                overs_idx = build_oversampled_train_index(
                    labels,
                    train_mask,
                    oversample_ratio=float(
                        params.get("oversample_ratio", 1.0)
                    ),
                )
                # Loss auf oversample-Index
                loss = F.nll_loss(logp[overs_idx], labels[overs_idx])
            elif strat == "class_weights":
                labels_train = labels[train_mask]
                class_weights = apply_gcn_balancing_strategy(
                    labels_train, params
                )  # unverändert
                loss = F.nll_loss(
                    logp[train_mask], labels_train, weight=class_weights
                )
            else:  # 'none'
                loss = F.nll_loss(logp[train_mask], labels[train_mask])

            if not th.isfinite(loss):
                print(
                    f"Non-finite loss detected at epoch {epoch}: {loss.item()}. Skipping step / early-break."
                )
                no_impr += 1
                if no_impr >= patience:
                    print(
                        f"Early stopping due to instability at epoch {epoch}"
                    )
                    break
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            if epoch >= 3:
                dur.append(time.time() - t0)

            net.eval()
            with th.no_grad():
                v_logits = net(g, x)[val_mask]
                v_logits = th.nan_to_num(
                    v_logits, nan=0.0, posinf=1e6, neginf=-1e6
                )
                v_logp = F.log_softmax(v_logits, 1)
                v_loss = F.nll_loss(v_logp, labels[val_mask]).item()
            scheduler.step(v_loss)

            if v_loss < best_val and math.isfinite(v_loss):
                best_val = v_loss
                no_impr = 0
                best_state = net.state_dict().copy()
                print(f"New best validation loss: {v_loss:.4f}")
            else:
                no_impr += 1
                print(f"No improvement ({no_impr}/{patience})")
            if epoch % 10 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:05d} | Train Loss {loss.item():.4f} | Val Loss {v_loss:.4f} | LR {cur_lr:.6f} | Time(s) {np.mean(dur) if dur else 0.0:.4f}"
                )
            if no_impr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        else:
            print(
                "No best_state captured (instability). Marking trial as FAIL."
            )
            return {"status": STATUS_FAIL}

        roc_auc, pr_auc, val_pack = evaluate(
            net, g, features.float(), labels, val_mask, "Validation (argmax)"
        )

        for k in test_metrics.keys():
            test_metrics[k].append(val_pack.get(k, np.nan))

        return {"loss": val_pack["log_loss"], "status": STATUS_OK}

    trials = Trials()
    print(
        "Starting hyperparameter optimization with scaling & safety fixes..."
    )
    best = fmin(
        fn=objective,
        space=params_space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=number_of_hyperopt,
        rstate=np.random.default_rng(
            SEED
        ),  # <- wichtig für Reproduzierbarkeit
    )

    # Train best & Evaluate (mit Threshold-Tuning + Best-Epoch-Transfer)
    print("\nEvaluating best model on validation set...")
    best_params = space_eval(params_space, best)

    try:
        g, features, labels, train_mask, test_mask, val_mask, df_edge = (
            robust_load_elliptic_dataset(
                best_params, artifact_dir=artifact_dir
            )
        )
    except Exception as e:
        print(f"Finale Datenladung fehlgeschlagen: {e}")
        raise SystemExit(1)

    analyze_graph_data(g, features, labels, train_mask, test_mask, val_mask)

    in_feats = features.size(1)
    if best_params["number_of_layers"] == 1:
        net = Net1(best_params, in_feats)
    elif best_params["number_of_layers"] == 2:
        net = Net2(best_params, in_feats)
    elif best_params["number_of_layers"] == 3:
        net = Net3(best_params, in_feats)
    else:
        net = Net4(best_params, in_feats)
    net = net.to(device)
    print(f"Best model architecture: {net}")

    lr = float(best_params.get("learning_rate", 1e-2))
    if best_params["optimizer"] == "SGD":
        optimizer = th.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = getattr(th.optim, str(best_params["optimizer"]))(
            net.parameters(), lr=lr
        )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    labels_train = labels[train_mask]
    class_weights = apply_gcn_balancing_strategy(labels_train, best_params)

    best_val = float("inf")
    patience = 10
    no_impr = 0
    best_state = None
    dur = []
    best_epoch = 0
    for epoch in range(number_of_epochs):
        if epoch >= 3:
            t0 = time.time()
        net.train()
        x = features.float()
        logits = net(g, x)
        logits = th.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        logp = F.log_softmax(logits, 1)
        # loss = F.nll_loss(logp[train_mask], labels_train, weight=class_weights)

        strat = best_params.get("balancing_strategy", "class_weights")
        if strat == "oversampling":
            overs_idx = build_oversampled_train_index(
                labels,
                train_mask,
                oversample_ratio=float(
                    best_params.get("oversample_ratio", 1.0)
                ),
            )
            # Loss auf oversample-Index
            loss = F.nll_loss(logp[overs_idx], labels[overs_idx])
        elif strat == "class_weights":
            labels_train = labels[train_mask]
            class_weights = apply_gcn_balancing_strategy(
                labels_train, best_params
            )  # unverändert
            loss = F.nll_loss(
                logp[train_mask], labels_train, weight=class_weights
            )
        else:  # 'none'
            loss = F.nll_loss(logp[train_mask], labels[train_mask])

        if not th.isfinite(loss):
            print(
                f"Non-finite loss detected at epoch {epoch}: {loss.item()}. Skipping step / early-break."
            )
            no_impr += 1
            if no_impr >= patience:
                print(f" Early stopping due to instability at epoch {epoch}")
                break
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)
        net.eval()
        with th.no_grad():
            v_logits = net(g, x)[val_mask]
            v_logits = th.nan_to_num(
                v_logits, nan=0.0, posinf=1e6, neginf=-1e6
            )
            v_logp = F.log_softmax(v_logits, 1)
            v_loss = F.nll_loss(v_logp, labels[val_mask]).item()
        scheduler.step(v_loss)
        if v_loss < best_val and math.isfinite(v_loss):
            best_val = v_loss
            no_impr = 0
            best_state = net.state_dict().copy()
            best_epoch = epoch
        else:
            no_impr += 1
        if epoch % 10 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:05d} | Train Loss {loss.item():.4f} | Val Loss {v_loss:.4f} | LR {cur_lr:.6f} | Time(s) {np.mean(dur) if dur else 0.0:.4f}"
            )
        if no_impr >= patience:
            print(f" Early stopping at epoch {epoch}")
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    print("\n=== Final Validation Evaluation (argmax) ===")
    roc_auc_val, pr_auc_val, best_val_pack = evaluate(
        net, g, features.float(), labels, val_mask, "Validation"
    )

    # Threshold-Tuning
    with th.no_grad():
        proba_val = F.softmax(
            th.nan_to_num(
                net(g, features.float())[val_mask],
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            ),
            1,
        )[:, 0]
        proba_val = (
            th.nan_to_num(proba_val, nan=0.0, posinf=1.0, neginf=0.0)
            .cpu()
            .numpy()
        )
    y_val_np = labels[val_mask].cpu().numpy()
    tau_star, score_star = tune_threshold(
        y_val_np, proba_val, metric=THRESH_METRIC
    )
    print(
        f"\nTuned validation threshold τ* = {tau_star:.3f} (metric={THRESH_METRIC}, score={score_star:.4f})"
    )

    print("\n=== Final Validation Evaluation (tuned threshold) ===")
    _, _, best_val_pack_tuned = evaluate(
        net,
        g,
        features.float(),
        labels,
        val_mask,
        "Validation (tuned)",
        tuned_threshold=tau_star,
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    feat_imp = calculate_gcn_feature_importance(
        net, g, features, labels, val_mask, n_repeats=3
    )
    feat_names = [f"feature_{i}" for i in range(features.shape[1])]
    df_perm = (
        pd.DataFrame({"feature": feat_names, "perm_importance": feat_imp})
        .sort_values("perm_importance", ascending=False)
        .reset_index(drop=True)
    )
    df_perm["perm_rank"] = np.arange(1, len(df_perm) + 1)
    print("\n=== Feature Importance Ranking (Top 15) ===")
    print(df_perm.head(15))

    # Final retrain on train+val and test on untouched test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 60)
    combined_mask = train_mask | val_mask

    if best_params["number_of_layers"] == 1:
        net_final = Net1(best_params, in_feats)
    elif best_params["number_of_layers"] == 2:
        net_final = Net2(best_params, in_feats)
    elif best_params["number_of_layers"] == 3:
        net_final = Net3(best_params, in_feats)
    else:
        net_final = Net4(best_params, in_feats)
    net_final = net_final.to(device)

    if best_params["optimizer"] == "SGD":
        optimizer_final = th.optim.SGD(
            net_final.parameters(), lr=lr, momentum=0.9
        )
    else:
        optimizer_final = getattr(th.optim, str(best_params["optimizer"]))(
            net_final.parameters(), lr=lr
        )

    # Final training setup (respect chosen balancing strategy)
    strat = best_params.get("balancing_strategy", "class_weights")
    overs_r = float(best_params.get("oversample_ratio", 1.0))
    w_comb = None

    if strat == "oversampling":
        # Oversampling => keine Class Weights, stattdessen je Epoch neue Oversample-Indizes
        w_comb = None
    elif strat == "class_weights":
        labels_combined = labels[combined_mask]
        cc = th.bincount(labels_combined).clamp(min=1).float()
        K = len(cc)
        w_comb = cc.sum() / (K * cc)
    else:  # 'none'
        w_comb = None

    print(
        f"Final training on {combined_mask.sum().item()} nodes (train + val)"
    )
    print(
        f"Final balancing: {strat}"
        + (
            f" (oversample_ratio={overs_r:.2f})"
            if strat == "oversampling"
            else ""
        )
    )
    if w_comb is not None:
        print(f"Final class weights: {w_comb}")
    print(
        f"Re-training for best_epoch={best_epoch} epochs to match early-stopped model"
    )

    for epoch in range(max(1, best_epoch + 1)):
        net_final.train()
        x = features.float()
        lg = net_final(g, x)
        lg = th.nan_to_num(lg, nan=0.0, posinf=1e6, neginf=-1e6)
        lp = F.log_softmax(lg, 1)

        if strat == "oversampling":
            # Re-sample pro Epoche, um Varianz zu erhöhen (max. 1:1 zur Majority)
            overs_idx_final = build_oversampled_train_index(
                labels, combined_mask, oversample_ratio=overs_r
            )
            ls = F.nll_loss(lp[overs_idx_final], labels[overs_idx_final])
        else:
            # class_weights oder none
            ls = F.nll_loss(
                lp[combined_mask], labels[combined_mask], weight=w_comb
            )

        if not th.isfinite(ls):
            print(
                f"Non-finite final-train loss at epoch {epoch}: {ls.item()}. Skipping step."
            )
            continue

        optimizer_final.zero_grad()
        ls.backward()
        nn.utils.clip_grad_norm_(net_final.parameters(), max_norm=5.0)
        optimizer_final.step()

        if epoch % 10 == 0:
            print(f"Final training epoch {epoch:05d} | Loss {ls.item():.4f}")

    print("\n=== FINAL TEST EVALUATION (argmax) ===")
    roc_auc_test, pr_auc_test, final_test_results = evaluate(
        net_final, g, features.float(), labels, test_mask, "Final Test"
    )

    print("\n=== FINAL TEST EVALUATION (tuned threshold) ===")
    _, _, final_test_results_tuned = evaluate(
        net_final,
        g,
        features.float(),
        labels,
        test_mask,
        "Final Test (tuned)",
        tuned_threshold=tau_star,
    )

    # Save JSON + Pickle
    all_losses = [loss for loss in trials.losses() if loss is not None]

    results_summary = {
        "run_info": {
            "model": "GCN",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": SEED,
            "device": str(device),
            "training_epochs": number_of_epochs,
            "hyperopt_evals": number_of_hyperopt,
            "add_unlabeled_nodes": add_unlabeled_nodes,
            "random_split": random_split,
            "environment": "GCN_with_artifact_features",
            "evaluation_scheme": "train_val_test_protocol",
            "feature_selection": "none",
            "balancing_strategy": best_params.get(
                "balancing_strategy", "class_weights"
            ),
            "early_stopping": True,
            "learning_rate_scheduler": True,
            "threshold_metric": THRESH_METRIC,
            "tuned_threshold": float(tau_star),
            "best_epoch": int(best_epoch),
        },
        "version_info": {
            "dgl_version": dgl.__version__,
            "torch_version": th.__version__,
            "python_version": os.sys.version,
        },
        "data_info": {
            "feature_count": int(features.size()[1]),
            "node_count": int(len(labels)),
            "edge_count_no_self": int(
                (g.number_of_edges() - g.number_of_nodes())
            ),
            "graph_edges_actual_with_self": int(g.number_of_edges()),
            "train_nodes": int(train_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
            "val_nodes": int(val_mask.sum().item()),
            "combined_train_val_nodes": int(
                (train_mask | val_mask).sum().item()
            ),
        },
        "best_params": _to_jsonable(_name_funcs_in_params(best_params)),
        "hyperopt_results": {
            "best_loss": float(
                min(all_losses) if all_losses else float("inf")
            ),
            "total_trials": len(trials.trials),
            "successful_trials": len(
                [
                    t
                    for t in trials.trials
                    if t["result"]["status"] == STATUS_OK
                ]
            ),
            "validation_metrics_summary": {
                "f1_fraud_mean": (
                    float(np.mean(test_metrics["f1_score"]))
                    if test_metrics["f1_score"]
                    else None
                ),
                "precision_fraud_mean": (
                    float(np.mean(test_metrics["precision"]))
                    if test_metrics["precision"]
                    else None
                ),
                "recall_fraud_mean": (
                    float(np.mean(test_metrics["recall"]))
                    if test_metrics["recall"]
                    else None
                ),
                "balanced_accuracy_mean": (
                    float(np.mean(test_metrics["balanced_accuracy"]))
                    if test_metrics["balanced_accuracy"]
                    else None
                ),
                "accuracy_mean": (
                    float(np.mean(test_metrics["accuracy"]))
                    if test_metrics["accuracy"]
                    else None
                ),
                "log_loss_mean": (
                    float(np.mean(test_metrics["log_loss"]))
                    if test_metrics["log_loss"]
                    else None
                ),
                "roc_auc_mean": (
                    float(np.mean(test_metrics["roc_auc"]))
                    if test_metrics["roc_auc"]
                    else None
                ),
                "pr_auc_mean": (
                    float(np.mean(test_metrics["pr_auc"]))
                    if test_metrics["pr_auc"]
                    else None
                ),
            },
        },
        "validation_metrics_argmax": {
            "f1_fraud": float(best_val_pack["f1_score"]),
            "precision_fraud": float(best_val_pack["precision"]),
            "recall_fraud": float(best_val_pack["recall"]),
            "balanced_accuracy": float(best_val_pack["balanced_accuracy"]),
            "log_loss": float(best_val_pack["log_loss"]),
            "accuracy": float(best_val_pack["accuracy"]),
            "roc_auc_fraud": float(best_val_pack["roc_auc"]),
            "pr_auc_fraud": float(best_val_pack["pr_auc"]),
            "pred_fraud": int(best_val_pack["indices_fraud"]),
            "pred_no_fraud": int(best_val_pack["indices_no_fraud"]),
            "actual_fraud": int(best_val_pack["labels_fraud"]),
            "actual_no_fraud": int(best_val_pack["labels_no_fraud"]),
            "confusion_matrix": best_val_pack["confusion_matrix"],
            "cm_c00": int(best_val_pack["cm_c00"]),
            "cm_c01": int(best_val_pack["cm_c01"]),
            "cm_c10": int(best_val_pack["cm_c10"]),
            "cm_c11": int(best_val_pack["cm_c11"]),
        },
        "validation_metrics_tuned": {
            "tuned_threshold": float(tau_star),
            "metric": THRESH_METRIC,
            "f1_fraud": float(best_val_pack_tuned["f1_score"]),
            "precision_fraud": float(best_val_pack_tuned["precision"]),
            "recall_fraud": float(best_val_pack_tuned["recall"]),
            "balanced_accuracy": float(
                best_val_pack_tuned["balanced_accuracy"]
            ),
            "log_loss": float(best_val_pack_tuned["log_loss"]),
            "accuracy": float(best_val_pack_tuned["accuracy"]),
            "roc_auc_fraud": float(best_val_pack_tuned["roc_auc"]),
            "pr_auc_fraud": float(best_val_pack_tuned["pr_auc"]),
            "pred_fraud": int(best_val_pack_tuned["indices_fraud"]),
            "pred_no_fraud": int(best_val_pack_tuned["indices_no_fraud"]),
            "actual_fraud": int(best_val_pack_tuned["labels_fraud"]),
            "actual_no_fraud": int(best_val_pack_tuned["labels_no_fraud"]),
            "confusion_matrix": best_val_pack_tuned["confusion_matrix"],
            "cm_c00": int(best_val_pack_tuned["cm_c00"]),
            "cm_c01": int(best_val_pack_tuned["cm_c01"]),
            "cm_c10": int(best_val_pack_tuned["cm_c10"]),
            "cm_c11": int(best_val_pack_tuned["cm_c11"]),
        },
        "final_test_metrics_argmax": {
            "f1_fraud": float(final_test_results["f1_score"]),
            "precision_fraud": float(final_test_results["precision"]),
            "recall_fraud": float(final_test_results["recall"]),
            "balanced_accuracy": float(
                final_test_results["balanced_accuracy"]
            ),
            "log_loss": float(final_test_results["log_loss"]),
            "accuracy": float(final_test_results["accuracy"]),
            "roc_auc_fraud": float(final_test_results["roc_auc"]),
            "pr_auc_fraud": float(final_test_results["pr_auc"]),
            "pred_fraud": int(final_test_results["indices_fraud"]),
            "pred_no_fraud": int(final_test_results["indices_no_fraud"]),
            "actual_fraud": int(final_test_results["labels_fraud"]),
            "actual_no_fraud": int(final_test_results["labels_no_fraud"]),
            "confusion_matrix": final_test_results["confusion_matrix"],
            "cm_c00": int(final_test_results["cm_c00"]),
            "cm_c01": int(final_test_results["cm_c01"]),
            "cm_c10": int(final_test_results["cm_c10"]),
            "cm_c11": int(final_test_results["cm_c11"]),
        },
        "final_test_metrics_tuned": {
            "tuned_threshold": float(tau_star),
            "metric": THRESH_METRIC,
            "f1_fraud": float(final_test_results_tuned["f1_score"]),
            "precision_fraud": float(final_test_results_tuned["precision"]),
            "recall_fraud": float(final_test_results_tuned["recall"]),
            "balanced_accuracy": float(
                final_test_results_tuned["balanced_accuracy"]
            ),
            "log_loss": float(final_test_results_tuned["log_loss"]),
            "accuracy": float(final_test_results_tuned["accuracy"]),
            "roc_auc_fraud": float(final_test_results_tuned["roc_auc"]),
            "pr_auc_fraud": float(final_test_results_tuned["pr_auc"]),
            "pred_fraud": int(final_test_results_tuned["indices_fraud"]),
            "pred_no_fraud": int(final_test_results_tuned["indices_no_fraud"]),
            "actual_fraud": int(final_test_results_tuned["labels_fraud"]),
            "actual_no_fraud": int(
                final_test_results_tuned["labels_no_fraud"]
            ),
            "confusion_matrix": final_test_results_tuned["confusion_matrix"],
            "cm_c00": int(final_test_results_tuned["cm_c00"]),
            "cm_c01": int(final_test_results_tuned["cm_c01"]),
            "cm_c10": int(final_test_results_tuned["cm_c10"]),
            "cm_c11": int(final_test_results_tuned["cm_c11"]),
        },
        "feature_importance": {
            "permutation_importance": {
                "scores": _to_jsonable(feat_imp),
                "calculation_method": "gcn_permutation_importance",
                "sorted_features": df_perm["feature"].tolist(),
                "sorted_importances": df_perm["perm_importance"].tolist(),
                "n_repeats": 3,
            }
        },
    }

    results_summary = _nan_to_none(results_summary)

    RESULTS_DIR = os.path.join(".", "Results_JSON")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(
        RESULTS_DIR, f"elliptic_GCN_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {json_path}")

    # Zusatzartefakte analog NB-Skript
    out_base = Path(results_dir)
    artifact_tag = artifact_dir_name

    # 1) Feature-Importance als Parquet speichern
    df_perm_path = out_base / f"elliptic_GCN_{artifact_tag}__perm.parquet"
    df_perm.to_parquet(df_perm_path, index=False)

    # 2) Hyperopt-Trials als joblib
    trials_path = out_base / f"elliptic_GCN_{artifact_tag}__trials.joblib"
    joblib.dump(trials, trials_path, compress=3)

    # 3) Bundle mit wichtigen Reload-Infos
    bundle = {
        "best_params": best_params,
        "tuned_threshold": float(tau_star),
        "best_epoch": int(best_epoch),
        "artifact_dir": artifact_dir_name,
        "feature_importance": {
            "sorted_features": df_perm["feature"].tolist(),
            "sorted_importances": df_perm["perm_importance"].tolist(),
            "n_repeats": 3,
        },
        # optional: falls du das Modell später reloaden möchtest:
        "model_state_dict": net_final.state_dict(),
    }
    bundle_path = out_base / f"elliptic_GCN_{artifact_tag}__bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)

    print("Saved extras:")
    print(f"- {df_perm_path}")
    print(f"- {trials_path}")
    print(f"- {bundle_path}")

    return json_path


def main():
    ap = argparse.ArgumentParser(
        description="GCN Elliptic – Single & Batch Runner"
    )
    ap.add_argument(
        "--artifact",
        help="Name eines Artefakt-Unterordners unter artifacts/elliptic",
    )
    ap.add_argument(
        "--folder",
        help="Ordner mit Artefakt-Unterordnern (Default: ./artifacts/elliptic)",
    )
    ap.add_argument("--pattern", help="Substring-Filter für Artefakt-Namen")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Überspringt Artefakte mit vorhandener JSON",
    )
    args = ap.parse_args()
    # reproducability
    args.jobs = 1

    if args.artifact or ARTIFACT_DIR_NAME_DEFAULT:
        name = args.artifact or ARTIFACT_DIR_NAME_DEFAULT
        run_for_artifact(name)
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
            out = os.path.join(RESULTS_DIR, f"elliptic_GCN_{name}.json")
            if os.path.exists(out):
                continue
        candidates.append(name)

    if not candidates:
        print("Keine passenden Artefakte gefunden.")
        return

    print(f"Starte Runs für {len(candidates)} Artefakte…")
    for name in candidates:
        try:
            path = run_for_artifact(name)
            print(f"Fertig: {name} -> {path}")
        except Exception as e:
            print(f"Fehler bei {name}: {e}")


if __name__ == "__main__":
    main()
