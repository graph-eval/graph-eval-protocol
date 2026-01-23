# -*- coding: utf-8 -*-
"""
Graph Attention Network

- Validation threshold tuning τ∗ (default metric: balanced accuracy), with reporting at both 0.5 and τ∗
- Best-epoch transfer: final retraining on (TRAIN+VAL) exactly up to the best epoch from the validation phase
  → prevents over-/underfitting and avoids AUC / balanced-accuracy drops on the final test set
- Cleaner JSON outputs and extras including τ∗, best_epoch, and the threshold metric
- Batch execution over multiple artifact directories (analogous to GCN)
- Additional artifacts: trials, feature importance, and bundle (joblib/parquet)
"""

import os

# reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import time
import math
import random
import numpy as np
import pandas as pd
import joblib
import argparse

from Utilities.common_utils import (
    has_required_files,
    _read_parquet_required,
    _to_jsonable,
)
from pathlib import Path
from datetime import datetime

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

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
    log_loss as skl_log_loss,
)


# Reproducibility Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
# if th.cuda.is_available():
#    th.cuda.manual_seed_all(SEED)
dgl.random.seed(SEED)

# Strikte Deterministik erzwingen (wirft Fehler bei nichtdeterministischen Ops)
th.use_deterministic_algorithms(True)

# Device configuration
# device = th.device("cuda" if th.cuda.is_available() else "cpu")
# Für Reproduzierbarkeit: CPU erzwingen
device = th.device("cpu")
print(f"Using device: {device}")

# th.backends.cudnn.deterministic = True
# th.backends.cudnn.benchmark = False

# Config parameters
# Optional Default für Single-Run; kann leer bleiben (None)
ARTIFACT_DIR_NAME_DEFAULT = "20251107T111140Z_base93+noInd+0Spectral+0Graphwave+0Role2Vec+0Prox_temporal_1-35_36-42_43-49_28ce25bc"

ARTIFACT_ROOT = os.path.join(".", "artifacts", "elliptic")
RESULTS_DIR = os.path.join(".", "Results_JSON", "GAT")
os.makedirs(RESULTS_DIR, exist_ok=True)

number_of_epochs = 200  # budget (ES enabled)
number_of_hyperopt = 40
add_unlabeled_nodes = False
random_split = False

# Threshold-Tuning Zielmetrik
THRESH_METRIC = "balanced_accuracy"  # 'balanced_accuracy' | 'f1' | 'accuracy'


def _ensure_txid_column(df: pd.DataFrame) -> pd.DataFrame:
    if "txId" in df.columns:
        df["txId"] = df["txId"].astype(str)
        return df
    idx_name = (df.index.name or "").lower()
    if idx_name in {"txid", "tx_id", "txhash", "id", "node_id"}:
        df = df.reset_index().rename(columns={df.index.name: "txId"})
        df["txId"] = df["txId"].astype(str)
        return df
    for v in ["txid", "tx_id", "TXID", "TxId", "id", "node_id", "txhash"]:
        if v in df.columns:
            df = df.rename(columns={v: "txId"})
            df["txId"] = df["txId"].astype(str)
            return df
    df = df.reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": "txId"})
    elif df.columns[0] != "txId":
        df = df.rename(columns={df.columns[0]: "txId"})
    df["txId"] = df["txId"].astype(str)
    return df


# GAT model layers/nets
class GATLayer(nn.Module):
    """
    Hidden: concat heads
    Output: typischerweise 1 head, mean (concat=False)
    """

    def __init__(
        self, in_feats, out_feats, params, concat=True, use_dropout=True
    ):
        super().__init__()
        self.params = params
        self.num_heads = int(params.get("num_heads", 1))
        self.concat = bool(concat)

        drop_rate = (
            float(params.get("dropout_rate", 0.0)) if use_dropout else 0.0
        )
        self.post_drop = nn.Dropout(drop_rate)

        self.gat = GATConv(
            in_feats,
            out_feats,
            num_heads=self.num_heads,
            feat_drop=float(params.get("feat_dropout", 0.0)),
            attn_drop=float(params.get("attn_dropout", 0.0)),
            negative_slope=float(params.get("negative_slope", 0.2)),
            residual=bool(params.get("residual", True)),
            activation=None,
            allow_zero_in_degree=True,
        )

    def forward(self, g, x):
        h = self.gat(g, x)  # (N, heads, out_feats)
        if self.concat:
            h = h.flatten(1)  # (N, heads * out_feats)
        else:
            h = h.mean(dim=1)  # (N, out_feats)
        return self.post_drop(h)


class Net1(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params
        # Output-Layer: 1 Head, concat=False -> 2 Logits
        params_out = dict(params)
        params_out["num_heads"] = 1
        self.layer1 = GATLayer(
            in_feats, 2, params_out, concat=False, use_dropout=False
        )

    def forward(self, g, x, return_embeddings: bool = False):
        logits = self.layer1(g, x)  # (N, 2)
        if return_embeddings:
            # Embeddings = Logits (nur 1 Layer)
            return logits, logits
        return logits


class Net2(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params

        h2 = int(params["layer2_input_neuron"])
        heads = int(params["num_heads"])

        # Hidden-Layer: concat=True -> Output-Dim = h2 * heads
        self.layer1 = GATLayer(
            in_feats, h2, params, concat=True, use_dropout=True
        )
        h2_in = h2 * heads

        # Output-Layer: 1 Head, concat=False -> immer 2 Logits
        params_out = dict(params)
        params_out["num_heads"] = 1
        self.layer2 = GATLayer(
            h2_in, 2, params_out, concat=False, use_dropout=False
        )

    def forward(self, g, x, return_embeddings: bool = False):
        act = getattr(F, str(self.params["activation_function"]))
        h = act(self.layer1(g, x))  # Embeddings
        logits = self.layer2(g, h)
        if return_embeddings:
            return logits, h
        return logits


class Net3(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params

        h2 = int(params["layer2_input_neuron"])
        h3 = int(params["layer3_input_neuron"])
        heads = int(params["num_heads"])

        # 1. Hidden-Layer
        self.layer1 = GATLayer(
            in_feats, h2, params, concat=True, use_dropout=True
        )
        h2_in = h2 * heads

        # 2. Hidden-Layer
        self.layer2 = GATLayer(
            h2_in, h3, params, concat=True, use_dropout=True
        )
        h3_in = h3 * heads

        # Output-Layer
        params_out = dict(params)
        params_out["num_heads"] = 1
        self.layer3 = GATLayer(
            h3_in, 2, params_out, concat=False, use_dropout=False
        )

    def forward(self, g, x, return_embeddings: bool = False):
        act = getattr(F, str(self.params["activation_function"]))
        h1 = act(self.layer1(g, x))
        h2 = act(self.layer2(g, h1))  # Embeddings
        logits = self.layer3(g, h2)
        if return_embeddings:
            return logits, h2
        return logits


class Net4(nn.Module):
    def __init__(self, params, in_feats):
        super().__init__()
        self.params = params

        h2 = int(params["layer2_input_neuron"])
        h3 = int(params["layer3_input_neuron"])
        h4 = int(params["layer4_input_neuron"])
        heads = int(params["num_heads"])

        # 1. Hidden-Layer
        self.layer1 = GATLayer(
            in_feats, h2, params, concat=True, use_dropout=True
        )
        h2_in = h2 * heads

        # 2. Hidden-Layer
        self.layer2 = GATLayer(
            h2_in, h3, params, concat=True, use_dropout=True
        )
        h3_in = h3 * heads

        # 3. Hidden-Layer
        self.layer3 = GATLayer(
            h3_in, h4, params, concat=True, use_dropout=True
        )
        h4_in = h4 * heads

        # Output-Layer
        params_out = dict(params)
        params_out["num_heads"] = 1
        self.layer4 = GATLayer(
            h4_in, 2, params_out, concat=False, use_dropout=False
        )

    def forward(self, g, x, return_embeddings: bool = False):
        act = getattr(F, str(self.params["activation_function"]))
        h1 = act(self.layer1(g, x))
        h2 = act(self.layer2(g, h1))
        h3 = act(self.layer3(g, h2))  # Embeddings
        logits = self.layer4(g, h3)
        if return_embeddings:
            return logits, h3
        return logits


# Data loading (artifacts)
def robust_load_elliptic_dataset(
    params, artifact_dir: str, drop_rate=None, variant=None, seed: int = 42
):
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

        X_tr["_split"] = "train"
        X_te["_split"] = "test"
        X_va["_split"] = "val"
        DF = pd.concat([X_tr, X_te, X_va], axis=0)
        y_all = pd.concat([y_tr, y_te, y_va], axis=0).reindex(DF.index)

        valid_mask = y_all.isin([0, 1])
        split = DF.pop("_split")
        train_mask_np = (split == "train").to_numpy() & valid_mask.to_numpy()
        test_mask_np = (split == "test").to_numpy() & valid_mask.to_numpy()
        val_mask_np = (split == "val").to_numpy() & valid_mask.to_numpy()

        y_all_clean = y_all.copy()
        y_all_clean[~valid_mask] = 0
        y_all_clean = y_all_clean.astype(int)

        suffix, edge_path = get_edge_suffix_and_path(
            drop_rate=drop_rate, variant=variant
        )

        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"Edgelist fehlt unter {edge_path}")
        edges = pd.read_csv(edge_path, usecols=["txId1", "txId2"], dtype=str)

        idx_map = {tx: i for i, tx in enumerate(DF.index)}
        mask_both = edges["txId1"].isin(idx_map) & edges["txId2"].isin(idx_map)
        E = edges.loc[mask_both].copy()
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

        # Train-based scaling
        train_idx = X_tr.index
        mu = FEATURES.loc[train_idx].mean()
        sd = FEATURES.loc[train_idx].std(ddof=0).replace(0, 1.0)
        FEATURES = (FEATURES - mu) / sd
        FEATURES = FEATURES.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        print("\n=== NaN ANALYSIS ===")
        print(
            f"NaN in X_train: {pd.isna(FEATURES.loc[X_tr.index]).to_numpy().sum()}"
        )
        print(
            f"NaN in X_val:   {pd.isna(FEATURES.loc[X_va.index]).to_numpy().sum()}"
        )
        print(
            f"NaN in X_test:  {pd.isna(FEATURES.loc[X_te.index]).to_numpy().sum()}"
        )
        print(f"NaN in y_train: {y_tr.isna().sum()}")
        print(f"NaN in y_val:   {y_va.isna().sum()}")
        print(f"NaN in y_test:  {y_te.isna().sum()}")

        feats = th.tensor(FEATURES.values, dtype=th.float32)

        labels = th.tensor(
            y_all_clean.values.astype(int), dtype=th.long, device=device
        )
        train_mask = th.from_numpy(train_mask_np).to(th.bool).to(device)
        test_mask = th.from_numpy(test_mask_np).to(th.bool).to(device)
        val_mask = th.from_numpy(val_mask_np).to(th.bool).to(device)

        g = g.to(device)
        feats = feats.to(device)

        print(f"Labeled train nodes: {int(train_mask.sum().item())}")
        print(f"Labeled val nodes:   {int(val_mask.sum().item())}")
        print(f"Labeled test nodes:  {int(test_mask.sum().item())}")

        df_edge = pd.DataFrame({"node_id1": src, "node_id2": dst})

        # Node-IDs in der Reihenfolge der Feature-Matrix (entspricht DF.index → txId)
        tx_ids = DF.index.astype(str).to_numpy()

        return (
            g,
            feats,
            labels,
            train_mask,
            test_mask,
            val_mask,
            df_edge,
            tx_ids,
        )

    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        raise RuntimeError(f"Datenladung fehlgeschlagen: {e}")


def export_gat_embeddings_to_parquet(
    model,
    g,
    features,
    tx_ids,
    artifact_dir_name: str,
    drop_rate=None,
    variant=None,
    seed: int = 42,
):
    """
    Exportiert Node-Embeddings eines trainierten GAT als Parquet-Datei,
    analog zu GCN / Graphwave / node2vec.

    Die Datei landet (global) unter ./artifacts/gat_embeddings.parquet
    und hat eine txId-Spalte wie die anderen Artefakte.
    """
    model.eval()
    with th.no_grad():
        logits, z = model(g, features, return_embeddings=True)

    z_np = z.detach().cpu().numpy()

    # tx_ids als numpy-Array
    if isinstance(tx_ids, th.Tensor):
        tx_ids_np = tx_ids.detach().cpu().numpy().astype(str)
    else:
        tx_ids_np = np.asarray(tx_ids, dtype=str)

    dim = z_np.shape[1]
    emb_cols = [f"gat_emb_{i}" for i in range(dim)]

    df = pd.DataFrame(z_np, columns=emb_cols)
    df.insert(0, "txId", tx_ids_np)

    # Globaler Artefakt-Pfad
    global_artifact_dir = os.path.join(".", "artifacts")
    os.makedirs(global_artifact_dir, exist_ok=True)

    suffix, _ = get_edge_suffix_and_path(drop_rate, variant)

    fname = f"gat_embeddings{suffix}.parquet"
    out_path = os.path.join(global_artifact_dir, fname)

    df.to_parquet(out_path, index=False)
    print(f"GAT-Embeddings als Parquet gespeichert: {out_path}")


# Threshold tuning helpers
def _metric_from_preds(y_true, y_pred, metric="balanced_accuracy"):
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    return balanced_accuracy_score(y_true, y_pred)


def find_best_threshold(
    y_true, proba_pos_class0, metric="balanced_accuracy", grid_points=1001
):
    ts = np.linspace(0.0, 1.0, grid_points)
    best_t, best_s = 0.5, -1.0
    for t in ts:
        y_pred = np.where(proba_pos_class0 >= t, 0, 1)
        s = _metric_from_preds(y_true, y_pred, metric=metric)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)


# Diagnostics / helpers
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
    e = g.number_of_edges()
    dens = e / (n * max(1, (n - 1)))
    print(f"\nGraph nodes: {n}")
    print(f"Graph edges: {e}")
    print(f"Graph density: {dens:.6f}")

    deg = (g.in_degrees() + g.out_degrees()).cpu()
    isolated = int((deg <= 2).sum().item())
    print(
        f"Isolated-or-nearly-isolated nodes (<=2 deg incl. self/rev): {isolated}"
    )

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
    strategy = params.get("balancing_strategy", "class_weights")
    print(f"Applying balancing strategy: {strategy}")
    if strategy == "none":
        return None
    class_counts = th.bincount(labels_train).clamp(min=1).float()
    K = len(class_counts)
    return class_counts.sum() / (K * class_counts)


def calculate_gnn_feature_importance(
    model, g, features, labels, mask, n_repeats=3
):
    print("Calculating GAT feature importance via permutation...")
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        base_pred = th.argmax(logits[mask], dim=1)
        base_acc = (base_pred == labels[mask]).float().mean().item()

    importances = th.zeros(features.shape[1], device=device)
    for j in range(features.shape[1]):
        diffs = []
        for _ in range(n_repeats):
            feat_perm = features.clone()
            perm_idx = th.randperm(features.shape[0], device=device)
            feat_perm[:, j] = features[perm_idx, j]
            with th.no_grad():
                perm_logits = model(g, feat_perm)
                perm_pred = th.argmax(perm_logits[mask], dim=1)
                perm_acc = (perm_pred == labels[mask]).float().mean().item()
            diffs.append(base_acc - perm_acc)
        importances[j] = th.tensor(diffs, device=device).mean()
    return importances.detach().cpu().numpy()


# Evaluation (threshold-aware)
def evaluate(model, g, features, labels, mask, split_name="Test", thresh=None):
    model.eval()
    with th.no_grad():
        logits_full = model(g, features)
        logits = logits_full[mask]
        proba = F.softmax(logits, 1)
        labels_split = labels[mask]

    y_true = labels_split.detach().cpu().numpy()
    y_proba = proba.detach().cpu().numpy()
    used_thresh = 0.5

    if thresh is None:
        y_pred = y_proba.argmax(axis=1)
    else:
        used_thresh = float(thresh)
        y_pred = np.where(y_proba[:, 0] >= used_thresh, 0, 1)

    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    prc = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    try:
        nll = skl_log_loss(
            (y_true == 0).astype(int), y_proba[:, 0], labels=[0, 1]
        )
    except Exception:
        nll = float("nan")

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
    TP = int(cm[0, 0])
    FN = int(cm[0, 1])
    FP = int(cm[1, 0])
    TN = int(cm[1, 1])

    pred_fraud = int((y_pred == 0).sum())
    pred_nofraud = int((y_pred == 1).sum())
    lbl_fraud = int((y_true == 0).sum())
    lbl_nofraud = int((y_true == 1).sum())

    print(
        f"\nDetailed {split_name} Classification Report (threshold={used_thresh:.3f}):"
    )
    print(
        classification_report(
            y_true, y_pred, target_names=["Fraud", "No Fraud"], zero_division=0
        )
    )
    print(f"Balanced Accuracy: {bal:.4f}")

    print(f"########## {split_name} ##########")
    print(f"Pred Fraud = {pred_fraud}")
    print(f"Pred No fraud = {pred_nofraud}")
    print(f"Label Fraud = {lbl_fraud}")
    print(f"Label No fraud = {lbl_nofraud}")
    print(f"F1-Score = {f1:.4f}")
    print(f"precision = {prc:.4f}")
    print(f"recall = {rec:.4f}")
    print(f"acc_Score = {acc:.4f}")
    print(f"Log_Loss = {nll:.4f}")
    print(f"ROC-AUC = {roc:.4f}")
    print(f"PR-AUC = {pr:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("               Predicted")
    print("               Fraud    No Fraud")
    print(f"Actual Fraud   {TP:6d}   {FN:6d}")
    print(f"Actual NoFraud {FP:6d}   {TN:6d}")

    return (
        roc,
        pr,
        {
            "indices_fraud": pred_fraud,
            "indices_no_fraud": pred_nofraud,
            "labels_fraud": lbl_fraud,
            "labels_no_fraud": lbl_nofraud,
            "f1_score": f1,
            "precision": prc,
            "recall": rec,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "log_loss": float(nll),
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "confusion_matrix": cm.tolist(),
            "true_negatives": TN,
            "false_positives": FP,
            "false_negatives": FN,
            "true_positives": TP,
            "threshold": used_thresh,
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
    return {k: v for k, v in params.items()}


# Hyperopt parameter space
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
    "activation_function": hp.choice(
        "activation_function", ["relu", "elu", "gelu", "leaky_relu", "silu"]
    ),
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
        "balancing_strategy", ["class_weights", "none"]
    ),
    "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.3),
    "learning_rate": hp.loguniform(
        "learning_rate", np.log(1e-4), np.log(5e-2)
    ),
    "num_heads": hp.choice("num_heads", [1, 2, 4, 8]),
    "negative_slope": hp.uniform("negative_slope", 0.01, 0.3),
    "residual": hp.choice("residual", [True, False]),
    "attn_dropout": hp.uniform("attn_dropout", 0.0, 0.4),
    "feat_dropout": hp.uniform("feat_dropout", 0.0, 0.4),
}


# Runner for one artefakt
def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
    drop_rate=None,
    variant=None,
    seed: int = 42,
) -> str:

    artifact_dir = os.path.join(artifact_root, artifact_dir_name)

    # Accumulator (Validation während Hyperopt)
    test_metrics = {
        "indices_fraud": [],
        "indices_no_fraud": [],
        "labels_fraud": [],
        "labels_no_fraud": [],
        "f1_score": [],
        "precision": [],
        "recall": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "log_loss": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    # Objective
    def objective(params):
        print("Hyperparameters:", params)

        if params["number_of_layers"] >= 3 and int(
            params["layer3_input_neuron"]
        ) >= int(params["layer2_input_neuron"]):
            return {"status": STATUS_FAIL}
        if params["number_of_layers"] >= 4 and int(
            params["layer4_input_neuron"]
        ) >= int(params["layer3_input_neuron"]):
            return {"status": STATUS_FAIL}

        try:
            (
                g,
                features,
                labels,
                train_mask,
                test_mask,
                val_mask,
                df_edge,
                tx_ids,
            ) = robust_load_elliptic_dataset(
                params,
                artifact_dir=artifact_dir,
                drop_rate=drop_rate,
                variant=variant,
                seed=seed,
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
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        best_state = None
        dur = []

        for epoch in range(number_of_epochs):
            if epoch >= 3:
                t0 = time.time()

            net.train()
            x = features.float()
            logits = net(g, x)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(
                logp[train_mask], labels_train, weight=class_weights
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch >= 3:
                dur.append(time.time() - t0)

            net.eval()
            with th.no_grad():
                val_logits = net(g, x)[val_mask]
                val_logp = F.log_softmax(val_logits, 1)
                val_loss = F.nll_loss(val_logp, labels[val_mask]).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = net.state_dict().copy()
                print(f"New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{patience})")

            if epoch % 10 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:05d} | Train Loss {loss.item():.4f} | Val Loss {val_loss:.4f} | LR {cur_lr:.6f} | Time(s) {np.mean(dur) if dur else 0.0:.4f}"
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            net.load_state_dict(best_state)

        roc_auc, pr_auc, val_pack = evaluate(
            net,
            g,
            features.float(),
            labels,
            val_mask,
            "Validation @0.5",
            thresh=0.5,
        )

        test_metrics["indices_fraud"].append(val_pack["indices_fraud"])
        test_metrics["indices_no_fraud"].append(val_pack["indices_no_fraud"])
        test_metrics["labels_fraud"].append(val_pack["labels_fraud"])
        test_metrics["labels_no_fraud"].append(val_pack["labels_no_fraud"])
        test_metrics["f1_score"].append(val_pack["f1_score"])
        test_metrics["precision"].append(val_pack["precision"])
        test_metrics["recall"].append(val_pack["recall"])
        test_metrics["accuracy"].append(val_pack["accuracy"])
        test_metrics["log_loss"].append(val_pack["log_loss"])
        test_metrics["balanced_accuracy"].append(val_pack["balanced_accuracy"])
        test_metrics["roc_auc"].append(val_pack["roc_auc"])
        test_metrics["pr_auc"].append(val_pack["pr_auc"])

        return {"loss": val_pack["log_loss"], "status": STATUS_OK}

    trials = Trials()
    print(
        f"\n=== Starting hyperparameter optimization for GAT (artifact={artifact_dir_name}) ==="
    )
    best = fmin(
        fn=objective,
        space=params_space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=number_of_hyperopt,
        rstate=np.random.default_rng(SEED),  # <- fixiert die Hyperopt-RNG
    )

    # Train final model & evaluate
    print("\nEvaluating best model on validation set...")
    best_params = space_eval(params_space, best)

    try:
        (
            g,
            features,
            labels,
            train_mask,
            test_mask,
            val_mask,
            df_edge,
            tx_ids,
        ) = robust_load_elliptic_dataset(
            best_params,
            artifact_dir=artifact_dir,
            drop_rate=drop_rate,
            variant=variant,
            seed=seed,
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
    print(f"Training with class weights: {class_weights}")

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None
    best_epoch = 0
    dur = []

    for epoch in range(number_of_epochs):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        x = features.float()
        logits = net(g, x)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(
            logp[train_mask], labels[train_mask], weight=class_weights
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        net.eval()
        with th.no_grad():
            v_logits = net(g, x)[val_mask]
            v_logp = F.log_softmax(v_logits, 1)
            v_loss = F.nll_loss(v_logp, labels[val_mask]).item()
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            best_state = net.state_dict().copy()
            best_epoch = epoch
            print(f"New best validation loss: {v_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        if epoch % 10 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:05d} | Train Loss {loss.item():.4f} | Val Loss {v_loss:.4f} | LR {cur_lr:.6f} | Time(s) {np.mean(dur) if dur else 0.0:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    print("\n=== Final Validation Evaluation ===")
    roc_auc_val_05, pr_auc_val_05, best_val_at05 = evaluate(
        net,
        g,
        features.float(),
        labels,
        val_mask,
        "Validation @0.5",
        thresh=0.5,
    )

    # ===== Threshold tuning on validation =====
    with th.no_grad():
        val_logits = net(g, features.float())[val_mask]
        val_proba = F.softmax(val_logits, 1).cpu().numpy()
        val_y = labels[val_mask].cpu().numpy()
    tau_star, score_star = find_best_threshold(
        val_y, val_proba[:, 0], metric=THRESH_METRIC, grid_points=1001
    )
    print(
        f"\n✓ Best validation threshold τ* = {tau_star:.3f} (metric={THRESH_METRIC}, score={score_star:.4f})"
    )

    print("\n=== Validation re-evaluation at tuned threshold ===")
    roc_auc_val_tuned, pr_auc_val_tuned, best_val_tuned = evaluate(
        net,
        g,
        features.float(),
        labels,
        val_mask,
        "Validation @τ*",
        thresh=tau_star,
    )

    # Feature importance (permutation)
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS (Permutation)")
    print("=" * 50)
    feat_imp = calculate_gnn_feature_importance(
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

    # class weights (kombiniert)
    if best_params.get("balancing_strategy", "class_weights") == "none":
        w_comb = None
    else:
        labels_combined = labels[combined_mask]
        cc = th.bincount(labels_combined).clamp(min=1).float()
        K = len(cc)
        w_comb = cc.sum() / (K * cc)

    print(
        f"Final training on {combined_mask.sum().item()} nodes (train + val)"
    )
    print(f"Final class weights: {w_comb}")
    print(
        f"Re-training for best_epoch={best_epoch} epochs to match early-stopped model"
    )

    for epoch in range(max(1, best_epoch + 1)):  # exakt bis beste Epoche
        net_final.train()
        x = features.float()
        lg = net_final(g, x)
        lp = F.log_softmax(lg, 1)
        ls = F.nll_loss(
            lp[combined_mask], labels[combined_mask], weight=w_comb
        )
        optimizer_final.zero_grad()
        ls.backward()
        optimizer_final.step()
        if epoch % 10 == 0:
            print(f"Final training epoch {epoch:05d} | Loss {ls.item():.4f}")

    export_gat_embeddings_to_parquet(
        model=net_final,
        g=g,
        features=features.float(),
        tx_ids=tx_ids,
        artifact_dir_name=artifact_dir_name,
        drop_rate=drop_rate,
        variant=variant,
        seed=seed,
    )

    print("\n=== FINAL TEST EVALUATION (UNTOUCHED DATA) ===")
    roc_auc_test_05, pr_auc_test_05, final_test_results_05 = evaluate(
        net_final,
        g,
        features.float(),
        labels,
        test_mask,
        "Final Test @0.5",
        thresh=0.5,
    )
    roc_auc_test_tuned, pr_auc_test_tuned, final_test_results_tuned = evaluate(
        net_final,
        g,
        features.float(),
        labels,
        test_mask,
        "Final Test @τ*",
        thresh=tau_star,
    )

    # Save JSON + Extras
    all_losses = [loss for loss in trials.losses() if loss is not None]

    results_summary = {
        "run_info": {
            "model": "GAT",
            "artifact_dir": artifact_dir_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "random_state": SEED,
            "device": str(device),
            "training_epochs": number_of_epochs,
            "hyperopt_evals": number_of_hyperopt,
            "add_unlabeled_nodes": add_unlabeled_nodes,
            "random_split": random_split,
            "environment": "GAT_with_artifact_features",
            "evaluation_scheme": "train_val_test_protocol",
            "feature_selection": "none",
            "balancing_strategy": best_params.get(
                "balancing_strategy", "class_weights"
            ),
            "early_stopping": True,
            "learning_rate_scheduler": True,
            "gat_params": {
                "num_heads": int(best_params.get("num_heads", 1)),
                "negative_slope": float(
                    best_params.get("negative_slope", 0.2)
                ),
                "residual": bool(best_params.get("residual", True)),
                "attn_dropout": float(best_params.get("attn_dropout", 0.0)),
                "feat_dropout": float(best_params.get("feat_dropout", 0.0)),
                "dropout_rate": float(best_params.get("dropout_rate", 0.0)),
            },
            "learning_rate": float(best_params.get("learning_rate", 1e-2)),
            "threshold_tuning": {
                "enabled": True,
                "objective": THRESH_METRIC,
                "tau_star": float(tau_star),
            },
            "best_epoch": int(best_epoch),
            "edge_drop_rate": drop_rate,
            "edge_variant": variant,
            "edge_seed": seed,
        },
        "version_info": {
            "dgl_version": dgl.__version__,
            "torch_version": th.__version__,
            "python_version": os.sys.version,
        },
        "data_info": {
            "feature_count": int(features.size()[1]),
            "node_count": int(len(labels)),
            "edge_count": int(df_edge.shape[0]),
            "graph_edges_actual": int(g.number_of_edges()),
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
        "validation_metrics": {
            "at_0_5": {
                "f1_fraud": float(best_val_at05["f1_score"]),
                "precision_fraud": float(best_val_at05["precision"]),
                "recall_fraud": float(best_val_at05["recall"]),
                "balanced_accuracy": float(best_val_at05["balanced_accuracy"]),
                "log_loss": float(best_val_at05["log_loss"]),
                "accuracy": float(best_val_at05["accuracy"]),
                "roc_auc_fraud": float(best_val_at05["roc_auc"]),
                "pr_auc_fraud": float(best_val_at05["pr_auc"]),
                "pred_fraud": int(best_val_at05["indices_fraud"]),
                "pred_no_fraud": int(best_val_at05["indices_no_fraud"]),
                "actual_fraud": int(best_val_at05["labels_fraud"]),
                "actual_no_fraud": int(best_val_at05["labels_no_fraud"]),
                "confusion_matrix": best_val_at05["confusion_matrix"],
            },
            "at_tau_star": {
                "threshold": float(tau_star),
                "f1_fraud": float(best_val_tuned["f1_score"]),
                "precision_fraud": float(best_val_tuned["precision"]),
                "recall_fraud": float(best_val_tuned["recall"]),
                "balanced_accuracy": float(
                    best_val_tuned["balanced_accuracy"]
                ),
                "log_loss": float(best_val_tuned["log_loss"]),
                "accuracy": float(best_val_tuned["accuracy"]),
                "roc_auc_fraud": float(best_val_tuned["roc_auc"]),
                "pr_auc_fraud": float(best_val_tuned["pr_auc"]),
                "pred_fraud": int(best_val_tuned["indices_fraud"]),
                "pred_no_fraud": int(best_val_tuned["indices_no_fraud"]),
                "actual_fraud": int(best_val_tuned["labels_fraud"]),
                "actual_no_fraud": int(best_val_tuned["labels_no_fraud"]),
                "confusion_matrix": best_val_tuned["confusion_matrix"],
            },
        },
        "final_test_metrics": {
            "at_0_5": {
                "f1_fraud": float(final_test_results_05["f1_score"]),
                "precision_fraud": float(final_test_results_05["precision"]),
                "recall_fraud": float(final_test_results_05["recall"]),
                "balanced_accuracy": float(
                    final_test_results_05["balanced_accuracy"]
                ),
                "log_loss": float(final_test_results_05["log_loss"]),
                "accuracy": float(final_test_results_05["accuracy"]),
                "roc_auc_fraud": float(final_test_results_05["roc_auc"]),
                "pr_auc_fraud": float(final_test_results_05["pr_auc"]),
                "pred_fraud": int(final_test_results_05["indices_fraud"]),
                "pred_no_fraud": int(
                    final_test_results_05["indices_no_fraud"]
                ),
                "actual_fraud": int(final_test_results_05["labels_fraud"]),
                "actual_no_fraud": int(
                    final_test_results_05["labels_no_fraud"]
                ),
                "confusion_matrix": final_test_results_05["confusion_matrix"],
            },
            "at_tau_star": {
                "threshold": float(tau_star),
                "f1_fraud": float(final_test_results_tuned["f1_score"]),
                "precision_fraud": float(
                    final_test_results_tuned["precision"]
                ),
                "recall_fraud": float(final_test_results_tuned["recall"]),
                "balanced_accuracy": float(
                    final_test_results_tuned["balanced_accuracy"]
                ),
                "log_loss": float(final_test_results_tuned["log_loss"]),
                "accuracy": float(final_test_results_tuned["accuracy"]),
                "roc_auc_fraud": float(final_test_results_tuned["roc_auc"]),
                "pr_auc_fraud": float(final_test_results_tuned["pr_auc"]),
                "pred_fraud": int(final_test_results_tuned["indices_fraud"]),
                "pred_no_fraud": int(
                    final_test_results_tuned["indices_no_fraud"]
                ),
                "actual_fraud": int(final_test_results_tuned["labels_fraud"]),
                "actual_no_fraud": int(
                    final_test_results_tuned["labels_no_fraud"]
                ),
                "confusion_matrix": final_test_results_tuned[
                    "confusion_matrix"
                ],
            },
        },
        "feature_importance": {
            "permutation_importance": {
                "scores": _to_jsonable(feat_imp),
                "calculation_method": "gat_permutation_importance",
                "sorted_features": df_perm["feature"].tolist(),
                "sorted_importances": df_perm["perm_importance"].tolist(),
                "n_repeats": 3,
            }
        },
    }

    results_summary = _nan_to_none(results_summary)

    out_base = Path(results_dir)
    os.makedirs(out_base, exist_ok=True)
    suffix, _ = get_edge_suffix_and_path(drop_rate=drop_rate, variant=variant)
    json_path = out_base / f"elliptic_GAT_{artifact_dir_name}{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(
        "TRAINING COMPLETED SUCCESSFULLY! (GAT + threshold tuning + best-epoch transfer)"
    )
    print("=" * 60)
    print(f"Results saved to: {json_path}")

    # ===== Zusatzartefakte =====
    artifact_tag = artifact_dir_name

    # 1) Feature-Importance als Parquet
    df_perm_path = (
        out_base / f"elliptic_GAT_{artifact_tag}{suffix}__perm.parquet"
    )
    df_perm.to_parquet(df_perm_path, index=False)

    # 2) Hyperopt-Trials als joblib
    trials_path = (
        out_base / f"elliptic_GAT_{artifact_tag}{suffix}__trials.joblib"
    )
    joblib.dump(trials, trials_path, compress=3)

    # 3) Bundle mit Reload-Infos
    bundle = {
        "best_params": best_params,
        "tau_star": float(tau_star),
        "best_epoch": int(best_epoch),
        "artifact_dir": artifact_dir_name,
        "feature_importance": {
            "sorted_features": df_perm["feature"].tolist(),
            "sorted_importances": df_perm["perm_importance"].tolist(),
            "n_repeats": 3,
        },
        "model_state_dict": net_final.state_dict(),
    }
    bundle_path = (
        out_base / f"elliptic_GAT_{artifact_tag}{suffix}__bundle.joblib"
    )
    joblib.dump(bundle, bundle_path, compress=3)

    print("Saved extras:")
    print(f"- {df_perm_path}")
    print(f"- {trials_path}")
    print(f"- {bundle_path}")

    return str(json_path)


def get_edge_suffix_and_path(drop_rate=None, variant=None):
    if variant is not None:
        suffix = f"_{variant}"
        edge_path = os.path.join(
            ".", "EllipticDataSet", f"elliptic_txs_edgelist{suffix}.csv"
        )
    elif drop_rate is None:
        suffix = ""
        edge_path = os.path.join(
            ".", "EllipticDataSet", "elliptic_txs_edgelist.csv"
        )
    else:
        suffix = f"_{drop_rate}"
        edge_path = os.path.join(
            ".", "EllipticDataSet", f"elliptic_txs_edgelist{suffix}.csv"
        )
    return suffix, edge_path


def main():

    ap = argparse.ArgumentParser(
        description="GAT Elliptic – Single & Batch Runner"
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

    ap.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Anteil gelöschter Kanten in Prozent (z.B. 25, 50, 75). Wenn None: voller Graph.",
    )

    ap.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Targeted graph variant (e.g. 'pagerankTop10'). Overrides drop_rate.",
    )

    args = ap.parse_args()
    # Für strikte Reproduzierbarkeit: immer Single-Process
    args.jobs = 1

    # --- Single-Run nur wenn explizit oder kein Folder angegeben ---
    if args.artifact:
        run_for_artifact(
            args.artifact,
            drop_rate=args.drop_rate,
            variant=args.variant,
            seed=42,
        )
        return

    if ARTIFACT_DIR_NAME_DEFAULT and not args.folder and not args.pattern:
        run_for_artifact(
            ARTIFACT_DIR_NAME_DEFAULT,
            drop_rate=args.drop_rate,
            variant=args.variant,
            seed=42,
        )
        return

    # Batch-Modus
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
            suffix, _ = get_edge_suffix_and_path(
                drop_rate=args.drop_rate, variant=args.variant
            )
            out = os.path.join(
                RESULTS_DIR, f"elliptic_GAT_{name}{suffix}.json"
            )
            if os.path.exists(out):
                continue
        candidates.append(name)

    if not candidates:
        print("Keine passenden Artefakte gefunden.")
        return

    print(f"Starte Runs für {len(candidates)} Artefakte…")

    for name in candidates:
        try:
            path = run_for_artifact(
                name,
                artifact_root=root,
                drop_rate=args.drop_rate,
                variant=args.variant,
                seed=42,
            )
            print(f"Fertig: {name} -> {path}")
        except Exception as e:
            print(f"Fehler bei {name}: {e}")


if __name__ == "__main__":
    main()
