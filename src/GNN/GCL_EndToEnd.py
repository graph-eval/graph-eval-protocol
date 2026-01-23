# -*- coding: utf-8 -*-
"""
Graph Contrastive Learning (BGRL-style) on GCN-Backbone

- Self-supervised pretraining (BGRL-like, without negative samples)
- Backbone: 2-layer GCN (same GCNLayer logic as in GCN.py)
- Downstream: linear 2-class classifier with (light) fine-tuning of the encoder
- Evaluation: identical to GCN/GAT via evaluate(), including threshold tuning τ* (THRESH_METRIC)

Key points:
- Hyperopt searches over:
  * Pretraining parameters (hidden_dim, learning_rate_pretrain, edge_drop_prob)
  * Classifier parameters (LR, weight_decay, class_weight_base/exp)
  * Encoder learning-rate factor during fine-tuning (encoder_lr_factor)
- Balancing is enforced consistently with GCN/GAT:
    balancing_strategy = 'class_weights'
- Pretraining is:
  * run shorter per Hyperopt trial
  * re-run longer for the final model with the best parameters

Start as GCN/GAT:
    python GCL.py --artifact <artifact_name>
or batch mode:
    python GCL.py --folder artifacts/elliptic --pattern base93
"""

import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import math
import random
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from hyperopt import hp, Trials, fmin, tpe, space_eval

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import GCN
from GCN import (
    robust_load_elliptic_dataset,
    analyze_graph_data,
    apply_gcn_balancing_strategy,
    evaluate,
    _nan_to_none,
    tune_threshold,
    THRESH_METRIC,
)

device = GCN.device  # gleiche Device-Config wie in GCN.py verwenden

# Configuration
ARTIFACT_ROOT = os.path.join(ROOT_DIR, "artifacts", "elliptic")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results_JSON", "GCL")
os.makedirs(RESULTS_DIR, exist_ok=True)

STATUS_OK = "ok"
STATUS_FAIL = "fail"

# Reproducability
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
dgl.random.seed(SEED)
th.use_deterministic_algorithms(True)

# Epochen für finales Training
NUM_EPOCHS_PRETRAIN_FINAL = 200
NUM_EPOCHS_CLASSIFIER_FINAL = 200

# Etwas schlanker für Hyperopt
NUM_EPOCHS_PRETRAIN_HPO = 150
NUM_EPOCHS_CLASSIFIER_HPO = 200

# Anzahl Hyperopt-Trials
NUM_HYPEROPT_TRIALS = 40

# Gemeinsamer Hyperopt-Space (Pretraining + Classifier + Fine-Tuning)
hpo_space = {
    # Pretraining-Parameter
    # Dimension des versteckten Layers (und Embedding)
    "hidden_dim": hp.choice("hidden_dim", [32, 64, 128]),
    # Lernrate fürs Self-Supervised-Pretraining
    "learning_rate_pretrain": hp.loguniform(
        "learning_rate_pretrain", np.log(5e-4), np.log(5e-3)
    ),
    # Edge-Drop-Rate für die Graph-Augmentierung
    "edge_drop_prob": hp.uniform("edge_drop_prob", 0.1, 0.4),
    # Classifier-Parameter
    "learning_rate_classifier": hp.loguniform(
        "learning_rate_classifier", np.log(5e-4), np.log(5e-2)
    ),
    "weight_decay_classifier": hp.loguniform(
        "weight_decay_classifier", np.log(1e-6), np.log(1e-2)
    ),
    # Class-Weights (Balancing wird erzwungen)
    "class_weight_base": hp.loguniform(
        "class_weight_base", np.log(0.5), np.log(5.0)
    ),
    "class_weight_exp": hp.loguniform(
        "class_weight_exp", np.log(0.5), np.log(3.0)
    ),
    # Fine-Tuning des Encoders
    # Encoder-LR = encoder_lr_factor * learning_rate_classifier
    "encoder_lr_factor": hp.choice("encoder_lr_factor", [0.05, 0.1, 0.2, 0.3]),
}


# GCN-Backbone für GCL
class GCNLayer(nn.Module):
    """
    Minimaler Wrapper um GCN.GCNLayer, damit GCL unabhängig von Net1..4 ist.
    """

    def __init__(self, in_feats, out_feats, params, use_dropout=True):
        super().__init__()
        self.base = GCN.GCNLayer(
            in_feats, out_feats, params, use_dropout=use_dropout
        )

    def forward(self, g, x):
        return self.base(g, x)


class GCNEncoder(nn.Module):
    """
    Einfacher 2-Layer GCN-Encoder für GCL.
    """

    def __init__(self, in_feats, hidden_dim, emb_dim, params):
        super().__init__()
        self.params = params
        self.layer1 = GCNLayer(in_feats, hidden_dim, params, use_dropout=True)
        self.layer2 = GCNLayer(hidden_dim, emb_dim, params, use_dropout=False)

    def forward(self, g, x):
        act_name = str(self.params.get("activation_function", "relu"))
        act = getattr(F, act_name)
        x = act(self.layer1(g, x))
        x = self.layer2(g, x)
        return x  # (N, emb_dim)


class BGRLModel(nn.Module):
    """
    BGRL-ähnliches Modell:
    - online_encoder: f_θ
    - target_encoder: f_ξ (EMA von θ)
    - predictor: Linear-Map auf Embedding
    """

    def __init__(
        self, in_feats, hidden_dim, emb_dim, params, ema_momentum=0.99
    ):
        super().__init__()
        self.online_encoder = GCNEncoder(in_feats, hidden_dim, emb_dim, params)
        self.target_encoder = GCNEncoder(in_feats, hidden_dim, emb_dim, params)
        self.predictor = nn.Linear(emb_dim, emb_dim)
        self.ema_momentum = ema_momentum

        # Init target = online
        for p_o, p_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_t.data.copy_(p_o.data)
            p_t.requires_grad = False

    @th.no_grad()
    def _update_target(self):
        m = self.ema_momentum
        for p_o, p_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_t.data.mul_(m).add_(p_o.data * (1.0 - m))

    def forward(self, g, x):
        return self.online_encoder(g, x)


# Augmentierungen + Loss
def augment_graph(g, features, edge_drop_prob=0.2, feat_drop_prob=0.3):
    """
    Einfache Graph-Augmentierung:
    - stochastisches Edge-Drop
    - Feature-Dropout (elementweise)
    """
    device_local = features.device

    # Edge-Drop
    num_edges = g.number_of_edges()
    mask = th.rand(num_edges, device=device_local) > edge_drop_prob
    src, dst = g.edges()
    src = src[mask]
    dst = dst[mask]

    g_aug = dgl.graph(
        (src, dst), num_nodes=g.number_of_nodes(), device=device_local
    )
    g_aug = dgl.to_simple(dgl.add_reverse_edges(g_aug))
    g_aug = dgl.add_self_loop(g_aug)

    # Feature-Dropout
    feat_mask = (th.rand_like(features) > feat_drop_prob).float()
    x_aug = features * feat_mask

    return g_aug, x_aug


def byol_loss(p, z):
    """
    Cosine Similarity Loss: 2 - 2*cos
    p: Predictor-Ausgabe (Online)
    z: Target-Embedding (detach)
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


# Downstream-Classifier
class GCLClassifier(nn.Module):
    """
    Wrapper: Encoder + linearer 2-Klassen-Head.
    Optional: Encoder einfrieren oder fein-tunen.
    """

    def __init__(
        self,
        encoder: nn.Module,
        emb_dim: int,
        num_classes: int = 2,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(emb_dim, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, g, x):
        z = self.encoder(g, x)
        logits = self.fc(z)
        return logits


# Pretraining (Self-Supervised)
def pretrain_gcl(
    params, g, features, num_epochs: int = 200, log_interval: int = 10
):
    """
    Führt BGRL-artiges Pretraining auf dem kompletten Graphen durch.
    Gibt den trainierten Encoder zurück.
    """
    in_feats = features.size(1)

    hidden_dim = int(params.get("hidden_dim", 64))
    # Embedding-Dimension: wenn nicht explizit gesetzt, = hidden_dim
    emb_dim = int(params.get("emb_dim", hidden_dim))

    lr = float(params.get("learning_rate_pretrain", 1e-3))
    edge_drop = float(params.get("edge_drop_prob", 0.2))
    feat_drop = float(params.get("feat_drop_prob", 0.3))
    ema_momentum = float(params.get("ema_momentum", 0.99))

    # Parameter-Satz für den GCN-Backbone (für GCNLayer)
    gcn_params = {
        "activation_function": params.get("activation_function", "relu"),
        "bias": params.get("bias", True),
        "message_function": params.get("message_function", "copy_u"),
        "reduce_function": params.get("reduce_function", "sum"),
        "dropout_rate": params.get("dropout_rate", 0.0),
    }

    model = BGRLModel(
        in_feats, hidden_dim, emb_dim, gcn_params, ema_momentum=ema_momentum
    ).to(device)

    optimizer = th.optim.Adam(
        model.online_encoder.parameters(),
        lr=lr,
        weight_decay=float(params.get("weight_decay_pretrain", 1e-5)),
    )

    print(
        f"\n=== Starte GCL Pretraining (BGRL-style) für {num_epochs} Epochen ==="
    )
    dur = []
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        g1, x1 = augment_graph(
            g, features, edge_drop_prob=edge_drop, feat_drop_prob=feat_drop
        )
        g2, x2 = augment_graph(
            g, features, edge_drop_prob=edge_drop, feat_drop_prob=feat_drop
        )

        z1 = model.online_encoder(g1, x1)
        z2 = model.online_encoder(g2, x2)

        with th.no_grad():
            t1 = model.target_encoder(g1, x1)
            t2 = model.target_encoder(g2, x2)

        p1 = model.predictor(z1)
        p2 = model.predictor(z2)

        loss = byol_loss(p1, t2) + byol_loss(p2, t1)

        if not th.isfinite(loss):
            print(
                f"[Pretrain] Non-finite loss in Epoch {epoch}: {loss.item()}. Abbruch."
            )
            break

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.online_encoder.parameters(), max_norm=5.0
        )
        optimizer.step()
        model._update_target()

        dur.append(time.time() - t0)
        if log_interval > 0 and (
            epoch % log_interval == 0 or epoch == 1 or epoch == num_epochs
        ):
            print(
                f"[Pretrain] Epoch {epoch:04d}/{num_epochs:04d} | Loss {loss.item():.4f} | Time/epoch {np.mean(dur):.4f}s"
            )

    print("=== Pretraining beendet ===")
    return model.online_encoder, emb_dim


# Supervised-Training (Linear-Head + Fine-Tuning)
def train_linear_classifier(
    params,
    encoder,
    emb_dim,
    g,
    features,
    labels,
    train_mask,
    val_mask,
    num_epochs: int = 200,
):
    """
    Trainiert den linearen Kopf + optionales Fine-Tuning des Encoders.
    Kein Early Stopping mehr – es werden immer alle num_epochs durchlaufen.
    Bestes Modell wird über Validation-LogLoss gemerkt und am Ende geladen.
    """
    freeze_encoder = bool(params.get("freeze_encoder", False))
    clf = GCLClassifier(
        encoder, emb_dim, num_classes=2, freeze_encoder=freeze_encoder
    ).to(device)

    lr = float(params.get("learning_rate_classifier", 5e-3))
    weight_decay = float(params.get("weight_decay_classifier", 0.0))
    encoder_lr_factor = float(params.get("encoder_lr_factor", 0.1))

    # Parameter-Gruppen für Optimizer
    encoder_params = [p for p in clf.encoder.parameters() if p.requires_grad]
    head_params = list(clf.fc.parameters())

    if encoder_params:
        optimizer = th.optim.Adam(
            [
                {
                    "params": encoder_params,
                    "lr": lr * encoder_lr_factor,
                    "weight_decay": weight_decay,
                },
                {
                    "params": head_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            ]
        )
    else:
        optimizer = th.optim.Adam(
            head_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    # Class Weights wie in GCN.py (Balancing wird erzwungen)
    labels_train = labels[train_mask]
    class_weights = apply_gcn_balancing_strategy(labels_train, params)

    best_state = None
    best_val_loss = float("inf")
    dur = []

    print(
        "\n=== Starte supervised Training (Linear-Classifier + Fine-Tuning, ohne Early Stopping) ==="
    )
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        clf.train()
        logits_full = clf(g, features)
        logits = logits_full[train_mask]

        logits = th.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        logp = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(logp, labels_train, weight=class_weights)

        if not th.isfinite(loss):
            print(
                f"[Classifier] Non-finite loss @epoch {epoch}: {loss.item()}. Skip."
            )
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), max_norm=5.0)
        optimizer.step()

        # Validation
        clf.eval()
        with th.no_grad():
            v_logits = clf(g, features)[val_mask]
            v_logits = th.nan_to_num(
                v_logits, nan=0.0, posinf=1e6, neginf=-1e6
            )
            v_logp = F.log_softmax(v_logits, dim=1)
            v_loss = F.nll_loss(v_logp, labels[val_mask]).item()

        dur.append(time.time() - t0)

        head_lr = optimizer.param_groups[-1]["lr"]
        improved = v_loss < best_val_loss and math.isfinite(v_loss)
        if improved:
            best_val_loss = v_loss
            best_state = {
                k: v.cpu().clone() for k, v in clf.state_dict().items()
            }

        tag = "BEST" if improved else "    "
        print(
            f"[Classifier] Epoch {epoch:04d} | TrainLoss {loss.item():.4f} | ValLoss {v_loss:.4f} {tag} | LR_head {head_lr:.6f}"
        )

    if best_state is not None:
        clf.load_state_dict(best_state)
    else:
        print(
            "Kein best_state im Classifier-Training – Modell könnte instabil sein."
        )

    return clf


# Feature Importance (Option A: Gradient-basiert)
def calculate_gcl_feature_importance_gradient(
    model, g, features, labels, mask, max_nodes: int = 5000
):
    """
    Gradient-basierte Feature Importance (Option A):

    - Input: vollständiger Graph, Features, Labels, Mask (z.B. Val-Mask)
    - Loss: NLL auf den maskierten Knoten
    - Importance: mittlere |∂Loss/∂x_j| über alle Knoten im Mask-Subset

    Gibt eine Python-Liste der Länge num_features zurück.
    """
    print(
        "\nBerechne GCL Feature Importance via Input-Gradienten (Option A)..."
    )

    model.eval()
    features = features.to(device)
    labels = labels.to(device)

    # Indizes der genutzten Knoten
    node_idx = th.nonzero(mask, as_tuple=False).squeeze()
    if node_idx.dim() == 0:
        node_idx = node_idx.unsqueeze(0)
    if node_idx.numel() == 0:
        print(
            "Keine Knoten in der Maske für Feature Importance gefunden – gebe NaNs zurück."
        )
        return np.full(features.shape[1], np.nan).tolist()

    # Subsampling, um Speicher/Runtime zu schonen
    if node_idx.numel() > max_nodes:
        perm = th.randperm(node_idx.numel(), device=node_idx.device)
        node_idx = node_idx[perm[:max_nodes]]

    # Features klonen und Gradienten erlauben
    x = features.clone().detach()
    x.requires_grad = True

    # Forward
    logits = model(g, x)
    logits = th.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    logp = F.log_softmax(logits, dim=1)

    y = labels[node_idx]
    loss = F.nll_loss(logp[node_idx], y)

    # Backward
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()

    grads = x.grad[node_idx]  # (subset_nodes, num_features)
    imp = grads.abs().mean(dim=0)  # (num_features,)

    return imp.detach().cpu().numpy().tolist()


# Orchestrierung pro Artefakt
def run_for_artifact(
    artifact_dir_name: str,
    artifact_root: str = ARTIFACT_ROOT,
    results_dir: str = RESULTS_DIR,
) -> str:
    """
    Lädt ein Artefakt, führt:
      1) Hyperopt über Pretraining- & Classifier-Parameter
      2) Finales Pretraining mit besten Parametern
      3) Finales supervised Training (inkl. Fine-Tuning)
      4) Evaluation mit Threshold-Tuning τ* (analog GCN/GAT)
      5) Feature Importance (Gradient-basiert, Option A auf Validation-Split)

    Speichert eine JSON mit Validation- und Test-Metriken (jeweils @0.5 und @τ*)
    sowie eine CSV mit Feature-Importances.
    """
    artifact_dir = os.path.join(artifact_root, artifact_dir_name)

    # Fester Basissatz an Parametern für Backbone & Defaults
    base_params = {
        # GCN-Backbone / Pretraining Defaults
        "activation_function": "relu",
        "hidden_dim": 64,
        "emb_dim": 64,
        "learning_rate_pretrain": 1e-3,
        "weight_decay_pretrain": 1e-5,
        "edge_drop_prob": 0.2,
        "feat_drop_prob": 0.3,
        "ema_momentum": 0.99,
        # Classifier-Defaults; werden von Hyperopt überschrieben
        "learning_rate_classifier": 5e-3,
        "weight_decay_classifier": 0.0,
        # Balancing-Strategie wie GCN (erzwungen)
        "balancing_strategy": "class_weights",
        "class_weight_base": 1.0,
        "class_weight_exp": 1.0,
        # Fine-Tuning
        "freeze_encoder": False,  # wir wollen default fein-tunen
        "encoder_lr_factor": 0.1,  # wird von Hyperopt überschrieben
        # GCN-Backbone-Defaults (kompatibel zu robust_load_elliptic_dataset)
        "message_function": "copy_u",
        "reduce_function": "sum",
        "bias": True,
        "number_of_layers": 2,
        "optimizer": "adam",
        "dropout_rate": 0.0,
    }

    print(f"\n=== Starte GCL-Run für Artefakt: {artifact_dir_name} ===")

    # Daten laden
    try:
        g, features, labels, train_mask, test_mask, val_mask, df_edges = (
            robust_load_elliptic_dataset(
                base_params, artifact_dir=artifact_dir
            )
        )
    except Exception as e:
        print(f"Datenladung fehlgeschlagen: {e}")
        out = {"status": STATUS_FAIL, "error": str(e)}
        json_path = os.path.join(
            results_dir, f"elliptic_GCL_{artifact_dir_name}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_nan_to_none(out), f, indent=2)
        return json_path

    analyze_graph_data(g, features, labels, train_mask, test_mask, val_mask)

    # 1) Hyperopt: Pretraining + Classifier + Fine-Tuning
    print(
        "\n=== [1/3] Hyperopt über Pretraining- und Classifier-Parameter ==="
    )

    def objective_hpo(hp_params):
        """
        Ein Hyperopt-Trial:
        - Pretraining mit hp_params (kürzer)
        - Classifier-Training mit hp_params (kürzer)
        - Threshold-Tuning τ* auf Validation
        - Ziel: Validation-LogLoss @τ* minimieren
        """
        # Seeds resetten für Reproduzierbarkeit zwischen Trials
        random.seed(SEED)
        np.random.seed(SEED)
        th.manual_seed(SEED)
        dgl.random.seed(SEED)

        trial_params = dict(base_params)
        trial_params.update(hp_params)

        try:
            # Pretraining (gekürzt für HPO)
            encoder_trial, emb_dim_trial = pretrain_gcl(
                trial_params,
                g,
                features,
                num_epochs=NUM_EPOCHS_PRETRAIN_HPO,
                log_interval=0,  # während HPO nicht spammen
            )

            # Classifier-Training (gekürzt für HPO)
            clf_trial = train_linear_classifier(
                trial_params,
                encoder_trial,
                emb_dim_trial,
                g,
                features,
                labels,
                train_mask,
                val_mask,
                num_epochs=NUM_EPOCHS_CLASSIFIER_HPO,
            )

        except Exception as e:
            print(f"[Hyperopt] Fehler im Training: {e}")
            return {"status": STATUS_FAIL}

        # Threshold-Tuning τ* auf Validation (analog GCN/GAT)
        clf_trial.eval()
        with th.no_grad():
            v_logits = clf_trial(g, features.float())[val_mask]
            v_logits = th.nan_to_num(
                v_logits, nan=0.0, posinf=1e6, neginf=-1e6
            )
            v_proba = F.softmax(v_logits, dim=1)
            v_proba = th.nan_to_num(v_proba, nan=1e-8, posinf=1e6, neginf=-1e6)
            v_proba = v_proba / (v_proba.sum(dim=1, keepdim=True) + 1e-12)
            p0 = v_proba[:, 0].cpu().numpy()

        y_val_np = labels[val_mask].cpu().numpy()
        tau_star, score_star = tune_threshold(
            y_val_np, p0, metric=THRESH_METRIC
        )

        # Evaluation @τ* (log_loss als Hyperopt-Ziel, konsistent mit GCN/GAT)
        _, _, val_tuned = evaluate(
            clf_trial,
            g,
            features.float(),
            labels,
            val_mask,
            "Validation (tuned, Hyperopt)",
            tuned_threshold=tau_star,
        )

        loss = float(val_tuned.get("log_loss", np.inf))
        print(
            f"[Hyperopt] τ*={tau_star:.4f}, {THRESH_METRIC}={score_star:.4f}, log_loss={loss:.4f}"
        )
        if not math.isfinite(loss):
            return {"status": STATUS_FAIL}
        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective_hpo,
        space=hpo_space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=NUM_HYPEROPT_TRIALS,
        rstate=np.random.default_rng(SEED),
    )

    best_hpo_params = space_eval(hpo_space, best)
    print(
        "\nBeste Hyperopt-Parameter (Pretraining + Classifier + Fine-Tuning):"
    )
    print(best_hpo_params)

    # Kombiniere Base-Params mit best_hpo_params für das finale Training
    final_params = dict(base_params)
    final_params.update(best_hpo_params)

    # 2) Finales Pretraining mit besten Parametern
    print("\n=== [2/3] Finales Pretraining mit besten Hyperparametern ===")
    encoder_final, emb_dim_final = pretrain_gcl(
        final_params,
        g,
        features,
        num_epochs=NUM_EPOCHS_PRETRAIN_FINAL,
        log_interval=10,
    )

    # 3) Finaler Linear-Classifier mit besten Parametern
    print(
        "\n=== [3/3] Finales Training des Linear-Classifiers mit besten Parametern ==="
    )
    clf = train_linear_classifier(
        final_params,
        encoder_final,
        emb_dim_final,
        g,
        features,
        labels,
        train_mask,
        val_mask,
        num_epochs=NUM_EPOCHS_CLASSIFIER_FINAL,
    )

    # 4) Evaluation mit Threshold-Tuning (wie GCN/GAT)

    # 4a) Validation @0.5
    print("\n=== Evaluation Validation @0.5 ===")
    roc_val_05, pr_val_05, val_results_05 = evaluate(
        clf,
        g,
        features.float(),
        labels,
        val_mask,
        "Validation (argmax @0.5)",
        tuned_threshold=None,
    )

    # 4b) τ* auf Validation bestimmen
    clf.eval()
    with th.no_grad():
        v_logits = clf(g, features.float())[val_mask]
        v_logits = th.nan_to_num(v_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        v_proba = F.softmax(v_logits, dim=1)
        v_proba = th.nan_to_num(v_proba, nan=1e-8, posinf=1e6, neginf=-1e6)
        v_proba = v_proba / (v_proba.sum(dim=1, keepdim=True) + 1e-12)
        p0_val = v_proba[:, 0].cpu().numpy()

    y_val_np = labels[val_mask].cpu().numpy()
    tau_star, score_star = tune_threshold(
        y_val_np, p0_val, metric=THRESH_METRIC
    )
    print(
        f"\nBestes τ* auf Validation (Metric={THRESH_METRIC}): {tau_star:.4f} (Score={score_star:.4f})"
    )

    # 4c) Validation @τ*
    print("\n=== Evaluation Validation @τ* ===")
    roc_val_tuned, pr_val_tuned, val_results_tuned = evaluate(
        clf,
        g,
        features.float(),
        labels,
        val_mask,
        "Validation (tuned)",
        tuned_threshold=tau_star,
    )

    # 4d) Final Test @0.5
    print("\n=== Evaluation Final Test @0.5 ===")
    roc_test_05, pr_test_05, test_results_05 = evaluate(
        clf,
        g,
        features.float(),
        labels,
        test_mask,
        "Final Test (argmax @0.5)",
        tuned_threshold=None,
    )

    # 4e) Final Test @τ*
    print("\n=== Evaluation Final Test @τ* ===")
    roc_test_tuned, pr_test_tuned, test_results_tuned = evaluate(
        clf,
        g,
        features.float(),
        labels,
        test_mask,
        "Final Test (tuned)",
        tuned_threshold=tau_star,
    )

    # 5) Feature Importance (Gradient-basiert, Option A)
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS (Gradient-based, Option A)")
    print("=" * 60)

    feat_imp_grad = calculate_gcl_feature_importance_gradient(
        clf,
        g,
        features.float(),
        labels,
        val_mask,  # analog GCN/GAT: Importance auf Validation-Split
        max_nodes=5000,
    )

    num_features = features.shape[1]
    feat_names = [f"feature_{i}" for i in range(num_features)]

    df_grad = pd.DataFrame(
        {
            "feature": feat_names,
            "grad_importance": feat_imp_grad,
        }
    )
    df_grad = df_grad.sort_values(
        "grad_importance", ascending=False
    ).reset_index(drop=True)
    df_grad["grad_rank"] = np.arange(1, len(df_grad) + 1)

    print("\n=== Gradient Feature Importance Ranking (Top 15) ===")
    print(df_grad.head(15))

    csv_imp_path = os.path.join(
        results_dir,
        f"elliptic_GCL_{artifact_dir_name}_feature_importance_gradient.csv",
    )
    df_grad.to_csv(csv_imp_path, index=False)
    print(f"\nGradient Feature Importance gespeichert unter {csv_imp_path}")

    # JSON-Output
    pretrain_keys = [
        "activation_function",
        "hidden_dim",
        "emb_dim",
        "learning_rate_pretrain",
        "weight_decay_pretrain",
        "edge_drop_prob",
        "feat_drop_prob",
        "ema_momentum",
    ]
    clf_keys = [
        "learning_rate_classifier",
        "weight_decay_classifier",
        "balancing_strategy",
        "class_weight_base",
        "class_weight_exp",
        "freeze_encoder",
        "encoder_lr_factor",
    ]

    best_pretrain_params = {
        k: final_params[k] for k in pretrain_keys if k in final_params
    }
    best_classifier_params = {
        k: final_params[k] for k in clf_keys if k in final_params
    }

    out = {
        "status": STATUS_OK,
        "model_type": "GCL_BGRL_GCN",
        "artifact": artifact_dir_name,
        "timestamp": datetime.utcnow().isoformat(),
        "params_base": base_params,
        "best_hyperopt_params": best_hpo_params,
        "best_pretrain_params": best_pretrain_params,
        "best_classifier_params": best_classifier_params,
        "final_params": final_params,
        "threshold_metric": THRESH_METRIC,
        "tuned_threshold": float(tau_star),
        # Validation
        "validation_results_argmax": val_results_05,
        "validation_results_tuned": val_results_tuned,
        # Test
        "test_results_argmax": test_results_05,
        "test_results_tuned": test_results_tuned,
        # Feature Importance
        "feature_importance_gradient": feat_imp_grad,
        "feature_importance_csv": csv_imp_path,
    }

    json_path = os.path.join(
        results_dir, f"elliptic_GCL_{artifact_dir_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_none(out), f, indent=2)

    print(f"\nFertig: Ergebnisse gespeichert unter {json_path}")
    return json_path


# CLI / Batch-Runner
def main():
    ap = argparse.ArgumentParser(
        description="GCL Elliptic – BGRL auf GCN-Backbone (mit Hyperopt)"
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

    if args.artifact:
        run_for_artifact(args.artifact)
        return

    root = args.folder or ARTIFACT_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Ordner nicht gefunden: {root}")

    pattern = args.pattern or ""
    all_dirs = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    candidates = [d for d in all_dirs if pattern in d]

    if args.skip_existing:
        remaining = []
        for name in candidates:
            out_path = os.path.join(RESULTS_DIR, f"elliptic_GCL_{name}.json")
            if os.path.exists(out_path):
                print(f"Skipping {name} (JSON existiert bereits)")
            else:
                remaining.append(name)
        candidates = remaining

    if not candidates:
        print("Keine passenden Artefakte gefunden.")
        return

    print(f"Starte Runs für {len(candidates)} Artefakte…")
    for name in candidates:
        if "base93" in name:
            try:
                path = run_for_artifact(name)
                print(f"Fertig: {name} -> {path}")
            except Exception as e:
                print(f"Fehler bei {name}: {e}")


if __name__ == "__main__":
    main()
