# -*- coding: utf-8 -*-
"""
Exports artifacts (Parquet) for Elliptic experiments:
- X_train / X_validation / X_test
- y_train / y_validation / y_test
- meta.json containing the configuration
"""

from __future__ import annotations
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from elliptic_data import load_elliptic_splits


# Konfiguration (kann von Wrappern überschrieben werden)
CFG: Dict[str, Any] = {
    "base_dir": "..",
    "week_max": 42,
    "split_mode": "random",  # "random" | "temporal"
    "temporal_cfg": None,
    "encode_labels": "01",
    "val_size": 0.33,
    "test_size": 0.25,
    "random_state": 42,
    "edgelist_file": "EllipticDataSet/elliptic_txs_edgelist.csv",
    "directed_graph": True,
    "gi_name_prefix": "gi_",
    "gi_subset": None,  # None = alle Indikatoren aus Utils
    "emb_name_prefix": "emb_",
    "embedding_dimensions": 64,
    # Artefakt-Tag:
    "feature_tag": "base93_def",
    # Edge-Drop:
    "variant": None,
    # GRAPH INDICATORS
    "add_degree_centrality": False,
    "add_in_out_degree": False,
    "add_pagerank": False,
    "add_betweenness_centrality": False,
    "add_eigenvector_centrality": False,
    "add_closeness_centrality": False,
    "add_clustering_coefficient": False,
    "add_square_clustering": False,
    "add_core_number": False,
    "add_triangles": False,
    "add_community_louvain": False,
    "add_community_leiden": False,
    "add_community_infomap": False,
    # Proximity Embeddings
    "add_node2vec_balanced": False,
    "add_node2vec_dfs": False,
    "add_node2vec_bfs": False,
    "add_deepwalk": False,
    # Spectral Embeddings
    "add_spectral_embeddings": False,
    # Structural Embeddings
    "add_ffstruc2vec_embeddings": False,
    "add_graphwave_embeddings": False,
    "add_role2vec_embeddings": False,
    # GNNs
    "add_gcn_embeddings": False,
    "add_gat_embeddings": False,
    "add_gcl_embeddings": False,
}


def _artifact_dir_name(tag: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(
        json.dumps(CFG, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    return f"{ts}_{tag}_{h}"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)


def main():
    # Daten laden (inkl. GI wenn konfiguriert)
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_elliptic_splits(
        base_dir=CFG["base_dir"],
        week_max=CFG["week_max"],
        non_negative_shift=True,
        encode_labels=CFG["encode_labels"],
        val_size=CFG.get("val_size", 0.33),
        test_size=CFG.get("test_size", 0.25),
        random_state=CFG.get("random_state", 42),
        split_mode=CFG["split_mode"],
        temporal_cfg=CFG.get("temporal_cfg"),
        # GRAPH INDICATORS
        add_degree_centrality=CFG.get("add_degree_centrality", False),
        add_in_out_degree=CFG.get("add_in_out_degree", False),
        add_pagerank=CFG.get("add_pagerank", False),
        add_betweenness_centrality=CFG.get(
            "add_betweenness_centrality", False
        ),
        add_eigenvector_centrality=CFG.get(
            "add_eigenvector_centrality", False
        ),
        add_closeness_centrality=CFG.get("add_closeness_centrality", False),
        add_clustering_coefficient=CFG.get(
            "add_clustering_coefficient", False
        ),
        add_square_clustering=CFG.get("add_square_clustering", False),
        add_core_number=CFG.get("add_core_number", False),
        add_triangles=CFG.get("add_triangles", False),
        add_community_louvain=CFG.get("add_community_louvain", False),
        add_community_leiden=CFG.get("add_community_leiden", False),
        add_community_infomap=CFG.get("add_community_infomap", False),
        # Proximity Embeddings:
        add_node2vec_balanced=CFG.get("add_node2vec_balanced", False),
        add_node2vec_dfs=CFG.get("add_node2vec_dfs", False),
        add_node2vec_bfs=CFG.get("add_node2vec_bfs", False),
        add_deepwalk=CFG.get("add_deepwalk", False),
        # Spectral Embeddings:
        add_spectral_embeddings=CFG.get("add_spectral_embeddings", False),
        # Structural Embeddings:
        add_ffstruc2vec_embeddings=CFG.get(
            "add_ffstruc2vec_embeddings", False
        ),
        add_graphwave_embeddings=CFG.get("add_graphwave_embeddings", False),
        add_role2vec_embeddings=CFG.get("add_role2vec_embeddings", False),
        # GNNs
        add_gcn_embeddings=CFG.get("add_gcn_embeddings", False),
        add_gat_embeddings=CFG.get("add_gat_embeddings", False),
        add_gcl_embeddings=CFG.get("add_gcl_embeddings", False),
        edgelist_file=CFG.get(
            "edgelist_file", "EllipticDataSet/elliptic_txs_edgelist.csv"
        ),
        directed_graph=CFG.get("directed_graph", True),
        gi_name_prefix=CFG.get("gi_name_prefix", "gi_"),
        gi_subset=CFG.get("gi_subset"),
        emb_name_prefix=CFG.get("emb_name_prefix", "emb_"),
        embedding_dimensions=CFG.get("embedding_dimensions", 64),
        # Edge-Drop:
        variant=CFG.get("variant"),
    )

    # Artefaktverzeichnis
    artifacts_root = os.path.join("..", "artifacts", "elliptic")
    tag = CFG.get("feature_tag", "base93")

    # langen GI-Block verkürzen
    ALL_GI = "I1+I2+I3+I4+I5+I6+I7+I8+I9+I10+I11+I12+I13"
    if ALL_GI in tag:
        tag = tag.replace(ALL_GI, "allGI")

    art_dir_name = _artifact_dir_name(tag)
    out_dir = os.path.join(artifacts_root, art_dir_name)
    _ensure_dir(out_dir)

    # Speichern
    _save_parquet(X_tr, os.path.join(out_dir, "X_train.parquet"))
    _save_parquet(X_val, os.path.join(out_dir, "X_validation.parquet"))
    _save_parquet(X_te, os.path.join(out_dir, "X_test.parquet"))
    _save_parquet(
        y_tr.to_frame("label"), os.path.join(out_dir, "y_train.parquet")
    )
    _save_parquet(
        y_val.to_frame("label"), os.path.join(out_dir, "y_validation.parquet")
    )
    _save_parquet(
        y_te.to_frame("label"), os.path.join(out_dir, "y_test.parquet")
    )

    # Meta
    gi_cols = [
        c
        for c in X_tr.columns
        if isinstance(c, str)
        and c.startswith(CFG.get("gi_name_prefix", "gi_"))
    ]
    meta = {
        "artifact_dir": art_dir_name,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "split_mode": CFG["split_mode"],
        "temporal_cfg": CFG.get("temporal_cfg"),
        "week_max": CFG["week_max"],
        "random_state": CFG.get("random_state", 42),
        "encode_labels": CFG.get("encode_labels", "01"),
        "shapes": {
            "X_train": list(X_tr.shape),
            "X_validation": list(X_val.shape),
            "X_test": list(X_te.shape),
        },
        "feature_tag": tag,
        "edge_drop": {
            "variant": CFG.get("variant"),
        },
        "graph_indicators": {
            "enabled": len(gi_cols) > 0,
            "prefix": CFG.get("gi_name_prefix", "gi_"),
            "count": len(gi_cols),
            "columns": gi_cols[:200],
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nArtefakte exportiert nach: {out_dir}")


if __name__ == "__main__":
    main()
