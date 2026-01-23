# -*- coding: utf-8 -*-
"""
Compute Leiden Communities for Elliptic Graph

- Loads edgelist
- Converts to igraph
- Computes Leiden communities (modularity-optimizing)
- Saves per-node community assignment as Parquet

Output columns:
    txId, leiden_comm
"""

import pandas as pd
import numpy as np
import igraph as ig
import leidenalg
import os
from tqdm import tqdm


def compute_leiden_communities(
    edgelist_path, output_path, resolution=1.0, random_state=42
):
    print("Computing Leiden communities...")

    # Load edgelist
    edges = []
    node_set = set()

    with open(edgelist_path, "r") as f:
        next(f)
        for line in f:
            u, v = line.strip().split(",")
            u = int(u)
            v = int(v)
            edges.append((u, v))
            node_set.add(u)
            node_set.add(v)

    # Mapping
    node_list = sorted(list(node_set))
    mapping = {node: idx for idx, node in enumerate(node_list)}

    mapped_edges = [(mapping[u], mapping[v]) for (u, v) in edges]

    # Build igraph
    g = ig.Graph()
    g.add_vertices(len(node_list))
    g.add_edges(mapped_edges)

    print(f"✓ Graph loaded: {g.vcount()} nodes, {g.ecount()} edges")

    # Optional: sampling for very large graphs
    if g.vcount() > 200000:
        print("Large graph detected – Leiden may be slow")
        print("  Consider sampling or enabling multiprocessing")

    print("Running Leiden algorithm...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state,
    )

    print(f"Leiden finished: {len(partition)} communities found")
    try:
        mod = partition.quality()
        print(f"Modularity: {mod:.4f}")
    except:
        pass

    print("Preparing output...")

    membership = partition.membership

    # node_list ist die Liste der Original-IDs in der gleichen Reihenfolge,
    # in der wir die Knoten im Graphen angelegt haben -> Index i passt zu membership[i]
    df_out = pd.DataFrame({"txId": node_list, "leiden_comm": membership})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    print(f"Leiden communities saved: {output_path}")
    print(f"Shape: {df_out.shape}")

    return df_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Leiden communities for Elliptic "
        "(full graph OR edge-drop variants OR targeted variants)."
    )
    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed in the edgelist (e.g. 25, 50). "
        "If omitted, the full graph edgelist is used. "
        "Mutually exclusive with --variant.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Alternative to --drop_rate. "
            "Expects an input file elliptic_txs_edgelist_<variant>.csv in the EllipticDataSet folder."
        ),
    )

    args = parser.parse_args()

    if args.drop_rate is not None and args.variant is not None:
        raise ValueError(
            "Please use either --drop_rate OR --variant, not both."
        )

    BASE_DIR = r"C:\Experiments"
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

    # --- Suffix & paths analog zu ffstruc2vec ---
    if args.variant is not None:
        # targeted variant
        suffix = f"_{args.variant}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
        file_name = f"leiden_communities{suffix}.parquet"
        mode_str = f"TARGETED VARIANT (variant={args.variant})"

    elif args.drop_rate is None:
        # full graph
        suffix = ""
        edgelist_path = "../EllipticDataSet/elliptic_txs_edgelist.csv"
        file_name = "leiden_communities.parquet"
        mode_str = "FULL GRAPH"

    else:
        # random edge-drop variant (25/50)
        suffix = f"_{args.drop_rate}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
        file_name = f"leiden_communities{suffix}.parquet"
        mode_str = f"EDGE-DROP (drop_rate={args.drop_rate}%)"

    output_path = os.path.join(ARTIFACTS_DIR, file_name)

    # Logging
    print("=" * 70)
    print("LEIDEN COMMUNITIES FOR ELLIPTIC")
    print(f"Mode          : {mode_str}")
    print(f"Input edgelist: {edgelist_path}")
    print(f"Output parquet: {output_path}")
    print("=" * 70)

    compute_leiden_communities(
        edgelist_path=edgelist_path, output_path=output_path
    )
