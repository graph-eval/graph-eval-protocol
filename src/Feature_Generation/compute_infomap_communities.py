# -*- coding: utf-8 -*-
"""
Compute Infomap Communities for Elliptic Graph

- Loads edgelist
- Converts to igraph
- Computes Infomap communities
- Saves per-node community assignment as Parquet

Output columns:
    txId, infomap_comm
"""

import os
import pandas as pd
import igraph as ig


def get_paths(drop_rate=None, variant=None):
    """
    Bestimmt Input-Edgelist + Output-Parquet.

    - variant:
      -> erwartet: ../EllipticDataSet/elliptic_txs_edgelist_<variant>.csv
      -> output:   artifacts/infomap_communities_<variant>.parquet

    - drop_rate: z.B. 25/50
      -> erwartet: ../EllipticDataSet/elliptic_txs_edgelist_<drop_rate>.csv
      -> output:   artifacts/infomap_communities_<drop_rate>.parquet

    - full graph:
      -> ../EllipticDataSet/elliptic_txs_edgelist.csv
      -> artifacts/infomap_communities.parquet
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # Basis-Pfade (wie in deiner alten Datei)
    BASE_DIR = r"C:\Experiments"
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

    # Suffix-Regel identisch zu ffstruc2vec:
    if variant is not None:
        suffix = f"_{variant}"
        edgelist_path = os.path.join(
            here, "..", "EllipticDataSet", f"elliptic_txs_edgelist{suffix}.csv"
        )
        file_name = f"infomap_communities{suffix}.parquet"
        mode_str = f"TARGETED DROP VARIANT (variant={variant})"
    elif drop_rate is None:
        suffix = ""
        edgelist_path = os.path.join(
            here, "..", "EllipticDataSet", "elliptic_txs_edgelist.csv"
        )
        file_name = "infomap_communities.parquet"
        mode_str = "FULL GRAPH"
    else:
        suffix = f"_{drop_rate}"
        edgelist_path = os.path.join(
            here, "..", "EllipticDataSet", f"elliptic_txs_edgelist{suffix}.csv"
        )
        file_name = f"infomap_communities{suffix}.parquet"
        mode_str = f"EDGE-DROP (drop_rate={drop_rate}%)"

    output_path = os.path.join(ARTIFACTS_DIR, file_name)

    return {
        "here": here,
        "edgelist_path": edgelist_path,
        "output_path": output_path,
        "suffix": suffix,
        "mode_str": mode_str,
        "artifacts_dir": ARTIFACTS_DIR,
    }


def compute_infomap_communities(edgelist_path, output_path, trials=10):
    print("Computing Infomap communities...")

    # Edgelist laden (LOGIK 1:1 beibehalten)
    edges = []
    node_set = set()

    with open(edgelist_path, "r") as f:
        next(f)  # Header überspringen
        for line in f:
            u, v = line.strip().split(",")
            u = int(u)
            v = int(v)
            edges.append((u, v))
            node_set.add(u)
            node_set.add(v)

    # Mapping Original-ID -> 0..N-1 (LOGIK 1:1 beibehalten)
    node_list = list(node_set)
    mapping = {node: idx for idx, node in enumerate(node_list)}
    mapped_edges = [(mapping[u], mapping[v]) for (u, v) in edges]

    # igraph-Graph bauen (LOGIK 1:1 beibehalten)
    g = ig.Graph()
    g.add_vertices(len(node_list))
    g.add_edges(mapped_edges)

    print(f"Graph loaded: {g.vcount()} nodes, {g.ecount()} edges")
    print("Running Infomap (igraph.community_infomap)...")

    # Infomap (LOGIK 1:1 beibehalten)
    partition = g.community_infomap(trials=trials)
    print(f"Infomap finished: {len(partition)} communities found")

    try:
        print(
            f"Infomap quality (code length proxy): {partition.quality():.4f}"
        )
    except Exception:
        pass

    membership = partition.membership

    # Output (LOGIK 1:1 beibehalten)
    df_out = pd.DataFrame({"txId": node_list, "infomap_comm": membership})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    print(f"Infomap communities saved: {output_path}")
    print(f"Shape: {df_out.shape}")

    return df_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Infomap communities for Elliptic "
        "(full graph OR edge-drop variants OR targeted variants)."
    )

    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed in the edgelist (e.g. 25, 50). "
        "If omitted, the full graph edgelist is used (unless --variant is set).",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Targeted variant suffix. "
        "Expects ../EllipticDataSet/elliptic_txs_edgelist_<variant>.csv",
    )

    args = parser.parse_args()

    if args.variant is not None and args.drop_rate is not None:
        raise ValueError(
            "Bitte entweder --variant ODER --drop_rate setzen, nicht beides."
        )

    paths = get_paths(drop_rate=args.drop_rate, variant=args.variant)

    # Logging (deine alte Logging-Idee beibehalten)
    print("=" * 70)
    print("INFOMAP COMMUNITIES FOR ELLIPTIC")
    print(f"Mode          : {paths['mode_str']}")
    print(f"Input edgelist: {paths['edgelist_path']}")
    print(f"Output parquet: {paths['output_path']}")
    print("=" * 70)

    compute_infomap_communities(
        edgelist_path=paths["edgelist_path"],
        output_path=paths["output_path"],
    )
