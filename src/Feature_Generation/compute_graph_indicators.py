# compute_graph_indicators_standalone.py
import os
import time
import json
import warnings
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import argparse

from tqdm import tqdm

warnings.filterwarnings("ignore")


# GEMEINSAME GRAPH-LOADING LOGIK
def load_graphs_and_hash(edgelist_csv):
    """Lädt Graphen einmalig und berechnet konsistenten Hash."""
    print("Loading graphs and computing hash...")

    # EXPLIZITER STRING-CAST FÜR ROBUSTHEIT
    df = pd.read_csv(edgelist_csv, usecols=["txId1", "txId2"], dtype=str)

    # KANONISIERUNG für konsistenten Hash
    canon = df.assign(
        a=df[["txId1", "txId2"]].min(axis=1),
        b=df[["txId1", "txId2"]].max(axis=1),
    )[["a", "b"]].sort_values(["a", "b"])
    graph_hash = hashlib.sha256(
        pd.util.hash_pandas_object(canon, index=False).values.tobytes()
    ).hexdigest()[:16]

    # BEIDE GRAPH-VERSIONEN erstellen
    G = nx.from_pandas_edgelist(
        df, source="txId1", target="txId2", create_using=nx.DiGraph()
    )
    G = nx.relabel_nodes(G, lambda n: str(n))
    G_u = G.to_undirected()

    print(
        f"Graphs loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    print(f"Graph hash: {graph_hash}")

    return G, G_u, graph_hash


def save_meta(path_parquet, name, graph_hash, **params):
    """Speichert konsistente Metadaten."""
    meta = {
        "name": name,
        "graph_hash": graph_hash,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": params,
    }
    meta_path = path_parquet.replace(".parquet", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# DEGREE INDICATORS
def compute_degree_centrality(G_u, output_path, graph_hash):
    """Degree Centrality (ungerichtet)."""
    print("Computing Degree Centrality...")
    t0 = time.time()

    deg = nx.degree_centrality(G_u)
    nodes = sorted(G_u.nodes(), key=str)  # EXPLIZITER STRING-KEY

    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_degree_centrality": np.array(
                [deg.get(node, 0.0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "degree_centrality",
        graph_hash,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(f"Degree Centrality saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


def compute_in_out_degree(G, output_path_in, output_path_out, graph_hash):
    """In- und Out-Degree (gerichtet)."""
    print("Computing In/Out Degree...")
    t0 = time.time()

    nodes = sorted(G.nodes(), key=str)  # EXPLIZITER STRING-KEY
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    df_in = pd.DataFrame(
        {
            "txId": nodes,
            "gi_in_degree": np.array(
                [in_deg.get(node, 0) for node in nodes], dtype="float32"
            ),
        }
    )

    df_out = pd.DataFrame(
        {
            "txId": nodes,
            "gi_out_degree": np.array(
                [out_deg.get(node, 0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path_in), exist_ok=True)
    df_in.to_parquet(output_path_in, index=False)
    df_out.to_parquet(output_path_out, index=False)

    save_meta(
        output_path_in,
        "in_degree",
        graph_hash,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
    save_meta(
        output_path_out,
        "out_degree",
        graph_hash,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )

    print(f"In/Out Degree saved ({time.time()-t0:.1f}s)")
    return df_in, df_out


# CENTRALITY INDICATORS
def compute_pagerank(
    G, output_path, graph_hash, alpha=0.85, max_iter=100, tol=1e-8
):
    """PageRank (deterministisch auf gerichtetem Graphen)."""
    print("Computing PageRank (directed graph)...")
    t0 = time.time()

    # GERICHTETER GRAPH FÜR PAGERANK
    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    nodes = sorted(G.nodes(), key=str)

    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_pagerank": np.array(
                [pr.get(node, 0.0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "pagerank",
        graph_hash,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )

    print(f"PageRank saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


def compute_betweenness_centrality(
    G_u, output_path, graph_hash, k=2000, seed=42
):
    """Betweenness Centrality (approximativ für große Graphen)."""
    print("Computing Betweenness Centrality...")
    t0 = time.time()

    n_nodes = G_u.number_of_nodes()

    if n_nodes > 5000:
        print(f"  Using approximation (k={min(k, n_nodes)})...")
        bc = nx.betweenness_centrality(G_u, k=min(k, n_nodes), seed=seed)
        approx = True
        k_eff = min(k, n_nodes)
    else:
        print("  Using exact computation...")
        bc = nx.betweenness_centrality(G_u)
        approx = False
        k_eff = n_nodes

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_betweenness_centrality": np.array(
                [bc.get(node, 0.0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "betweenness_centrality",
        graph_hash,
        approximate=approx,
        k=k_eff,
        seed=seed,
        n_nodes=n_nodes,
        n_edges=G_u.number_of_edges(),
    )

    print(
        f"Betweenness Centrality saved: {output_path} ({time.time()-t0:.1f}s)"
    )
    return df


def compute_eigenvector_centrality(
    G_u, output_path, graph_hash, max_iter=1000, tol=1e-6
):
    """Eigenvector Centrality (komponentenweise mit konstanter Initialisierung)."""
    print("Computing Eigenvector Centrality...")
    t0 = time.time()

    values = {}
    components = list(nx.connected_components(G_u))

    print(f"  Processing {len(components)} connected components...")

    for i, comp in enumerate(components, 1):
        H = G_u.subgraph(comp).copy()
        if H.number_of_nodes() < 2:
            # Einzelknoten: Eigenvector = 0
            values.update({node: 0.0 for node in H.nodes()})
            continue

        try:
            # KONSTANTE INITIALISIERUNG FÜR MAXIMALE STABILITÄT
            n = H.number_of_nodes()
            nstart = {node: 1.0 / n for node in H.nodes()}
            ev = nx.eigenvector_centrality(
                H, max_iter=max_iter, tol=tol, nstart=nstart
            )
            values.update(ev)
        except Exception as e:
            print(f"    Component {i} failed: {e}, using zeros")
            values.update({node: 0.0 for node in H.nodes()})

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_eigenvector_centrality": np.array(
                [values.get(node, 0.0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "eigenvector_centrality",
        graph_hash,
        max_iter=max_iter,
        tol=tol,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(
        f"Eigenvector Centrality saved: {output_path} ({time.time()-t0:.1f}s)"
    )
    return df


def compute_closeness_centrality(G_u, output_path, graph_hash):
    """Closeness Centrality."""
    print("Computing Closeness Centrality...")
    t0 = time.time()

    # Nur für kleinere Komponenten sinnvoll
    components = list(nx.connected_components(G_u))
    values = {}

    for i, comp in enumerate(components, 1):
        H = G_u.subgraph(comp).copy()
        if H.number_of_nodes() > 1:
            try:
                closeness = nx.closeness_centrality(H)
                values.update(closeness)
            except:
                values.update({node: 0.0 for node in H.nodes()})
        else:
            values.update({node: 0.0 for node in H.nodes()})

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_closeness_centrality": np.array(
                [values.get(node, 0.0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "closeness_centrality",
        graph_hash,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(f"Closeness Centrality saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


# CLUSTERING INDICATORS
def compute_clustering_coefficient(G_u, output_path, graph_hash):
    """Local Clustering Coefficient."""
    print("Computing Clustering Coefficient...")
    t0 = time.time()

    clustering = nx.clustering(G_u)
    nodes = sorted(G_u.nodes(), key=str)

    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_clustering_coefficient": np.array(
                [clustering.get(node, 0.0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "clustering_coefficient",
        graph_hash,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(
        f"Clustering Coefficient saved: {output_path} ({time.time()-t0:.1f}s)"
    )
    return df


def compute_square_clustering(G_u, output_path, graph_hash):
    """Square Clustering Coefficient."""
    print("Computing Square Clustering...")
    t0 = time.time()

    try:
        square_clustering = nx.square_clustering(G_u)
    except:
        square_clustering = {node: 0.0 for node in G_u.nodes()}

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_square_clustering": np.array(
                [square_clustering.get(node, 0.0) for node in nodes],
                dtype="float32",
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "square_clustering",
        graph_hash,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(f"Square Clustering saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


# STRUCTURAL INDICATORS
def compute_core_number(G_u, output_path, graph_hash):
    """K-Core Decomposition."""
    print("Computing Core Number...")
    t0 = time.time()

    try:
        core_number = nx.core_number(G_u)
    except:
        core_number = {node: 0 for node in G_u.nodes()}

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_core_number": np.array(
                [core_number.get(node, 0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "core_number",
        graph_hash,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(f"Core Number saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


def compute_triangles(G_u, output_path, graph_hash):
    """Number of Triangles."""
    print("Computing Triangles...")
    t0 = time.time()

    try:
        triangles = nx.triangles(G_u)
    except:
        triangles = {node: 0 for node in G_u.nodes()}

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_triangles": np.array(
                [triangles.get(node, 0) for node in nodes], dtype="float32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "triangles",
        graph_hash,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(f"Triangles saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


# COMMUNITY DETECTION (LOUVAIN)
def compute_louvain_community(
    G_u, output_path, graph_hash, seed=42, resolution=1.0
):
    """Louvain Community Detection (deterministisch)."""
    print("Computing Louvain Communities...")
    t0 = time.time()

    try:
        import community as community_louvain

        partition = community_louvain.best_partition(
            G_u, random_state=seed, resolution=resolution
        )
    except Exception as e:
        print(f"Louvain failed: {e}, using default communities")
        partition = {node: 0 for node in G_u.nodes()}

    nodes = sorted(G_u.nodes(), key=str)
    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_community_louvain": np.array(
                [partition.get(node, -1) for node in nodes], dtype="int32"
            ),
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "community_louvain",
        graph_hash,
        seed=seed,
        resolution=resolution,
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )

    print(f"Louvain Communities saved: {output_path} ({time.time()-t0:.1f}s)")
    return df


def compute_louvain_extras(
    G_u, partition, output_path, graph_hash, small_thresh=10
):
    """
    Erzeugt label-freie Community-Features pro Node:
      - gi_comm_size           (int)
      - gi_comm_density        (float)
      - gi_comm_is_small       (int {0,1})
      - gi_comm_code_1..gi_comm_code_5  (5D deterministische Hash-Embedding)
    """
    import numpy as np
    import pandas as pd
    import hashlib

    # Mapping: node -> community
    nodes = sorted(G_u.nodes(), key=str)
    comm = np.array([partition.get(n, -1) for n in nodes], dtype=np.int64)

    # --- Community-Statistiken ---
    # Größe
    _, counts = np.unique(comm, return_counts=True)
    comm_ids = np.unique(comm)
    size_map = dict(zip(comm_ids, counts))
    size = np.array([size_map[c] for c in comm], dtype=np.int32)

    # Dichte je Community: |E_in| / (n*(n-1)/2)  (für n>=2, sonst 0)
    density_map = {}
    for c in comm_ids:
        members = [n for n in nodes if partition.get(n, -1) == c]
        n = len(members)
        if n < 2:
            density_map[c] = 0.0
            continue
        H = G_u.subgraph(members)
        e = H.number_of_edges()
        denom = n * (n - 1) / 2.0
        density_map[c] = float(e) / denom if denom > 0 else 0.0
    density = np.array([density_map[c] for c in comm], dtype=np.float32)

    is_small = (size < int(small_thresh)).astype(np.int32)

    # --- 5D deterministische Hash-Embedding pro Community ---
    def _hash_to_vec(c, dim=5):
        # stabiler 5D-Vektor aus SHA256(c), normalisiert
        rs = np.random.RandomState(
            int(hashlib.sha256(str(int(c)).encode()).hexdigest()[:8], 16)
        )
        v = rs.normal(size=dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    code_cache = {int(c): _hash_to_vec(int(c), dim=5) for c in comm_ids}
    codes = np.vstack([code_cache[int(c)] for c in comm]).astype(np.float32)

    df = pd.DataFrame(
        {
            "txId": nodes,
            "gi_comm_size": size,
            "gi_comm_density": density,
            "gi_comm_is_small": is_small,
        }
    )
    for j in range(5):
        df[f"gi_comm_code_{j+1}"] = codes[:, j]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_meta(
        output_path,
        "community_louvain_extras",
        graph_hash,
        small_thresh=int(small_thresh),
        n_nodes=G_u.number_of_nodes(),
        n_edges=G_u.number_of_edges(),
    )
    print(f"Louvain extras saved: {output_path}")
    return df


def get_paths(drop_rate=None, variant=None):
    """
    Einheitliche Pfad-/Suffix-Logik
    """
    here = os.path.dirname(os.path.abspath(__file__))

    if variant is not None:
        suffix = f"_{variant}"
        edgelist_path = os.path.join(
            here, "..", "EllipticDataSet", f"elliptic_txs_edgelist{suffix}.csv"
        )
    elif drop_rate is None:
        suffix = ""
        edgelist_path = os.path.join(
            here, "..", "EllipticDataSet", "elliptic_txs_edgelist.csv"
        )
    else:
        suffix = f"_{drop_rate}"
        edgelist_path = os.path.join(
            here, "..", "EllipticDataSet", f"elliptic_txs_edgelist{suffix}.csv"
        )

    artifacts_dir = os.path.join(here, "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    return {
        "here": here,
        "suffix": suffix,
        "edgelist_path": edgelist_path,
        "artifacts_dir": artifacts_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute graph indicators for Elliptic (full graph, edge-drop variants, or targeted variants)."
    )
    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed in the edgelist variant (e.g. 25, 50). If omitted: full graph.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Alternative to drop_rate. "
            "Expects ../EllipticDataSet/elliptic_txs_edgelist_<variant>.csv"
        ),
    )
    args = parser.parse_args()

    paths = get_paths(drop_rate=args.drop_rate, variant=args.variant)
    suffix = paths["suffix"]
    edgelist_path = paths["edgelist_path"]
    artifacts_dir = paths["artifacts_dir"]

    print("=" * 70)
    print("COMPUTING GRAPH INDICATORS - STANDALONE")
    if args.variant is not None:
        print(f"Mode: TARGETED VARIANT (variant={args.variant})")
    elif args.drop_rate is None:
        print("Mode: FULL GRAPH")
    else:
        print(f"Mode: EDGE-DROP VARIANT (drop_rate={args.drop_rate}%)")
    print(f"Using edgelist: {edgelist_path}")
    print("Consistent graph hash + deterministic computation")
    print("=" * 70)

    # GRAPH EINMALIG LADEN
    G, G_u, graph_hash = load_graphs_and_hash(edgelist_path)

    # ALLE INDICATORS BERECHNEN
    compute_degree_centrality(
        G_u,
        os.path.join(artifacts_dir, f"degree_centrality{suffix}.parquet"),
        graph_hash,
    )
    compute_in_out_degree(
        G,
        os.path.join(artifacts_dir, f"in_degree{suffix}.parquet"),
        os.path.join(artifacts_dir, f"out_degree{suffix}.parquet"),
        graph_hash,
    )
    compute_pagerank(
        G, os.path.join(artifacts_dir, f"pagerank{suffix}.parquet"), graph_hash
    )
    compute_betweenness_centrality(
        G_u,
        os.path.join(artifacts_dir, f"betweenness_centrality{suffix}.parquet"),
        graph_hash,
    )
    compute_eigenvector_centrality(
        G_u,
        os.path.join(artifacts_dir, f"eigenvector_centrality{suffix}.parquet"),
        graph_hash,
    )
    compute_closeness_centrality(
        G_u,
        os.path.join(artifacts_dir, f"closeness_centrality{suffix}.parquet"),
        graph_hash,
    )
    compute_clustering_coefficient(
        G_u,
        os.path.join(artifacts_dir, f"clustering_coefficient{suffix}.parquet"),
        graph_hash,
    )
    compute_square_clustering(
        G_u,
        os.path.join(artifacts_dir, f"square_clustering{suffix}.parquet"),
        graph_hash,
    )
    compute_core_number(
        G_u,
        os.path.join(artifacts_dir, f"core_number{suffix}.parquet"),
        graph_hash,
    )
    compute_triangles(
        G_u,
        os.path.join(artifacts_dir, f"triangles{suffix}.parquet"),
        graph_hash,
    )

    # Louvain Communities + Extras
    df_comm = compute_louvain_community(
        G_u,
        os.path.join(artifacts_dir, f"community_louvain{suffix}.parquet"),
        graph_hash,
    )

    try:
        part = dict(
            zip(
                df_comm["txId"].astype(str),
                df_comm["gi_community_louvain"].astype(int),
            )
        )
        compute_louvain_extras(
            G_u,
            part,
            os.path.join(
                artifacts_dir, f"community_louvain_extras{suffix}.parquet"
            ),
            graph_hash,
            small_thresh=10,
        )
    except Exception as e:
        print(f"[WARN] Could not compute Louvain extras: {e}")

    print("=" * 70)
    print("ALL GRAPH INDICATORS COMPLETED!")
    print(f"Graph hash: {graph_hash}")
    print(f"Artifacts written to: {artifacts_dir} (suffix='{suffix}')")
    print("Individual Parquet files + Meta JSON created")
    print("=" * 70)
