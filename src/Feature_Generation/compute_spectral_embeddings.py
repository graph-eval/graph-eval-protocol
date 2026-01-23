# compute_spectral_embeddings.py
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")


def _normalized_laplacian_sparse(G_undirected) -> csr_matrix:
    """Sparse normalisierte Laplace-Matrix L_sym = I - D^{-1/2} A D^{-1/2}."""
    L = nx.normalized_laplacian_matrix(G_undirected)  # SciPy CSR
    if not issparse(L):
        L = csr_matrix(L)
    return L.astype(float)


def _eigs_smallest(
    L: csr_matrix, k: int, tol=1e-3, maxiter=2000, retries=(1e-2, 1e-1)
):
    """
    Kleinste Eigenwerte/-vektoren (SM) mit robuster Rückfallebene.
    Gibt (w, V) zurück (beide reell).
    """
    n = L.shape[0]
    k = min(
        k, max(1, n - 1)
    )  # höchstens n-1 nicht-triviale Eigenwerte im zusammenhängenden Graph
    # 1. Versuch
    try:
        w, V = eigsh(L, k=k, which="SM", tol=tol, maxiter=maxiter)
        return w.real, V.real
    except Exception:
        pass
    # Retry mit relaxter Toleranz
    for t in retries:
        try:
            w, V = eigsh(L, k=k, which="SM", tol=t, maxiter=maxiter)
            return w.real, V.real
        except Exception:
            continue
    # letztens: versuche kleineren k
    k2 = max(2, k // 2)
    w, V = eigsh(L, k=k2, which="SM", tol=1e-2, maxiter=maxiter)
    return w.real, V.real


def _pick_nontrivial_evecs(
    evals: np.ndarray, evecs: np.ndarray, dims: int, zero_eps=1e-9
):
    """
    Wähle die ersten 'dims' **nicht-trivialen** Eigenvektoren (λ > ~0).
    Padding mit Nullen, falls Komponente zu klein.
    """
    idx_sorted = np.argsort(evals)  # klein -> groß
    evals = evals[idx_sorted]
    evecs = evecs[:, idx_sorted]

    # Null-Eigenwerte überspringen (für jede zusammenhängende Komponente ≥ 1)
    nontrivial = np.where(evals > zero_eps)[0]
    if len(nontrivial) == 0:
        # degenerierter Fall (isolierte Knoten): gebe nur Nullen zurück
        return np.zeros((evecs.shape[0], dims), dtype=np.float32)

    take = nontrivial[:dims]
    chosen = evecs[:, take]

    # ggf. auf Ziel-Dimension padden
    if chosen.shape[1] < dims:
        out = np.zeros((chosen.shape[0], dims), dtype=np.float32)
        out[:, : chosen.shape[1]] = chosen.astype(np.float32)
        return out
    return chosen.astype(np.float32)


def compute_spectral_embeddings_standalone(
    edgelist_path: str,
    output_path: str,
    dimensions: int = 64,
    comp_small_threshold: int = 50,
    max_k_cap: int = 128,
    tol: float = 1e-3,
    maxiter: int = 2000,
):
    """
    Berechnet **Spectral (Laplacian) Embeddings** komponentenweise und speichert Parquet.
    - Sparse-Eigenproblem (eigsh) statt dichtem eigh
    - Nicht-triviale Eigenvektoren (λ > 0) als Embedding
    - Kleinstkomponenten werden effizient behandelt
    """
    t0 = time.time()
    print("Computing spectral embeddings (component-wise, sparse eigsh) ...")

    # 1) Graph laden
    print("[SP] Loading edge list ...")
    df_e = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(df_e, source="txId1", target="txId2")
    # Wir arbeiten mit **ungerichtetem** Graphen für Laplacian-Einbettungen
    G = nx.relabel_nodes(G, lambda n: str(n)).to_undirected()
    print(
        f"[SP] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
    )

    # 2) Komponenten bestimmen (absteigend nach Größe)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    print(
        f"[SP] Connected components: {len(comps)} (largest={len(comps[0]):,})"
    )

    # Container für Ergebnisblöcke
    parts = []

    for ci, nodes in enumerate(comps, start=1):
        nodes = list(nodes)
        n = len(nodes)
        print(f"[SP] Comp {ci:3d} | n={n:5d} | ", end="")

        if n == 1:
            # isolierter Knoten -> Nullvektor
            emb = np.zeros((1, dimensions), dtype=np.float32)
            df_part = pd.DataFrame(
                emb, columns=[f"emb_spectral_{i}" for i in range(dimensions)]
            )
            df_part["txId"] = nodes
            parts.append(df_part)
            print("isolated")
            continue

        H = G.subgraph(nodes)
        # Kleinstkomponenten behandeln wir schnell: Nullvektor + Degree-Spur
        if n <= comp_small_threshold:
            deg = np.array(
                [H.degree(v) for v in nodes], dtype=np.float32
            ).reshape(-1, 1)
            # normiertes Degree-Feature + Nullen auffüllen
            if deg.max() > 0:
                deg = deg / (deg.max() + 1e-9)
            emb = np.zeros((n, dimensions), dtype=np.float32)
            emb[:, 0:1] = deg
            df_part = pd.DataFrame(
                emb, columns=[f"emb_spectral_{i}" for i in range(dimensions)]
            )
            df_part["txId"] = nodes
            parts.append(df_part)
            print("small-fast")
            continue

        # 3) Laplacian (sparse)
        L = _normalized_laplacian_sparse(H)

        # Anzahl gewünschter EVecs: dims + evtl. etwas Puffer (aber deckeln)
        want_k = min(dimensions + 4, max_k_cap, n - 1)
        print(f"LARGE-sparse: eigsh(k={want_k}) ...", flush=True)
        t_comp = time.time()
        try:
            evals, evecs = _eigs_smallest(
                L, k=want_k, tol=tol, maxiter=maxiter
            )
        except Exception as e:
            # sehr selten: reduziere k und versuche erneut
            want_k2 = max(2, min(dimensions + 2, n - 1))
            print(f"  retry with k={want_k2} due to: {e}")
            evals, evecs = _eigs_smallest(
                L, k=want_k2, tol=1e-2, maxiter=maxiter
            )

        # 4) Nicht-triviale EVs auswählen und auf dims bringen
        emb = _pick_nontrivial_evecs(evals, evecs, dims=dimensions)

        # Ergebnis zusammenbauen (Knotenreihenfolge = nodes)
        df_part = pd.DataFrame(
            emb, columns=[f"emb_spectral_{i}" for i in range(dimensions)]
        )
        df_part["txId"] = nodes
        parts.append(df_part)
        print(f"done in {time.time() - t_comp:.1f}s")

    # 5) Alle Komponenten zusammenführen
    df_all = pd.concat(parts, axis=0, ignore_index=True)
    df_all = df_all[
        ["txId"] + [c for c in df_all.columns if c.startswith("emb_spectral_")]
    ]

    # 6) Speichern
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    print(
        f"[SP] Saved: {output_path}  (rows={len(df_all):,}, cols={df_all.shape[1]})"
    )

    # Meta-Datei
    meta = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "n_components": int(len(comps)),
        "largest_component": int(len(comps[0])),
        "dimensions": int(dimensions),
        "comp_small_threshold": int(comp_small_threshold),
        "max_k_cap": int(max_k_cap),
        "tol": float(tol),
        "maxiter": int(maxiter),
        "runtime_sec": round(time.time() - t0, 2),
    }
    meta_path = output_path.replace(".parquet", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SP] Meta: {meta_path}")

    return df_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute spectral embeddings for Elliptic "
        "(full graph OR edge-drop variants)."
    )

    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed (25, 50). If omitted, the full graph is used.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Alternative to --drop_rate."
            "Expects a file elliptic_txs_edgelist_<variant>.csv in ../EllipticDataSet/."
        ),
    )

    args = parser.parse_args()

    # Build EDGELIST PATH + OUTPUT SUFFIX
    if args.variant is not None:
        suffix = f"_{args.variant}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
    elif args.drop_rate is None:
        suffix = ""
        edgelist_path = "../EllipticDataSet/elliptic_txs_edgelist.csv"
    else:
        suffix = f"_{args.drop_rate}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"

    output_path = f"../artifacts/spectral_embeddings{suffix}.parquet"

    # Logging
    print("=" * 70)
    print("SPECTRAL EMBEDDINGS  — FINAL VERSION")
    if args.variant is not None:
        print(f"Mode: TARGETED DROP VARIANT (variant={args.variant})")
    elif args.drop_rate is None:
        print("Mode: FULL GRAPH")
    else:
        print(f"Mode: EDGE-DROP (drop_rate={args.drop_rate}%)")
    print(f"Input Edgelist: {edgelist_path}")
    print(f"Output Parquet: {output_path}")
    print("=" * 70)

    # Execute computation
    compute_spectral_embeddings_standalone(
        edgelist_path,
        output_path,
        dimensions=64,
        comp_small_threshold=50,
        max_k_cap=128,
        tol=1e-3,
        maxiter=2000,
    )
