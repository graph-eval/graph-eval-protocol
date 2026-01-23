# -*- coding: utf-8 -*-
"""
GraphWave structural embeddings with full coverage (component-wise).

- Iterate over ALL connected components of the graph.
- Large components (>= LARGE_MIN_N): truncated spectral GraphWave with sparse eigsh (fast),
  batched heat-kernel row-moments (no N×N materialization).
- Small components (< LARGE_MIN_N): dense LAPACK eigh (exact), then identical wavelet pipeline.
- Concatenate features of all nodes and project to TARGET_DIM with a single, global
  GaussianRandomProjection (same input feature length for all comps -> consistent space).

Output
------
Parquet with columns: txId (str), emb_graphwave_{0..TARGET_DIM-1}
JSON sidecar with run metadata for reproducibility.

Requirements: numpy, pandas, networkx, scipy, scikit-learn, tqdm
"""

import os
import json
import time
import math
import warnings
from typing import Iterable, Tuple, List, Dict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from sklearn.random_projection import GaussianRandomProjection

warnings.filterwarnings("ignore")


# Default configuration
EDGELIST_PATH = "../EllipticDataSet/elliptic_txs_edgelist.csv"
OUTPUT_PARQUET = "../artifacts/graphwave_embeddings_components.parquet"
OUTPUT_METAJSON = OUTPUT_PARQUET.replace(".parquet", ".meta.json")

# GraphWave / pipeline params
TARGET_DIM = 64
N_SCALES = 6  # number of diffusion scales (taus)
TAU_MIN, TAU_MAX = 1e-2, 1e1  # log-spaced range
LARGE_MIN_N = 1000  # components with >= this many nodes use sparse eigsh
K_EIG_LARGE = 32  # number of smallest eigenpairs for large components
BATCH_SIZE_ROWS = 1024  # batch rows when forming heat-kernel row moments
EIG_TOL = 1e-3
EIG_MAXIT = 2000

RANDOM_SEED = 42


# Utilities
def _as_str_ids(seq: Iterable) -> List[str]:
    return list(map(lambda x: str(x), seq))


def _normalized_laplacian(G: nx.Graph) -> csr_matrix:
    """Return normalized Laplacian as CSR (float64)."""
    L = nx.normalized_laplacian_matrix(G)
    return L.asfptype().tocsr()


def _compute_eigendecomp(
    L: csr_matrix, n: int, large_min_n: int, k_eig_large: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition strategy:
    - if n >= large_min_n: truncated sparse eigsh for k smallest eigenpairs
    - else: dense eigh of full Laplacian (exact)
    Returns (evals, evecs) with shapes (k,) and (n,k) respectively.
    """
    if n >= large_min_n:
        k = max(2, min(k_eig_large, n - 1))  # safety
        vals, vecs = eigsh(L, k=k, which="SM", tol=EIG_TOL, maxiter=EIG_MAXIT)
        return vals.real, vecs.real
    else:
        # For small components a full dense eig is usually cheap and exact.
        A = L.toarray()
        vals, vecs = eigh(A)
        # Keep the first min(n-1, k_eig_large or all) to align feature cost with large comps
        k = min(
            A.shape[0] - 1, max(K_EIG_LARGE, 8)
        )  # keep at least 8 for stability
        vals = vals[:k]
        vecs = vecs[:, :k]
        return vals, vecs


def _build_taus(n_scales: int, tmin: float, tmax: float) -> np.ndarray:
    return np.logspace(math.log10(tmin), math.log10(tmax), n_scales)


def _row_moments(mat: np.ndarray) -> np.ndarray:
    """
    Compute simple statistical row-moments:
    mean, std, median, max, min. Returns shape (n_rows, 5).
    """
    # All along axis=1
    mu = mat.mean(axis=1)
    sd = mat.std(axis=1)
    med = np.median(mat, axis=1)
    mx = mat.max(axis=1)
    mn = mat.min(axis=1)
    return np.stack([mu, sd, med, mx, mn], axis=1)  # (n, 5)


def _heat_kernel_row_moments_batched(
    evecs: np.ndarray,
    evals: np.ndarray,
    taus: np.ndarray,
    batch_size_rows: int = 1024,
) -> np.ndarray:
    """
    For each tau, compute H_tau = V * diag(exp(-tau*lambda)) * V^T
    but never form the full N×N. We compute row blocks:

      H_rows = (V_rows * diag(exp(-tau*lambda))) @ V^T

    Then extract row-wise moments and stack across taus.

    Returns array of shape (n_nodes, len(taus)*5).
    """
    n, k = evecs.shape
    n_stats = 5
    out = np.zeros((n, len(taus) * n_stats), dtype=np.float64)

    # Precompute exp(-tau * lambda) for all taus,k
    exp_mat = np.exp(-np.outer(taus, evals))  # (n_scales, k)

    Vt = evecs.T  # (k, n)

    # Iterate rows in batches
    row_idx = np.arange(n)
    for start in range(0, n, batch_size_rows):
        end = min(start + batch_size_rows, n)
        Vr = evecs[start:end, :]  # (b, k)
        # Process each tau
        for t_idx, w in enumerate(exp_mat):  # w shape (k,)
            # scale columns of Vr by w: (b,k) * (k,) -> (b,k)
            VrW = Vr * w[np.newaxis, :]
            # (b,k) @ (k,n) -> (b,n)
            H_rows = VrW @ Vt
            # moments (b,5)
            M = _row_moments(H_rows)
            # place into output
            out[start:end, t_idx * n_stats : (t_idx + 1) * n_stats] = M
    return out  # (n, len(taus)*5)


# Main compute routine
def compute_graphwave_full_coverage(
    edgelist_path: str = EDGELIST_PATH,
    output_parquet: str = OUTPUT_PARQUET,
    output_metajson: str = OUTPUT_METAJSON,
    target_dim: int = TARGET_DIM,
    n_scales: int = N_SCALES,
    tau_min: float = TAU_MIN,
    tau_max: float = TAU_MAX,
    large_min_n: int = LARGE_MIN_N,
    k_eig_large: int = K_EIG_LARGE,
    batch_size_rows: int = BATCH_SIZE_ROWS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    t0 = time.time()
    rng = np.random.default_rng(random_seed)

    print("[GW] Loading edge list ...")
    edges = pd.read_csv(edgelist_path)
    # Expect columns txId1, txId2; cast to str once
    edges["txId1"] = edges["txId1"].astype(str)
    edges["txId2"] = edges["txId2"].astype(str)
    G = nx.from_pandas_edgelist(edges, source="txId1", target="txId2")
    print(
        f"[GW] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
    )

    # Connected components (undirected)
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    comps = sorted(comps, key=lambda g: g.number_of_nodes(), reverse=True)
    print(
        f"[GW] Connected components: {len(comps)} "
        f"(largest={comps[0].number_of_nodes():,})"
    )

    taus = _build_taus(n_scales, tau_min, tau_max)
    n_stats = 5
    feat_len = n_scales * n_stats  # same for all components

    # We'll collect features + ids for ALL nodes, then apply ONE global projector
    all_ids: List[str] = []
    all_feats: List[np.ndarray] = []

    for ci, H in enumerate(comps, start=1):
        n = H.number_of_nodes()
        ids = _as_str_ids(H.nodes())
        if n < 2:
            # Degenerate component: single node -> zeros
            feats = np.zeros((n, feat_len), dtype=np.float64)
            print(f"[GW] Comp {ci:>3} | n={n:>5} -> degenerate, zeros.")
        else:
            size_tag = "LARGE-sparse" if n >= large_min_n else "small-dense"
            print(
                f"[GW] Comp {ci:>3} | n={n:>5} | {size_tag}: eigendecomp ..."
            )

            L = _normalized_laplacian(H)
            try:
                evals, evecs = _compute_eigendecomp(
                    L, n, large_min_n=large_min_n, k_eig_large=k_eig_large
                )
                # Safety: drop any tiny negative evals due to num. noise
                evals = np.clip(evals, 0.0, None)
                # Row moments (batched) identical for both branches
                feats = _heat_kernel_row_moments_batched(
                    evecs, evals, taus, batch_size_rows=batch_size_rows
                )
            except Exception as e:
                # Fallback: zeros if eig fails (rare)
                print(
                    f"[GW]    !! eigendecomp failed ({str(e)}), filling zeros."
                )
                feats = np.zeros((n, feat_len), dtype=np.float64)

        all_ids.extend(ids)
        all_feats.append(feats)

    X_feat = np.vstack(all_feats)  # (N_total, feat_len)

    # Global, data-independent projection (same transform for all comps)
    projector = GaussianRandomProjection(
        n_components=target_dim, random_state=random_seed
    )
    X_emb = projector.fit_transform(X_feat)  # (N_total, target_dim)

    # Assemble dataframe
    emb_cols = [f"emb_graphwave_{i}" for i in range(target_dim)]
    df_out = pd.DataFrame(X_emb, columns=emb_cols)
    df_out.insert(0, "txId", np.array(all_ids, dtype=str))

    # Save
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_out.to_parquet(output_parquet, index=False)

    # Meta
    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "edgelist_path": edgelist_path,
        "output_parquet": output_parquet,
        "params": {
            "target_dim": target_dim,
            "n_scales": n_scales,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "taus": taus.tolist(),
            "large_min_n": large_min_n,
            "k_eig_large": k_eig_large,
            "batch_size_rows": batch_size_rows,
            "random_seed": random_seed,
            "eig_tol": EIG_TOL,
            "eig_maxit": EIG_MAXIT,
        },
        "graph": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_components": len(comps),
            "largest_component_size": comps[0].number_of_nodes(),
        },
        "features": {
            "feature_len_before_projection": int(feat_len),
            "stats_per_scale": ["mean", "std", "median", "max", "min"],
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    with open(output_metajson, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[GW] Saved embeddings: {output_parquet}  (rows={len(df_out):,})")
    print(f"[GW] Meta: {output_metajson}")
    return df_out


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GraphWave structural embeddings for Elliptic "
        "(full graph OR edge-drop variants)."
    )
    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed in the edgelist (e.g. 25, 50). "
        "If omitted, the full graph edgelist is used.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Alternative zur drop_rate-Variante. "
            "Erwartet dann eine Datei "
            "elliptic_txs_edgelist_<variant>.csv im EllipticDataSet-Ordner."
        ),
    )

    args = parser.parse_args()

    # Pfade & Suffix bestimmen
    if args.variant is not None:
        suffix = f"_{args.variant}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
    elif args.drop_rate is None:
        suffix = ""
        edgelist_path = EDGELIST_PATH
    else:
        suffix = f"_{args.drop_rate}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"

    output_parquet = (
        f"../artifacts/graphwave_embeddings_components{suffix}.parquet"
    )
    output_metajson = output_parquet.replace(".parquet", ".meta.json")

    # Logging
    print("=" * 70)
    print("GRAPHWAVE STRUCTURAL EMBEDDINGS")
    if args.variant is not None:
        print(f"Mode: TARGETED DROP VARIANT (variant={args.variant})")
    elif args.drop_rate is None:
        print("Mode: FULL GRAPH")
    else:
        print(f"Mode: EDGE-DROP VARIANT (drop_rate={args.drop_rate}%)")

    print(f"Input edgelist: {edgelist_path}")
    print(f"Output parquet: {output_parquet}")
    print(f"Output meta   : {output_metajson}")
    print("=" * 70)

    # Aufruf
    compute_graphwave_full_coverage(
        edgelist_path=edgelist_path,
        output_parquet=output_parquet,
        output_metajson=output_metajson,
        target_dim=TARGET_DIM,
        n_scales=N_SCALES,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        large_min_n=LARGE_MIN_N,
        k_eig_large=K_EIG_LARGE,
        batch_size_rows=BATCH_SIZE_ROWS,
        random_seed=RANDOM_SEED,
    )
