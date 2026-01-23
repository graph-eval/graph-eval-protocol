# compute_proximity_embeddings_final_deterministic.py
import os
import time
import json
import warnings
import hashlib
import sys
import shutil

# BLAS/NumPy THREAD LIMITS ZUERST SETZEN
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# PYTHONHASHSEED - WICHTIG: Muss in der Shell/Umgebung gesetzt werden!
# Beispiel: set PYTHONHASHSEED=42 (Windows) oder export PYTHONHASHSEED=42 (Linux)

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import random
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _stable_int(s: str, mod: int = 100000) -> int:
    """STABILER Hash der über Prozessgrenzen hinweg konsistent ist."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % mod


def set_perfect_determinism(seed=42):
    """Setzt ALLE Seeds und Thread-Limits für perfekte Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED sollte bereits in der Umgebung gesetzt sein
    # Wir prüfen das hier und warnen falls nicht
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        print(f"WARNING: PYTHONHASHSEED not set to {seed} in environment!")
        print(
            f"Current PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'not set')}"
        )
        print(
            f"Please run: set PYTHONHASHSEED={seed} (Windows) or export PYTHONHASHSEED={seed} (Linux)"
        )

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def _safe_temp_dir_cleanup(temp_dir: str):
    """Sicheres Löschen des Temp-Verzeichnisses mit Fehlerbehandlung."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Temp directory cleaned up: {temp_dir}")
    except Exception as e:
        print(f"Could not clean up temp directory {temp_dir}: {e}")


def compute_node2vec_embeddings_final_deterministic(
    edgelist_path: str,
    output_path: str,
    dimensions: int = 64,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window_size: int = 10,
    workers: int = 1,
    seed: int = 42,
    prefix: str = "emb_n2v_bal",
    all_txids: list = None,
    epochs: int = 5,
    negative: int = 10,  # MEHR NEGATIVE SAMPLES FÜR BESSERE QUALITÄT
    cleanup_temp: bool = True,  # Steuerung der Temp-Bereinigung
):
    """100% DETERMINISTISCHE Node2Vec Berechnung mit allen Fixes."""

    print(f"Computing Node2Vec {prefix} (100% DETERMINISTIC)...")
    t0 = time.time()

    # PERFEKTEN DETERMINISMUS AKTIVIEREN
    set_perfect_determinism(seed)

    # TEMP-VERZEICHNIS SICHER ANLEGEN
    temp_dir = f"../artifacts/temp_{prefix}_{seed}"
    try:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Temp directory created: {temp_dir}")
    except Exception as e:
        print(f"Could not create temp directory {temp_dir}: {e}")
        temp_dir = None

    # Graph laden
    print("Loading full graph...")
    df_edges = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(df_edges, source="txId1", target="txId2")
    G = nx.relabel_nodes(G, lambda n: str(n))

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)

    print(
        f"Graph: {n_nodes:,} nodes, {n_edges:,} edges, {n_components} components"
    )
    print(
        f"100% deterministic: seed={seed}, workers={workers}, epochs={epochs}, negative={negative}"
    )
    print(f"BLAS/NumPy threads: {os.environ.get('OMP_NUM_THREADS', '1')}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'not set')}")
    if temp_dir:
        print(f"Temp directory: {temp_dir}")

    try:
        # Node2Vec mit deterministischen Einstellungen
        node2vec = Node2Vec(
            G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            seed=seed,
            temp_folder=temp_dir if temp_dir else None,
        )

        print("Training Node2Vec model (100% deterministic)...")

        # Word2Vec Training mit OPTIMIERTEN PARAMETERN
        model = node2vec.fit(
            window=window_size,
            min_count=1,
            batch_words=4,
            seed=seed,
            hs=0,
            negative=negative,
            epochs=epochs,
            compute_loss=True,
        )

        # ROBUSTE UND DETERMINISTISCHE EMBEDDING-EXTRAKTION
        print("  Extracting embeddings (100% deterministic)...")
        kv = model.wv
        graph_nodes = sorted([str(n) for n in G.nodes()])

        embeddings = []
        for node in tqdm(graph_nodes, desc=f"Extracting {prefix}"):
            if node in kv.key_to_index:
                vec = kv.get_vector(node, norm=False).astype(
                    np.float32, copy=False
                )
            else:
                vec = np.zeros(dimensions, dtype=np.float32)
            embeddings.append(vec)

        X = np.array(embeddings)

        # DataFrame mit eindeutigen Spalten und SPEICHEROPTIMIERTEN DATENTYPEN
        columns = [f"{prefix}_{i}" for i in range(dimensions)]
        df_embeddings = pd.DataFrame(X, columns=columns)
        df_embeddings["txId"] = graph_nodes

        # SPEICHEROPTIMIERUNG
        df_embeddings[columns] = df_embeddings[columns].astype("float32")

        # VOLLSTÄNDIGE ABDECKUNG mit ROBUSTER COVERAGE-BERECHNUNG
        if all_txids is not None:
            all_txids_str = sorted([str(tx) for tx in all_txids])
            print(
                f"  Ensuring coverage for {len(all_txids_str)} requested txIds..."
            )

            df_complete = pd.DataFrame({"txId": all_txids_str})
            df_complete = df_complete.merge(
                df_embeddings, on="txId", how="left"
            )
            df_complete[columns] = (
                df_complete[columns].fillna(0.0).astype("float32")
            )
            df_embeddings = df_complete

            # ROBUSTE COVERAGE-BERECHNUNG
            coverage = (df_embeddings[columns].abs().sum(axis=1) > 0).sum()
            print(
                f"Robust coverage: {coverage}/{len(all_txids_str)} nodes have embeddings"
            )

        # Speichern
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_embeddings.to_parquet(output_path, index=False)

        # TEMP-VERZEICHNIS OPTIONAL AUFRÄUMEN
        if cleanup_temp and temp_dir:
            _safe_temp_dir_cleanup(temp_dir)

        # Metadaten
        meta = {
            "method": "node2vec_final_deterministic",
            "prefix": prefix,
            "seed": seed,
            "deterministic": True,
            "workers": workers,
            "epochs": epochs,
            "negative_samples": negative,
            "blas_threads": os.environ.get("OMP_NUM_THREADS"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "temp_directory_used": temp_dir is not None,
            "temp_directory_cleaned": cleanup_temp and temp_dir is not None,
            "dimensions": dimensions,
            "parameters": {
                "p": p,
                "q": q,
                "walk_length": walk_length,
                "num_walks": num_walks,
                "window_size": window_size,
            },
            "graph_info": {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_components": n_components,
            },
            "coverage": {
                "nodes_with_embeddings": n_nodes,
                "requested_nodes": len(all_txids) if all_txids else n_nodes,
                "actual_coverage": len(df_embeddings),
                "robust_coverage": int(coverage) if all_txids else n_nodes,
            },
            "runtime_seconds": round(time.time() - t0, 2),
        }

        meta_path = output_path.replace(".parquet", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{prefix} 100% DETERMINISTIC embeddings saved: {output_path}")
        print(
            f"Shape: {df_embeddings.shape}, Memory: {df_embeddings.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )
        print(
            f"Seed: {seed}, Workers: {workers}, Epochs: {epochs}, Negative: {negative}"
        )
        if temp_dir and cleanup_temp:
            print(f"Temp directory cleaned up")
        print(f"Runtime: {meta['runtime_seconds']}s")

        return df_embeddings

    except Exception as e:
        # TEMP-VERZEICHNIS AUCH IM FEHLERFALL AUFRÄUMEN
        if temp_dir:
            _safe_temp_dir_cleanup(temp_dir)
        print(f"Node2Vec {prefix} failed: {e}")
        return None


def compute_deepwalk_embeddings_final_deterministic(
    edgelist_path: str,
    output_path: str,
    dimensions: int = 64,
    walk_length: int = 80,
    num_walks: int = 10,
    window_size: int = 10,
    workers: int = 1,
    seed: int = 42,
    prefix: str = "emb_dw",
    all_txids: list = None,
    epochs: int = 5,
    negative: int = 10,
):
    """100% DETERMINISTISCHE DeepWalk Berechnung mit STABILEM HASH."""

    print(f"Computing DeepWalk {prefix} (100% DETERMINISTIC)...")
    t0 = time.time()

    # PERFEKTEN DETERMINISMUS AKTIVIEREN
    set_perfect_determinism(seed)

    # Graph laden
    print("Loading full graph...")
    df_edges = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(df_edges, source="txId1", target="txId2")
    G = nx.relabel_nodes(G, lambda n: str(n))

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)

    print(
        f"Graph: {n_nodes:,} nodes, {n_edges:,} edges, {n_components} components"
    )
    print(
        f"100% deterministic: seed={seed}, workers={workers}, epochs={epochs}, negative={negative}"
    )
    print(f"BLAS/NumPy threads: {os.environ.get('OMP_NUM_THREADS', '1')}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'not set')}")

    try:
        # DETERMINISTISCHE RANDOM WALKS MIT STABILEM HASH
        print("  Generating deterministic random walks (stable hash)...")
        walks = []
        nodes_list = sorted(list(G.nodes()))

        rng = random.Random(seed)

        for walk_iter in tqdm(range(num_walks), desc="Generating walks"):
            shuffled_nodes = nodes_list.copy()
            rng.shuffle(shuffled_nodes)

            for node in shuffled_nodes:
                walk = [node]
                current_node = node

                # STABILER HASH statt Python's hash()
                walk_seed = (
                    seed + walk_iter * 100000 + _stable_int(str(node), 100000)
                )
                rng_walk = random.Random(walk_seed)

                for step in range(walk_length - 1):
                    neighbors = sorted(list(G.neighbors(current_node)))
                    if neighbors:
                        current_node = rng_walk.choice(neighbors)
                        walk.append(current_node)
                    else:
                        break
                walks.append([str(n) for n in walk])

        # DETERMINISTISCHES WORD2VEC MIT OPTIMIERTEN PARAMETERN
        print("  Training DeepWalk model (100% deterministic)...")
        model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=window_size,
            min_count=1,
            sg=1,
            workers=workers,
            seed=seed,
            hs=0,
            negative=negative,
            epochs=epochs,
            compute_loss=True,
            sorted_vocab=1,
            batch_words=10000,
        )

        # ROBUSTE UND DETERMINISTISCHE EMBEDDING-EXTRAKTION
        print("  Extracting embeddings (100% deterministic)...")
        kv = model.wv
        graph_nodes = sorted([str(n) for n in G.nodes()])

        embeddings = []
        for node in tqdm(graph_nodes, desc=f"Extracting {prefix}"):
            if node in kv.key_to_index:
                vec = kv.get_vector(node).astype(np.float32, copy=False)
            else:
                vec = np.zeros(dimensions, dtype=np.float32)
            embeddings.append(vec)

        X = np.array(embeddings)

        # DataFrame mit SPEICHEROPTIMIERTEN DATENTYPEN
        columns = [f"{prefix}_{i}" for i in range(dimensions)]
        df_embeddings = pd.DataFrame(X, columns=columns)
        df_embeddings["txId"] = graph_nodes

        # SPEICHEROPTIMIERUNG
        df_embeddings[columns] = df_embeddings[columns].astype("float32")

        # VOLLSTÄNDIGE ABDECKUNG mit ROBUSTER COVERAGE-BERECHNUNG
        if all_txids is not None:
            all_txids_str = sorted([str(tx) for tx in all_txids])
            print(
                f"  Ensuring coverage for {len(all_txids_str)} requested txIds..."
            )

            df_complete = pd.DataFrame({"txId": all_txids_str})
            df_complete = df_complete.merge(
                df_embeddings, on="txId", how="left"
            )
            df_complete[columns] = (
                df_complete[columns].fillna(0.0).astype("float32")
            )
            df_embeddings = df_complete

            # ROBUSTE COVERAGE-BERECHNUNG
            coverage = (df_embeddings[columns].abs().sum(axis=1) > 0).sum()
            print(
                f" Robust coverage: {coverage}/{len(all_txids_str)} nodes have embeddings"
            )

        # Speichern
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_embeddings.to_parquet(output_path, index=False)

        # Metadaten
        meta = {
            "method": "deepwalk_final_deterministic",
            "prefix": prefix,
            "seed": seed,
            "deterministic": True,
            "workers": workers,
            "epochs": epochs,
            "negative_samples": negative,
            "blas_threads": os.environ.get("OMP_NUM_THREADS"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "stable_hash_used": True,  # Dokumentiert den stabilen Hash
            "dimensions": dimensions,
            "parameters": {
                "walk_length": walk_length,
                "num_walks": num_walks,
                "window_size": window_size,
            },
            "graph_info": {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_components": n_components,
            },
            "coverage": {
                "nodes_with_embeddings": n_nodes,
                "requested_nodes": len(all_txids) if all_txids else n_nodes,
                "actual_coverage": len(df_embeddings),
                "robust_coverage": int(coverage) if all_txids else n_nodes,
            },
            "runtime_seconds": round(time.time() - t0, 2),
        }

        meta_path = output_path.replace(".parquet", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{prefix} 100% DETERMINISTIC embeddings saved: {output_path}")
        print(
            f"Shape: {df_embeddings.shape}, Memory: {df_embeddings.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )
        print(
            f"Seed: {seed}, Workers: {workers}, Epochs: {epochs}, Negative: {negative}"
        )
        print(f"Stable hash used for walk generation")
        print(f"Runtime: {meta['runtime_seconds']}s")

        return df_embeddings

    except Exception as e:
        print(f"DeepWalk {prefix} failed: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="100% deterministic proximity embeddings for Elliptic "
        "(full graph OR edge-drop variants)."
    )
    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed (25, 50). If omitted: full graph.",
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

    # Build edgelist path + output suffix
    if args.variant is not None:
        suffix = f"_{args.variant}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
    elif args.drop_rate is None:
        suffix = ""
        edgelist_path = "../EllipticDataSet/elliptic_txs_edgelist.csv"
    else:
        suffix = f"_{args.drop_rate}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"

    MASTER_SEED = 42

    print("=" * 70)
    print("100% DETERMINISTIC PROXIMITY EMBEDDINGS - FINAL VERSION")
    if args.variant is not None:
        print(f"Mode: TARGETED DROP VARIANT (variant={args.variant})")
    elif args.drop_rate is None:
        print("Mode: FULL GRAPH")
    else:
        print(f"Mode: EDGE-DROP VARIANT (drop_rate={args.drop_rate}%)")

    print(f"Input edgelist: {edgelist_path}")
    print(f"Master Seed: {MASTER_SEED}")
    print(f"BLAS/NumPy threads: {os.environ.get('OMP_NUM_THREADS', '1')}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'NOT SET!')}")
    print("=" * 70)

    if os.environ.get("PYTHONHASHSEED") != str(MASTER_SEED):
        print("CRITICAL: PYTHONHASHSEED not set correctly!")
        print(f"   Windows CMD: set PYTHONHASHSEED={MASTER_SEED}")
        print(f"   PowerShell:  $env:PYTHONHASHSEED='{MASTER_SEED}'")
        print(f"   Linux/Mac:   export PYTHONHASHSEED={MASTER_SEED}")
        sys.exit(1)

    # === Node2Vec Balanced ===
    compute_node2vec_embeddings_final_deterministic(
        edgelist_path,
        f"../artifacts/node2vec_balanced_embeddings{suffix}.parquet",
        p=1.0,
        q=1.0,
        prefix="emb_n2v_bal",
        seed=MASTER_SEED,
        workers=1,
        epochs=5,
        negative=10,
        cleanup_temp=True,
    )

    # === Node2Vec DFS ===
    compute_node2vec_embeddings_final_deterministic(
        edgelist_path,
        f"../artifacts/node2vec_dfs_embeddings{suffix}.parquet",
        p=0.25,
        q=4.0,
        prefix="emb_n2v_dfs",
        seed=MASTER_SEED,
        workers=1,
        epochs=5,
        negative=10,
        cleanup_temp=True,
    )

    # === Node2Vec BFS ===
    compute_node2vec_embeddings_final_deterministic(
        edgelist_path,
        f"../artifacts/node2vec_bfs_embeddings{suffix}.parquet",
        p=4.0,
        q=0.25,
        prefix="emb_n2v_bfs",
        seed=MASTER_SEED,
        workers=1,
        epochs=5,
        negative=10,
        cleanup_temp=True,
    )

    # === DeepWalk ===
    compute_deepwalk_embeddings_final_deterministic(
        edgelist_path,
        f"../artifacts/deepwalk_embeddings{suffix}.parquet",
        prefix="emb_dw",
        seed=MASTER_SEED,
        workers=1,
        epochs=5,
        negative=10,
    )

    print("=" * 70)
    print("ALL 100% DETERMINISTIC COMPUTATIONS COMPLETED!")
    print(f"Suffix: '{suffix}'")
    print("Stable hash & determinism enabled")
    print("Temp directories cleaned up")
    print("=" * 70)
