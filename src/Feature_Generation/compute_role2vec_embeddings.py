# compute_role2vec_embeddings.py
import pandas as pd
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import warnings
import random
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")


def weisfeiler_lehman_features(graph, iterations=3):
    """Berechnet Weisfeiler-Lehman Node-Labels für strukturelle Ähnlichkeit."""
    print("  Computing Weisfeiler-Lehman features...")

    # Initial labels basierend auf Degree
    node_labels = {node: str(graph.degree(node)) for node in graph.nodes()}

    for iteration in range(iterations):
        new_labels = {}
        for node in graph.nodes():
            # Sammle Nachbar-Labels
            neighbor_labels = sorted(
                [node_labels[neighbor] for neighbor in graph.neighbors(node)]
            )
            # Neuer Label: aktueller Label + sortierte Nachbar-Labels
            new_label = node_labels[node] + "_" + "_".join(neighbor_labels)
            # Hash für Kompaktheit
            new_labels[node] = str(hash(new_label) % 10**8)
        node_labels = new_labels

    return node_labels


def structural_random_walk(graph, wl_features, start_node, length=80):
    """Random walk der strukturell ähnliche Nodes bevorzugt."""
    walk = [start_node]
    current_node = start_node

    for _ in range(length - 1):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            # Wähle Nachbarn mit ähnlichen WL Features
            current_wl = wl_features[current_node]

            # Berechne strukturelle Ähnlichkeit zu Nachbarn
            similarities = []
            for neighbor in neighbors:
                # Ähnlichkeit basierend auf WL Feature-Overlap
                similarity = (
                    1.0 if wl_features[neighbor] == current_wl else 0.0
                )
                similarities.append(similarity)

            # Wähle zufällig unter strukturell ähnlichen Nachbarn
            similar_neighbors = [
                n for i, n in enumerate(neighbors) if similarities[i] > 0.5
            ]

            if similar_neighbors:
                current_node = random.choice(similar_neighbors)
            else:
                current_node = random.choice(neighbors)

            walk.append(current_node)
        else:
            break

    return [str(n) for n in walk]


def compute_role2vec_embeddings(
    edgelist_path, output_path, dimensions=64, num_walks=10, walk_length=80
):
    """Berechnet Role2Vec Embeddings für strukturelle Rollen."""

    print("Computing Role2Vec structural embeddings...")

    random.seed(42)
    np.random.seed(42)

    # Graph laden
    print("Loading graph...")
    df_edges = pd.read_csv(edgelist_path)
    G = nx.from_pandas_edgelist(df_edges, source="txId1", target="txId2")
    G = nx.relabel_nodes(G, lambda n: str(n))

    print(
        f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # Für große Graphen: Sampling
    if G.number_of_nodes() > 50000:
        print("  Large graph - sampling 50k nodes...")
        nodes = list(G.nodes())
        # Degree-basierte Auswahl
        degrees = [(node, G.degree(node)) for node in nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [node for node, deg in degrees[:50000]]
        G = G.subgraph(selected_nodes)
        print(f"  Using subgraph with {len(selected_nodes)} nodes")

    # Weisfeiler-Lehman Features berechnen
    wl_features = weisfeiler_lehman_features(G)

    # Strukturelle Random Walks generieren
    print("  Generating structural random walks...")
    walks = []
    nodes = list(G.nodes())

    for walk_idx in tqdm(range(num_walks), desc="Walks"):
        random.shuffle(nodes)
        for node in nodes:
            walk = structural_random_walk(
                G, wl_features, node, length=walk_length
            )
            walks.append(walk)

    print(f"  Generated {len(walks)} structural walks")

    # Word2Vec Modell trainieren
    print("  Training Word2Vec model...")
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=5,
        min_count=1,
        sg=1,  # Skip-gram
        workers=4,
        seed=42,
    )

    # Embeddings extrahieren
    print("  Extracting embeddings...")
    embeddings = []
    valid_nodes = []

    for node in G.nodes():
        if str(node) in model.wv:
            embeddings.append(model.wv[str(node)])
            valid_nodes.append(node)
        else:
            # Fallback: Null-Vektor
            embeddings.append(np.zeros(dimensions))
            valid_nodes.append(node)

    # DataFrame erstellen
    columns = [f"emb_role2vec_{i}" for i in range(dimensions)]
    df_embeddings = pd.DataFrame(embeddings, columns=columns)
    df_embeddings["txId"] = valid_nodes
    df_embeddings = df_embeddings[["txId"] + columns]

    # Als Parquet speichern
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_embeddings.to_parquet(output_path, index=False)

    print(f"Role2Vec embeddings saved: {output_path}")
    print(f"Shape: {df_embeddings.shape}")
    print("Role2Vec captures STRUCTURAL ROLES via Weisfeiler-Lehman!")

    return df_embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Role2Vec structural embeddings for Elliptic "
        "(full graph OR edge-drop variants OR targeted-drop variants)."
    )
    parser.add_argument(
        "--drop_rate",
        type=int,
        default=None,
        help="Percentage of edges removed (e.g. 25, 50)."
        "If omitted, the full graph edgelist is used (unless --variant is set).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Targeted-drop variant suffix."
        "Expects an input file '../EllipticDataSet/elliptic_txs_edgelist_<variant>.csv'.",
    )

    args = parser.parse_args()

    # Konsistenz: entweder drop_rate ODER variant (wie bei ffstruc2vec)
    if args.drop_rate is not None and args.variant is not None:
        raise ValueError(
            "Please specify either --drop_rate OR --variant, not both."
        )

    # Suffix & Pfade bestimmen (einheitlich)
    if args.variant is not None:
        suffix = f"_{args.variant}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
        mode_str = f"TARGETED-DROP (variant={args.variant})"
    elif args.drop_rate is None:
        suffix = ""
        edgelist_path = "../EllipticDataSet/elliptic_txs_edgelist.csv"
        mode_str = "FULL GRAPH"
    else:
        suffix = f"_{args.drop_rate}"
        edgelist_path = f"../EllipticDataSet/elliptic_txs_edgelist{suffix}.csv"
        mode_str = f"EDGE-DROP (drop_rate={args.drop_rate}%)"

    output_path = f"../artifacts/role2vec_embeddings{suffix}.parquet"

    # Logging
    print("=" * 70)
    print("ROLE2VEC STRUCTURAL EMBEDDINGS")
    print(f"Mode          : {mode_str}")
    print(f"Input edgelist: {edgelist_path}")
    print(f"Output parquet: {output_path}")
    print("=" * 70)

    # Berechnung starten
    compute_role2vec_embeddings(
        edgelist_path=edgelist_path, output_path=output_path, dimensions=64
    )
