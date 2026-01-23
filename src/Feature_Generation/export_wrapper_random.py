from preprocessing_export import CFG, main
import argparse

# EINSTELLUNGEN (Random Split)
VAL_SIZE = 0.20
TEST_SIZE = 0.20
WEEK_MAX = 49

VARIANT = None  # Default None = voller Graph

BASE_TAG = "base93"

# GRAPH INDICATORS
ADD_DEGREE_CENTRALITY = False
ADD_IN_OUT_DEGREE = False
ADD_PAGERANK = False
ADD_BETWEENNESS_CENTRALITY = False
ADD_EIGENVECTOR_CENTRALITY = False
ADD_CLOSENESS_CENTRALITY = False
ADD_CLUSTERING_COEFFICIENT = False
ADD_SQUARE_CLUSTERING = False
ADD_CORE_NUMBER = False
ADD_TRIANGLES = False
ADD_COMMUNITY_LOUVAIN = False
ADD_COMMUNITY_LEIDEN = False
ADD_COMMUNITY_INFOMAP = False

# SPECTRAL EMBEDDINGS
ADD_SPECTRAL = False

# PROXIMITY EMBEDDINGS
ADD_NODE2VEC_BALANCED = False
ADD_NODE2VEC_DFS = False
ADD_NODE2VEC_BFS = False
ADD_DEEPWALK = False

# STRUCTURAL EMBEDDINGS
ADD_FFSTRUC2VEC = False
ADD_GRAPHWAVE = False
ADD_ROLE2VEC = False

# GNN EMBEDDINGS
ADD_GCN_EMBEDDINGS = False
ADD_GAT_EMBEDDINGS = False
ADD_GCL_EMBEDDINGS = False


FEATURE_FLAGS = {
    # Graph Indicators
    "deg_cent": "ADD_DEGREE_CENTRALITY",
    "inout": "ADD_IN_OUT_DEGREE",
    "pagerank": "ADD_PAGERANK",
    "betweenness": "ADD_BETWEENNESS_CENTRALITY",
    "eigenvector": "ADD_EIGENVECTOR_CENTRALITY",
    "closeness": "ADD_CLOSENESS_CENTRALITY",
    "clustering": "ADD_CLUSTERING_COEFFICIENT",
    "square_clustering": "ADD_SQUARE_CLUSTERING",
    "core": "ADD_CORE_NUMBER",
    "triangles": "ADD_TRIANGLES",
    "louvain": "ADD_COMMUNITY_LOUVAIN",
    "leiden": "ADD_COMMUNITY_LEIDEN",
    "infomap": "ADD_COMMUNITY_INFOMAP",
    # Spectral
    "spectral": "ADD_SPECTRAL",
    # Structural
    "ffstruc2vec": "ADD_FFSTRUC2VEC",
    "graphwave": "ADD_GRAPHWAVE",
    "role2vec": "ADD_ROLE2VEC",
    # GNN
    "gcn": "ADD_GCN_EMBEDDINGS",
    "gat": "ADD_GAT_EMBEDDINGS",
    "gcl": "ADD_GCL_EMBEDDINGS",
    # Proximity
    "node2vec_bal": "ADD_NODE2VEC_BALANCED",
    "node2vec_dfs": "ADD_NODE2VEC_DFS",
    "node2vec_bfs": "ADD_NODE2VEC_BFS",
    "deepwalk": "ADD_DEEPWALK",
}


def run_export(split_seed: int):

    indicator_tags = []
    if ADD_DEGREE_CENTRALITY:
        indicator_tags.append("I1")
    if ADD_IN_OUT_DEGREE:
        indicator_tags.append("I2")
    if ADD_PAGERANK:
        indicator_tags.append("I3")
    if ADD_BETWEENNESS_CENTRALITY:
        indicator_tags.append("I4")
    if ADD_EIGENVECTOR_CENTRALITY:
        indicator_tags.append("I5")
    if ADD_CLOSENESS_CENTRALITY:
        indicator_tags.append("I6")
    if ADD_CLUSTERING_COEFFICIENT:
        indicator_tags.append("I7")
    if ADD_SQUARE_CLUSTERING:
        indicator_tags.append("I8")
    if ADD_CORE_NUMBER:
        indicator_tags.append("I9")
    if ADD_TRIANGLES:
        indicator_tags.append("I10")
    if ADD_COMMUNITY_LOUVAIN:
        indicator_tags.append("I11")
    if ADD_COMMUNITY_LEIDEN:
        indicator_tags.append("I12")
    if ADD_COMMUNITY_INFOMAP:
        indicator_tags.append("I13")
    INDICATOR_TAG = "+".join(indicator_tags) if indicator_tags else "noInd"

    SPECTRAL_TAG = "1Spectral" if ADD_SPECTRAL else "0Spectral"

    FFSTRUC2VEC_TAG = "1ff2Vec" if ADD_FFSTRUC2VEC else "0ff2Vec"
    GRAPHWAVE_TAG = "1Graphwave" if ADD_GRAPHWAVE else "0Graphwave"
    ROLE2VEC_TAG = "1Role2Vec" if ADD_ROLE2VEC else "0Role2Vec"

    GCN_TAG = "1GCN" if ADD_GCN_EMBEDDINGS else "0GCN"
    GAT_TAG = "1GAT" if ADD_GAT_EMBEDDINGS else "0GAT"
    GCL_TAG = "1GCL" if ADD_GCL_EMBEDDINGS else "0GCL"

    proximity_tags = []
    if ADD_NODE2VEC_BALANCED:
        proximity_tags.append("N2VBal")
    if ADD_NODE2VEC_DFS:
        proximity_tags.append("N2VDFS")
    if ADD_NODE2VEC_BFS:
        proximity_tags.append("N2VBFS")
    if ADD_DEEPWALK:
        proximity_tags.append("DeepWalk")
    PROXIMITY_TAG = "+".join(proximity_tags) if proximity_tags else "0Prox"

    EDGE_TAG = f"_edgesVar_{VARIANT}"

    SPLIT_TAG = f"_splitseed{split_seed}"

    FEATURE_TAG = (
        f"{BASE_TAG}+"
        f"{INDICATOR_TAG}+{SPECTRAL_TAG}+{FFSTRUC2VEC_TAG}+{GRAPHWAVE_TAG}+{ROLE2VEC_TAG}"
        f"+{GCN_TAG}+{GAT_TAG}+{GCL_TAG}+{PROXIMITY_TAG}"
        f"{EDGE_TAG}{SPLIT_TAG}"
        f"_random_val{int(VAL_SIZE*100)}_test{int(TEST_SIZE*100)}"
    )

    CFG.update(
        {
            "split_mode": "random",
            "feature_tag": FEATURE_TAG,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "week_max": WEEK_MAX,
            # Graph / Embedding Infrastruktur (wie in temporal 49)
            "edgelist_file": "EllipticDataSet/elliptic_txs_edgelist.csv",
            "directed_graph": True,
            "gi_name_prefix": "gi_",
            "gi_subset": None,
            "emb_name_prefix": "emb_",
            "embedding_dimensions": 64,
            # GRAPH INDICATORS
            "add_degree_centrality": ADD_DEGREE_CENTRALITY,
            "add_in_out_degree": ADD_IN_OUT_DEGREE,
            "add_pagerank": ADD_PAGERANK,
            "add_betweenness_centrality": ADD_BETWEENNESS_CENTRALITY,
            "add_eigenvector_centrality": ADD_EIGENVECTOR_CENTRALITY,
            "add_closeness_centrality": ADD_CLOSENESS_CENTRALITY,
            "add_clustering_coefficient": ADD_CLUSTERING_COEFFICIENT,
            "add_square_clustering": ADD_SQUARE_CLUSTERING,
            "add_core_number": ADD_CORE_NUMBER,
            "add_triangles": ADD_TRIANGLES,
            "add_community_louvain": ADD_COMMUNITY_LOUVAIN,
            "add_community_leiden": ADD_COMMUNITY_LEIDEN,
            "add_community_infomap": ADD_COMMUNITY_INFOMAP,
            # PROXIMITY EMBEDDINGS
            "add_node2vec_balanced": ADD_NODE2VEC_BALANCED,
            "add_node2vec_dfs": ADD_NODE2VEC_DFS,
            "add_node2vec_bfs": ADD_NODE2VEC_BFS,
            "add_deepwalk": ADD_DEEPWALK,
            # SPECTRAL EMBEDDINGS
            "add_spectral_embeddings": ADD_SPECTRAL,
            # STRUCTURAL EMBEDDINGS
            "add_ffstruc2vec_embeddings": ADD_FFSTRUC2VEC,
            "add_graphwave_embeddings": ADD_GRAPHWAVE,
            "add_role2vec_embeddings": ADD_ROLE2VEC,
            # GNNS
            "add_gcn_embeddings": ADD_GCN_EMBEDDINGS,
            "add_gat_embeddings": ADD_GAT_EMBEDDINGS,
            "add_gcl_embeddings": ADD_GCL_EMBEDDINGS,
            # Edge-Drop:
            "variant": VARIANT,
            "random_state": split_seed,
        }
    )

    print(f"=== RANDOM export: {FEATURE_TAG} ===")
    main()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random-Seed für den Random-Split (wird auch im Feature-Tag codiert).",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Edge-Drop (z.B. 25 für 25% Kantenentfernung)",
    )

    # ersetze in export_wrapper_random.py den enable-Argumentblock durch:
    parser.add_argument(
        "--enable",
        nargs="+",
        default=None,
        help="Ein oder mehrere Graphfeatures",
    )

    args = parser.parse_args()

    # ersetze die enable-Logik darunter durch:
    if args.enable is not None:
        from_this_script = globals()

        # alles aus
        for name in list(from_this_script.keys()):
            if name.startswith("ADD_"):
                from_this_script[name] = False

        # Tokens normalisieren: erlaubt sowohl Leerzeichen- als auch Komma-getrennt
        tokens = []
        for t in args.enable:
            tokens.extend(
                [x.strip().lower() for x in t.split(",") if x.strip()]
            )

        # "base" heißt: keine Graphsignale aktivieren (also: einfach alles aus lassen)
        tokens = [t for t in tokens if t != "base"]

        # jetzt alle gewünschten aktivieren
        unknown = [t for t in tokens if t not in FEATURE_FLAGS]
        if unknown:
            raise ValueError(
                f"Unbekannte Features: {unknown}. Bekannt: {sorted(FEATURE_FLAGS.keys())}"
            )

        for key in tokens:
            from_this_script[FEATURE_FLAGS[key]] = True

    VARIANT = args.variant

    run_export(args.seed)
