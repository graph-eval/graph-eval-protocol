import itertools
import subprocess
import sys

PYTHON_EXE = sys.executable

# SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# #SEEDS = [42]

# FEATURES = [
#     "base",

#     # Graph Indicators
#     "deg_cent", "inout", "pagerank", "betweenness", "eigenvector",
#     "closeness", "clustering", "square_clustering", "core", "triangles",
#     "louvain", "leiden", "infomap",

#     # Spectral
#     "spectral",

#     # Structural Embeddings
#     "ffstruc2vec", "graphwave", "role2vec",

#     # Proximity Embeddings
#     "node2vec_bal", "node2vec_dfs", "node2vec_bfs", "deepwalk",

#     # GNN Embeddings
#     "gcn", "gat", "gcl"
# ]

# VARIANTS = ["0", "25", "50"]

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
VARIANTS = ["50"]

FEATURE_SETS = [
    [
        "deg_cent",
        "inout",
        "pagerank",
        "betweenness",
        "eigenvector",
        "closeness",
        "clustering",
        "square_clustering",
        "core",
        "triangles",
        "louvain",
        "leiden",
        "infomap",
        "spectral",
        "ffstruc2vec",
        "graphwave",
        "role2vec",
        "node2vec_bal",
        "node2vec_dfs",
        "node2vec_bfs",
        "deepwalk",
        "gcn",
        "gat",
        "gcl",
    ]
]


def main():
    if VARIANTS == []:
        print(
            "Nothing to do. Choose variants via '--variants' for required edge drops as input."
        )
        sys.exit(1)

    for feature_set, seed, variant in itertools.product(
        FEATURE_SETS, SEEDS, VARIANTS
    ):

        cmd = [PYTHON_EXE, "export_wrapper_random.py", "--seed", str(seed)]

        # mehrere --enable Tokens
        cmd += ["--enable", *feature_set]

        if variant is not None:
            cmd += ["--variant", str(variant)]

        print(
            f"\n=== Starte Run: Features={feature_set}, Seed={seed}, Variant={variant} ==="
        )
        print("CMD:", " ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(
                f"Fehler bei Features={feature_set}, Seed={seed}, Variant={variant}"
            )
        else:
            print(
                f"Fertig: Features={feature_set}, Seed={seed}, Variant={variant}"
            )


if __name__ == "__main__":
    main()
