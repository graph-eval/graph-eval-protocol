#!/usr/bin/python
# -*- coding: utf-8 -*-

import graph
import logging
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier  # type: ignore
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from hyperopt import hp, Trials, fmin, tpe
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
# from visualize_embeddings import (
#     visualize_embeddings_from_model,
#     visualize_embeddings_from_model_barbell,
# )
import argparse
import ffstruc2vec
import pickle
import random
import numpy as np
#import pandas as pd
import math
import matplotlib.pyplot as plt

plt.style.use("ggplot")


#logging.basicConfig(
#    filename="ffstruc2vec.log",
#    filemode="w",
#    level=logging.INFO,
#    format="%(asctime)s %(message)s",
#)

logging.basicConfig(
    level=logging.INFO,   # oder WARNING
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # <– nur Konsole, keine Datei
)


def parse_args():
    """
    Parses the ffstruc2vec arguments.
    """

    logging.info("Parsing arguments")

    parser = argparse.ArgumentParser(description="Run ffstruc2vec.")

    parser.add_argument(
        "--input", nargs="?", default="graph/karate.edgelist", help="Input graph path"
    )

    parser.add_argument(
        "--output", nargs="?", default="emb/karate.emb", help="Embeddings path"
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help="Number of dimensions. Default is 128.",
    )

    parser.add_argument(
        "--walk-length",
        type=int,
        default=80,
        help="Length of walk per source. Default is 80.",
    )

    parser.add_argument(
        "--num-walks",
        type=int,
        default=10,
        help="Number of walks per source. Default is 10.",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Context size for optimization. Default is 10.",
    )

    parser.add_argument(
        "--until_layer",
        type=int,
        default=3,
        help="Calculation until the layer. Default is 3.",
    )

    parser.add_argument("--iter", default=5, type=int, help="Number of epochs in SGD")

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers. Default is 8.",
    )

    parser.add_argument(
        "--weighted",
        dest="weighted",
        action="store_true",
        help=("Boolean specifying (un)weighted. " "Default is unweighted."),
    )
    parser.add_argument("--unweighted", dest="unweighted", action="store_false")
    parser.set_defaults(weighted=False)

    parser.add_argument(
        "--directed",
        dest="directed",
        action="store_true",
        help="Graph is (un)directed. Default is undirected.",
    )
    parser.add_argument("--undirected", dest="undirected", action="store_false")
    parser.set_defaults(directed=False)

    parser.add_argument("--OPT1", default=False, type=bool, help="optimization 1")
    parser.add_argument("--OPT2", default=False, type=bool, help="optimization 2")
    parser.add_argument("--OPT3", default=False, type=bool, help="optimization 3")

    parser.add_argument(
        "--weight_layer_0",
        default=1,
        type=float,
        help=(
            "Weight of layer 0 in case of the usage of the "
            "flattened graph. Default is 1."
        ),
    )

    parser.add_argument(
        "--weight_layer_1",
        default=1,
        type=float,
        help=(
            "Weight of layer 1 in case of the usage of the "
            "flattened graph. Default is 1."
        ),
    )

    parser.add_argument(
        "--weight_layer_2",
        default=1,
        type=float,
        help=(
            "Weight of layer 2 in case of the usage of the "
            "flattened graph. Default is 1."
        ),
    )

    parser.add_argument(
        "--weight_layer_3",
        default=1,
        type=float,
        help=(
            "Weight of layer 3 in case of the usage of the "
            "flattened graph. Default is 1."
        ),
    )

    parser.add_argument(
        "--weight_layer_4",
        default=1,
        type=float,
        help=(
            "Weight of layer 4 in case of the usage of the "
            "flattened graph. Default is 1."
        ),
    )

    parser.add_argument(
        "--weight_layer_5",
        default=1,
        type=float,
        help=(
            "Weight of layer 5 in case of the usage of the "
            "flattened graph. Default is 1."
        ),
    )

    parser.add_argument(
        "--weight_transformer",
        default=math.e,
        type=float,
        help=(
            "Weight transformer for weight calculation out "
            "of distances. Default is Euler's number e."
        ),
    )

    parser.add_argument(
        "--cost_function",
        default=1,
        type=int,
        help=(
            "Cost function to be used to evaluate "
            "structural similarity of nodes "
            "(0: DTW, 1: mean&varianz, 2: med&varianz). "
            "Default is 1."
        ),
    )

    parser.add_argument(
        "--cost_calc",
        default=0,
        type=int,
        help=(
            "Cost calculatin to be used to evaluate "
            "structural similarity of nodes "
            "(0: max(a, b) / min(a, b), 1: max(a, b) - min(a, b) "
            "Default is 0."
        ),
    )

    parser.add_argument(
        "--mean_factor",
        default=0.5,
        type=float,
        help=(
            "Weighting factor for mean and var if mean&var "
            "distance is used. Default is 0,5."
        ),
    )

    parser.add_argument(
        "--med_factor",
        default=0.5,
        type=float,
        help=(
            "Weighting factor for median and var if med&var "
            "distance is used. Default is 0,5."
        ),
    )

    parser.add_argument(
        "--probability_distribution",
        default=0,
        type=int,
        help=(
            "Probability distribution for choosing next "
            "node during Deep Walk (0: linear distribution, "
            "1: softmax). Default is 0."
        ),
    )

    parser.add_argument(
        "--softmax_factor",
        default=1,
        type=float,
        # xxx    parser.add_argument('--softmax_factor', default=1, type=int,
        help=(
            "Applied factor in case of the usage of the "
            "flattened graph with softmax probability "
            "distribution. Default is 1."
        ),
    )

    parser.add_argument(
        "--path_labels",
        default="",
        type=str,
        help=(
            "Path to labels for the application of an "
            'classification algorithm. Default is "".'
        ),
    )

    parser.add_argument(
        "--active_feature",
        default=0,
        type=int,
        help=(
            "Feature of nodes to be used to calculate "
            "the distances between the nodes. "
            "(0: node degree, 1: pagerank, "
            "2: closeness_centrality, 3: betweenness_centrality, "
            "4: eigenvector_centrality, 5: core_number, "
            "6: clustering). "
            "Default is 0."
        ),
    )

    parser.add_argument(
        "--pagerank_scale",
        default=1,
        type=float,
        help=("Applied factor in case of the usage of the " "pagerank. Default is 1."),
    )

    parser.add_argument(
        "--method",
        default=3,
        type=int,
        help=(
            "Method to be applied "
            "(0: Hyperopt Classification, 1: Simple "
            "Classification, 2: Hyperopt Clustering, "
            "3: Only embedding) "
            "Default is 3."
        ),
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for NumPy and Python's random module (default: 42)",
    )

    return parser.parse_args()


def read_graph(args):
    """
    Reads the input network.
    """
    logging.info(" - Loading graph...")
    G = graph.load_edgelist(args.input, undirected=True)
    logging.info(" - Graph loaded.")
    return G


def learn_embeddings(args):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    logging.info("Initializing creation of the representations...")
    walks = LineSentence("random_walks.txt")
    model = Word2Vec(
        walks,
        vector_size=args.dimensions,
        window=args.window_size,
        min_count=0,
        hs=1,
        sg=1,
        workers=args.workers,
        epochs=args.iter,
    )
    model.wv.save_word2vec_format(
        args.output
        + "_cost_"
        + str(args.cost_function)
        + "_costcalc_"
        + str(args.cost_calc)
        + "_ps_"
        + str(round(args.pagerank_scale, 3))
        + "_meanf_"
        + str(round(args.mean_factor, 3))
        + "_w_"
        + str(round(args.weight_layer_0, 3))
        + "_"
        + str(round(args.weight_layer_1, 3))
        + "_"
        + str(round(args.weight_layer_2, 3))
        + "_"
        + str(round(args.weight_layer_3, 3))
        + "_"
        + str(round(args.weight_layer_4, 3))
        + "_"
        + str(round(args.weight_layer_5, 3))
        + "_active_"
        + str(args.active_feature)
        + "_prob_"
        + str(args.probability_distribution)
        + "_sf_"
        + str(round(args.softmax_factor, 3))
        + "_"
        + datetime.now().strftime("%d%m%Y_%H%M%S")
        + ".emb"
    )
    logging.info("Representations created.")

    return model.wv


def exec_ffstruc2vec(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    if args.OPT3:
        until_layer = args.until_layer
    else:
        until_layer = None

    G = read_graph(args)
    G = ffstruc2vec.Graph(G, args.directed, args.workers, untilLayer=until_layer)

    if args.OPT1:
        G.preprocess_neighbors_with_bfs_compact()
    else:
        G.preprocess_neighbors_with_bfs(args.active_feature)

    if args.OPT2:
        G.create_vectors()
        G.calc_distances(
            active_feature=args.active_feature,
            mean_factor=args.mean_factor,
            med_factor=args.med_factor,
            cost_calc=args.cost_calc,
            cost_function=args.cost_function,
            compactDegree=args.OPT1
        )
    else:
        G.calc_distances_all_vertices(
            active_feature=args.active_feature,
            mean_factor=args.mean_factor,
            med_factor=args.med_factor,
            cost_calc=args.cost_calc,
            cost_function=args.cost_function,
            compactDegree=args.OPT1,
        )

    G.create_distances_network(
        until_layer=args.until_layer,
        weight_layer_0=args.weight_layer_0,
        weight_layer_1=args.weight_layer_1,
        weight_layer_2=args.weight_layer_2,
        weight_layer_3=args.weight_layer_3,
        weight_layer_4=args.weight_layer_4,
        weight_layer_5=args.weight_layer_5,
        active_feature=args.active_feature,
        pagerank_scale=args.pagerank_scale,
        weight_transformer=args.weight_transformer,
    )
    logging.info(" G.create_distances_network(...) finished")

    G.preprocess_parameters_random_walk()
    logging.info(" G.preprocess_parameters_random_walk() finished")

    G.simulate_walks(
        args.num_walks,
        args.walk_length,
        args.probability_distribution,
        args.softmax_factor,
    )
    logging.info(" G.simulate_walks(...) finished")

    return G


def main(args, params=None):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    logging.info("exec_ffstruc2vec")
    exec_ffstruc2vec(args)

    logging.info("learn embeddings")
    model = learn_embeddings(args)

    # logging.info("visualize embeddings")
    # visualize_embeddings_from_model(model, args)
    # if "barbell" in args.input:
    #     visualize_embeddings_from_model_barbell(model, args)

    # if args.method == 2:
    #     vectors = model.vectors
    #     if params["clustering"]["type"] == "kmeans":
    #         kmeans = KMeans(n_clusters=6, random_state=0).fit(vectors)
    #         score = kmeans.score(vectors)
    #         logging.info(f"...score of clustering = {score}")
    #     return -score

    # elif args.method == 0:
    #     # embedding als features für die knoten nehmen
    #     keys = model.index_to_key
    #     vectors = model.vectors

    #     labels = pd.read_csv(args.path_labels, sep=" ")
    #     # labels = pd.read_csv("graph/labels-brazil-airports.txt", sep=" ")
    #     keys_df = pd.DataFrame(keys).rename(columns={0: "node"}).astype(int)
    #     labels_ordered = pd.merge(keys_df, labels, on="node", how="left")

    #     X = vectors
    #     y = labels_ordered["label"]

    #     scores = []
    #     for i in range(10):
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=0.2, random_state=i
    #         )

    #         if params["classifier"]["type"] == "lr":
    #             clf = LogisticRegressionCV(
    #                 cv=10,
    #                 multi_class="ovr",
    #                 solver="liblinear",
    #                 penalty=params["classifier"]["lr_penalty"],
    #                 random_state=i,
    #             )
    #         elif params["classifier"]["type"] == "svc":
    #             clf = SVC(
    #                 C=params["classifier"]["svc_C"],
    #                 kernel=params["classifier"]["svc_kernel"],
    #                 max_iter=100,
    #             )
    #         # elif params["classifier"]["type"] == "xgb":
    #         #     clf = XGBClassifier(
    #         #         max_depth=int(params["classifier"]["xgb_max_depth"]),
    #         #         max_leaves=int(params["classifier"]["xgb_max_leaves"]),
    #         #         min_samples_leaf=int(params["classifier"]["xgb_min_samples_leaf"]),
    #         #         booster=params["classifier"]["xgb_booster"],
    #         #         n_estimators=int(params["classifier"]["xgb_n_estimators"]),
    #         #     )
    #         elif params["classifier"]["type"] == "gb":
    #             clf = GradientBoostingClassifier(
    #                 max_depth=int(params["classifier"]["gb_max_depth"]),
    #                 max_leaf_nodes=int(params["classifier"]["gb_max_leaf_nodes"]),
    #                 min_samples_leaf=int(params["classifier"]["gb_min_samples_leaf"]),
    #                 n_estimators=int(params["classifier"]["gb_n_estimators"]),
    #             )
    #         elif params["classifier"]["type"] == "rf":
    #             clf = RandomForestClassifier(
    #                 max_depth=int(params["classifier"]["rf_max_depth"]),
    #                 max_leaf_nodes=int(params["classifier"]["rf_max_leaf_nodes"]),
    #                 min_samples_leaf=int(params["classifier"]["rf_min_samples_leaf"]),
    #                 n_estimators=int(params["classifier"]["rf_n_estimators"]),
    #             )
    #         else:  # params['classifier']['type'] == 'mlp':
    #             hidden_layer = []
    #             hidden_layer.append(
    #                 int(params["classifier"]["mlp_hidden_layer_sizes_1"])
    #             )
    #             hidden_layer.append(
    #                 int(params["classifier"]["mlp_hidden_layer_sizes_2"])
    #             )
    #             hidden_layer = tuple(hidden_layer)
    #             clf = MLPClassifier(hidden_layer_sizes=hidden_layer)

    #         clf.fit(X_train, y_train)
    #         scores.append(clf.score(X_test, y_test))

    #     logging.info(
    #         f"...average score of all random train-test-split runs = {sum(scores) / len(scores)}"
    #     )
    #     return -(sum(scores) / len(scores))
    # elif args.method == 1:
    #     # embedding als features für die knoten nehmen
    #     keys = model.index_to_key
    #     vectors = model.vectors

    #     labels = pd.read_csv(args.path_labels, sep=" ")
    #     # labels = pd.read_csv("graph/labels-brazil-airports.txt", sep=" ")
    #     keys_df = pd.DataFrame(keys).rename(columns={0: "node"}).astype(int)
    #     labels_ordered = pd.merge(keys_df, labels, on="node", how="left")

    #     X = vectors
    #     y = labels_ordered["label"]

    #     scores = []
    #     for i in range(10):
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=0.2, random_state=i
    #         )

    #         clf = LogisticRegressionCV(cv=10, multi_class="ovr", random_state=i)

    #         clf.fit(X_train, y_train)
    #         scores.append(clf.score(X_test, y_test))

    #     logging.info(
    #         f"...average score of all random train-test-split runs = {sum(scores) / len(scores)}"
    #     )
    #     return scores


def objective(params):
    args.dimensions = int(params["dimensions"])
    args.walk_length = int(params["walk-length"])
    args.num_walks = int(params["num-walks"])
    args.window_size = int(params["window-size"])
    #    args.until_layer = int(params['until_layer'])
    args.iter = int(params["iter"])
    args.cost_calc = int(params["cost_calc"])

    if params["active_feature"]["type"] == "node_degree":
        args.active_feature = 0
    elif params["active_feature"]["type"] == "pagerank":
        args.active_feature = 1
        args.pagerank_scale = params["active_feature"]["pagerank_scale"]
    elif params["active_feature"]["type"] == "closeness_centrality":
        args.active_feature = 2
    elif params["active_feature"]["type"] == "betweenness_centrality":
        args.active_feature = 3
    elif params["active_feature"]["type"] == "eigenvector_centrality":
        args.active_feature = 4
    elif params["active_feature"]["type"] == "core_number":
        args.active_feature = 5
    elif params["active_feature"]["type"] == "clustering":
        args.active_feature = 6

    if params["cost_function"]["type"] == "dtw":
        args.cost_function = 0
    elif params["cost_function"]["type"] == "mean&var":
        args.cost_function = 1
        args.mean_factor = params["cost_function"]["mean_factor"]
    else:  # params['cost_function']['type'] == 'med&var':
        args.cost_function = 2
        args.med_factor = params["cost_function"]["med_factor"]

    if params["probability_distribution"]["type"] == "linear":
        args.probability_distribution = 0
    else:  # params['probability_distribution']['type'] == 'softmax'
        args.probability_distribution = 1
        args.softmax_factor = params["probability_distribution"]["softmax_factor"]

    args.weight_layer_0 = params["weight_layer_0"]
    args.weight_layer_1 = params["weight_layer_1"]
    args.weight_layer_2 = params["weight_layer_2"]
    args.weight_layer_3 = params["weight_layer_3"]
    args.weight_layer_4 = params["weight_layer_4"]
    args.weight_layer_5 = params["weight_layer_5"]

    args.weight_transformer = params["weight_transformer"]

    score = main(args, params)

    return score


if __name__ == "__main__":
    args = parse_args()

    if args.method == 2:
        ##################################################
        # Optimization of hyperparameters using Hyperopt #
        ##################################################

        # Determination of the parameters to be optimised and the associated value ranges
        params = {
            "dimensions": hp.choice("dimensions", np.arange(64, 257, dtype=int)),
            "walk-length": hp.choice("walk-length", np.arange(60, 100, dtype=int)),
            "num-walks": hp.choice("num-walks", np.arange(5, 15, dtype=int)),
            "window-size": hp.choice("window-size", np.arange(3, 20, dtype=int)),
            "iter": hp.choice("iter", np.arange(4, 10, dtype=int)),
            "cost_calc": hp.choice("cost_calc", np.arange(0, 2, dtype=int)),
            "clustering": hp.choice(
                "clustering",
                [
                    {
                        "type": "kmeans",
                    }
                ],
            ),
        }

        params["active_feature"] = hp.choice(
            "active_feature",
            [
                {"type": "node_degree"},
                {
                    "type": "pagerank",
                    "pagerank_scale": hp.uniform("pagerank_scale", 0, 10),
                },
                {"type": "closeness_centrality"},
                {"type": "betweenness_centrality"},
                {"type": "eigenvector_centrality"},
                {"type": "core_number"},
                {"type": "clustering"},
            ],
        )
        params["cost_function"] = hp.choice(
            "cost_function",
            [
                {
                    "type": "dtw",
                },
                {
                    "type": "mean&var",
                    "mean_factor": hp.uniform("mean_factor", 0, 1),
                },
                {
                    "type": "med&var",
                    "med_factor": hp.uniform("med_factor", 0, 1),
                },
            ],
        )

        params["probability_distribution"] = hp.choice(
            "probability_distribution",
            [
                {
                    "type": "linear",
                },
                {
                    "type": "softmax",
                    "softmax_factor": hp.uniform("softmax_factor", 0.01, 200),
                },
            ],
        )
        params["weight_layer_0"] = hp.uniform("weight_layer_0", 0, 50)
        params["weight_layer_1"] = hp.uniform("weight_layer_1", 0, 50)
        params["weight_layer_2"] = hp.uniform("weight_layer_2", 0, 50)
        params["weight_layer_3"] = hp.uniform("weight_layer_3", 0, 50)
        params["weight_layer_4"] = hp.uniform("weight_layer_4", 0, 50)
        params["weight_layer_5"] = hp.uniform("weight_layer_5", 0, 50)

        params["weight_transformer"] = hp.uniform("weight_transformer", 1, 4)

        # Save the tested parameter combinations and the associated result in trials
        trials = Trials()

        # Optimisation with Hyperopt by determining the minimum
        best = fmin(
            fn=objective, space=params, trials=trials, algo=tpe.suggest, max_evals=1000
        )
        dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        results = {
            "best": best,
            "trials": trials,
            "best_test": min(trials.losses()),
            "args": args,
        }
        with open(f"optimization/results_{dt_string}.p", "wb") as output_file:
            pickle.dump(results, output_file)

    elif args.method == 0:
        ##################################################
        # Optimization of hyperparameters using Hyperopt #
        ##################################################

        # Determination of the parameters to be optimised and the associated value ranges
        params = {
            "dimensions": hp.choice("dimensions", np.arange(64, 257, dtype=int)),
            "walk-length": hp.choice("walk-length", np.arange(60, 100, dtype=int)),
            "num-walks": hp.choice("num-walks", np.arange(5, 15, dtype=int)),
            "window-size": hp.choice("window-size", np.arange(3, 20, dtype=int)),
            "iter": hp.choice("iter", np.arange(4, 10, dtype=int)),
            "cost_calc": hp.choice("cost_calc", np.arange(0, 2, dtype=int)),
            "classifier": hp.choice(
                "classifier",
                [
                    {"type": "lr", "lr_penalty": hp.choice("lr_penalty", ["l1", "l2"])},
                    {
                        "type": "svc",
                        "svc_C": hp.uniform("svc_C", 0.1, 1),
                        "svc_kernel": hp.choice(
                            "svc_kernel", ["rbf", "poly", "sigmoid"]
                        ),
                    },
                    {
                        "type": "xgb",
                        "xgb_max_depth": hp.choice(
                            "xgb_max_depth", np.arange(1, 5, dtype=int)
                        ),
                        "xgb_max_leaves": hp.choice(
                            "xgb_max_leaves", np.arange(10, 20, dtype=int)
                        ),
                        "xgb_min_samples_leaf": hp.choice(
                            "xgb_min_samples_leaf", np.arange(1, 5, dtype=int)
                        ),
                        "xgb_booster": hp.choice("xgb_booster", ["gbtree", "dart"]),
                        "xgb_n_estimators": hp.choice(
                            "xgb_n_estimators", np.arange(2, 20, dtype=int)
                        ),
                    },
                    {
                        "type": "gb",
                        "gb_max_depth": hp.choice(
                            "gb_max_depth", np.arange(1, 5, dtype=int)
                        ),
                        "gb_max_leaf_nodes": hp.choice(
                            "gb_max_leaf_nodes", np.arange(10, 20, dtype=int)
                        ),
                        "gb_min_samples_leaf": hp.choice(
                            "gb_min_samples_leaf", np.arange(1, 5, dtype=int)
                        ),
                        "gb_n_estimators": hp.choice(
                            "gb_n_estimators", np.arange(2, 20, dtype=int)
                        ),
                    },
                    {
                        "type": "rf",
                        "rf_max_depth": hp.choice(
                            "rf_max_depth", np.arange(1, 5, dtype=int)
                        ),
                        "rf_max_leaf_nodes": hp.choice(
                            "rf_max_leaf_nodes", np.arange(10, 20, dtype=int)
                        ),
                        "rf_min_samples_leaf": hp.choice(
                            "rf_min_samples_leaf", np.arange(1, 5, dtype=int)
                        ),
                        "rf_n_estimators": hp.choice(
                            "rf_n_estimators", np.arange(2, 20, dtype=int)
                        ),
                    },
                    {
                        "type": "mlp",
                        "mlp_hidden_layer_sizes_1": hp.choice(
                            "mlp_hidden_layer_sizes_1", np.arange(20, 100, dtype=int)
                        ),
                        "mlp_hidden_layer_sizes_2": hp.choice(
                            "mlp_hidden_layer_sizes_2", np.arange(5, 20, dtype=int)
                        ),
                    },
                ],
            ),
        }

        params["active_feature"] = hp.choice(
            "active_feature",
            [
                {"type": "node_degree"},
                {
                    "type": "pagerank",
                    "pagerank_scale": hp.uniform("pagerank_scale", 0, 10),
                },
                {"type": "closeness_centrality"},
                {"type": "betweenness_centrality"},
                {"type": "eigenvector_centrality"},
                {"type": "core_number"},
                {"type": "clustering"},
            ],
        )
        params["cost_function"] = hp.choice(
            "cost_function",
            [
                {
                    "type": "dtw",
                },
                {
                    "type": "mean&var",
                    "mean_factor": hp.uniform("mean_factor", 0, 1),
                },
                {
                    "type": "med&var",
                    "med_factor": hp.uniform("med_factor", 0, 1),
                },
            ],
        )

        params["probability_distribution"] = hp.choice(
            "probability_distribution",
            [
                {
                    "type": "linear",
                },
                {
                    "type": "softmax",
                    "softmax_factor": hp.uniform("softmax_factor", 0.01, 200),
                },
            ],
        )
        params["weight_layer_0"] = hp.uniform("weight_layer_0", 0, 50)
        params["weight_layer_1"] = hp.uniform("weight_layer_1", 0, 50)
        params["weight_layer_2"] = hp.uniform("weight_layer_2", 0, 50)
        params["weight_layer_3"] = hp.uniform("weight_layer_3", 0, 50)
        params["weight_layer_4"] = hp.uniform("weight_layer_4", 0, 50)
        params["weight_layer_5"] = hp.uniform("weight_layer_5", 0, 50)

        params["weight_transformer"] = hp.uniform("weight_transformer", 1, 4)

        # Save the tested parameter combinations and the associated result in trials
        trials = Trials()

        # Optimisation with Hyperopt by determining the minimum
        best = fmin(
            fn=objective, space=params, trials=trials, algo=tpe.suggest, max_evals=100
        )
        dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        results = {
            "best": best,
            "trials": trials,
            "best_test": min(trials.losses()),
            "args": args,
        }
        with open(f"optimization/results_{dt_string}.p", "wb") as output_file:
            pickle.dump(results, output_file)
    elif args.method == 1:
        scores = main(args)
        dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
        results = {"result": sum(scores) / len(scores), "scores": scores, "args": args}
        with open(f"optimization/results_{dt_string}.p", "wb") as output_file:
            pickle.dump(results, output_file)
    else:
        main(args)
