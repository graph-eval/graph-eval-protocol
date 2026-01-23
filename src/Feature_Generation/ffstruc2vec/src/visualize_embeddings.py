# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import logging

from datetime import datetime
from sklearn.decomposition import PCA

plt.style.use("ggplot")


def visualize_embeddings_from_model_barbell(model, args):
    draw_boarder = False
    draw_node_names = False

    df_node_colors = pd.DataFrame(
        data=[
            [0, "#0127ff"],  # 0
            [1, "#0127ff"],  # 56
            [2, "#0127ff"],  # 65
            [3, "#0127ff"],  # 43
            [4, "#0127ff"],  # 28
            [5, "#0127ff"],  # 29
            [6, "#0127ff"],  # 8
            [7, "#0127ff"],  # 45
            [8, "#0127ff"],  # 10
            [9, "#83bc54"],  # 22
            [20, "#83bc54"],  # 20
            [21, "#0127ff"],  # 40
            [22, "#0127ff"],  # 53
            [23, "#0127ff"],  # 49
            [24, "#0127ff"],  # 54
            [25, "#0127ff"],  # 63
            [26, "#0127ff"],  # 5
            [27, "#0127ff"],  # 11
            [28, "#0127ff"],  # 31
            [29, "#0127ff"],  # 18
            [10, "#b9391c"],  # 36
            [19, "#b9391c"],  # 50
            [11, "#66a8f3"],  # 7
            [18, "#66a8f3"],  # 58
            [12, "#d8bd47"],  # 27
            [17, "#d8bd47"],  # 9
            [13, "#7752a1"],  # 62
            [16, "#7752a1"],  # 6
            [14, "#dddee0"],  # 15
            [15, "#dddee0"],  # 14
        ],
        columns=["node", "color"],
    )

    keys = model.index_to_key

    keys_df = pd.DataFrame(keys).rename(columns={0: "node"}).astype(int)
    colors_ordered = pd.merge(keys_df, df_node_colors, on="node", how="left")

    vectors = model.vectors

    twodim = PCA().fit_transform(vectors)[:, :2]

    plt.figure(figsize=(19, 13))
    plt.rcParams["font.size"] = "34"

    ax = plt.gca()

    plt.rc("grid", linestyle="-", linewidth=3, color="#f0f0f0")
    plt.scatter(
        twodim[:, 0],
        twodim[:, 1],
        s=600,
        edgecolors="k",
        c=colors_ordered["color"],
        zorder=3,
    )
    plt.grid(visible=True, zorder=0)

    ax.spines["top"].set_linewidth(0)
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["right"].set_linewidth(0)
    ax.tick_params(axis="x", colors="#818181")
    ax.tick_params(axis="y", colors="#818181")

    if draw_boarder:
        ax.spines["top"].set_linewidth(2)
        ax.spines["top"].set_color("#999999")
        ax.spines["right"].set_linewidth(2)
        ax.spines["right"].set_color("#999999")

    if draw_node_names:
        black_list = [67, 55, 37, 7, 57, 1, 26, 12, 24, 35]

        for word, (x, y) in zip(keys, twodim):
            if int(word) in black_list:
                color = "black"
            else:
                color = "white"
            if int(word) < 10:
                plt.text(x - 0.015, y - 0.012, s=word, fontsize=15, c=color)
            else:
                plt.text(x - 0.024, y - 0.012, s=word, fontsize=15, c=color)

    import matplotlib.pylab as pylab

    params = {
        "xtick.labelsize": "30",
        "ytick.labelsize": "30",
    }
    pylab.rcParams.update(params)

    plt.savefig(
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
        + "_barbell"
        + ".png"
    )

    # plt.savefig(r"C:\Users\mario\Desktop\FOM\PHD\Python\Node_embedding\barbell_embedding_ffstruc2vec.png")


def visualize_embeddings_from_model(model, args):
    keys = model.index_to_key
    vectors = model.vectors

    if args.method >= 2:
        pass
    else:
        labels = pd.read_csv(args.path_labels, sep=" ")
        keys_df = pd.DataFrame(keys).rename(columns={0: "node"}).astype(int)
        labels_ordered = pd.merge(keys_df, labels, on="node", how="left")

    twodim = PCA().fit_transform(vectors)[:, :2]
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 8})

    if args.method >= 2:
        plt.scatter(twodim[:, 0], twodim[:, 1], s=30, edgecolors="k", c="r")
    else:
        plt.scatter(
            twodim[:, 0], twodim[:, 1], s=30, edgecolors="k", c=labels_ordered["label"]
        )

    for word, (x, y) in zip(keys, twodim):
        plt.text(x + 0.00, y + 0.00, word)

    plt.savefig(
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
        + ".png"
    )
    logging.info(f"File saved: {args.output}.png")
