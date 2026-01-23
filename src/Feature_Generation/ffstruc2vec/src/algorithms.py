# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.random import choice
from time import time
from collections import deque

from utils import (
    saveVariableOnDisk,
    restoreVariableFromDisk,
    isPickle
)


def generate_parameters_random_walk(workers):
    logging.info("Loading distances_nets from disk...")

    sum_weights = {}
    amount_edges = {}

    layer = 0
    while isPickle("distances_nets_weights-layer-" + str(layer)):
        logging.info("Executing layer {}...".format(layer))
        weights = restoreVariableFromDisk("distances_nets_weights-layer-" + str(layer))

        for k, list_weights in weights.items():
            if layer not in sum_weights:
                sum_weights[layer] = 0
            if layer not in amount_edges:
                amount_edges[layer] = 0

            for w in list_weights:
                sum_weights[layer] += w
                amount_edges[layer] += 1

        logging.info("Layer {} executed.".format(layer))
        layer += 1

    average_weight = {}
    for layer in sum_weights.keys():
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]

    logging.info("Saving average_weights on disk...")
    saveVariableOnDisk(average_weight, "average_weight")

    amount_neighbours = {}

    layer = 0
    while isPickle("distances_nets_weights-layer-" + str(layer)):
        logging.info("Executing layer {}...".format(layer))
        weights = restoreVariableFromDisk("distances_nets_weights-layer-" + str(layer))

        amount_neighbours[layer] = {}

        for k, list_weights in weights.items():
            cont_neighbours = 0
            for w in list_weights:
                if w > average_weight[layer]:
                    cont_neighbours += 1
            amount_neighbours[layer][k] = cont_neighbours

        logging.info("Layer {} executed.".format(layer))
        layer += 1

    logging.info("Saving amount_neighbours on disk...")
    saveVariableOnDisk(amount_neighbours, "amount_neighbours")


def chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer):
    v_list = graphs[layer][v]

    idx = alias_draw(alias_method_j[layer][v], alias_method_q[layer][v])
    v = v_list[idx]

    return v


def chooseNeighbor_flatten(
    v, graphs, weights, probability_distribution, softmax_factor
):
    v_list = graphs[0][v]
    w_list = weights[v]

    if probability_distribution == 0:
        # linear probability distribution
        summe = sum([x for x in w_list])
        neighbors_prob_dist = [x / summe for x in w_list]
    elif probability_distribution == 1:
        # softmax probability distribution
        w_list_factored = [softmax_factor * x for x in w_list]
        logging.info(w_list_factored)
        summe = sum([np.exp(x) for x in w_list_factored])
        while math.isinf(summe):
            w_list_factored = [x / 2 for x in w_list_factored]
            summe = sum([np.exp(x) for x in w_list_factored])
        neighbors_prob_dist = [np.exp(x) / summe for x in w_list_factored]

    v = choice(v_list, 1, p=neighbors_prob_dist)

    return v[0]


def exec_random_walk(
    graphs,
    alias_method_j,
    alias_method_q,
    v,
    walk_length,
    amount_neighbours,
    probability_distribution,
    softmax_factor,
    weights,
):
    original_v = v
    t0 = time()

    path = deque()
    path.append(v)

    while len(path) < walk_length:
        v = chooseNeighbor_flatten(
            v, graphs, weights, probability_distribution, softmax_factor
        )
        path.append(v)

    t1 = time()
    logging.info("RW - vertex {}. Time : {}s".format(original_v, (t1 - t0)))

    return path


def exec_ramdom_walks_for_chunck(
    vertices,
    graphs,
    alias_method_j,
    alias_method_q,
    walk_length,
    amount_neighbours,
    probability_distribution,
    softmax_factor,
):
    walks = deque()
    weights = restoreVariableFromDisk("distances_nets_weights_layer_comb")
    for v in vertices:
        walks.append(
            exec_random_walk(
                graphs,
                alias_method_j,
                alias_method_q,
                v,
                walk_length,
                amount_neighbours,
                probability_distribution,
                softmax_factor,
                weights,
            )
        )
    return walks


def generate_random_walks_large_graphs(
    num_walks,
    walk_length,
    workers,
    vertices,
    probability_distribution,
    softmax_factor,
):
    logging.info("Loading distances_nets from disk...")

    graphs = restoreVariableFromDisk("distances_nets_graphs")
    alias_method_j = restoreVariableFromDisk("nets_weights_alias_method_j")
    alias_method_q = restoreVariableFromDisk("nets_weights_alias_method_q")
    amount_neighbours = restoreVariableFromDisk("amount_neighbours")

    logging.info("Creating RWs...")
    t0 = time()

    walks = deque()

    for walk_iter in range(num_walks):
        random.shuffle(vertices)
        logging.info("Execution iteration {} ...".format(walk_iter))
        walk = exec_ramdom_walks_for_chunck(
            vertices,
            graphs,
            alias_method_j,
            alias_method_q,
            walk_length,
            amount_neighbours,
            probability_distribution,
            softmax_factor,
        )
        walks.extend(walk)
        logging.info("Iteration {} executed.".format(walk_iter))

    t1 = time()
    logging.info("RWs created. Time : {}m".format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)


def generate_random_walks(
    num_walks,
    walk_length,
    workers,
    vertices,
    probability_distribution,
    softmax_factor,
):
    logging.info("Loading distances_nets on disk...")

    graphs = restoreVariableFromDisk("distances_nets_graphs")
    logging.info(
        "... finished: graphs = restoreVariableFromDisk('distances_nets_graphs')"
    )

    alias_method_j = restoreVariableFromDisk("nets_weights_alias_method_j")
    logging.info(
        "... finished: alias_method_j = restoreVariableFromDisk('nets_weights_alias_method_j')"
    )

    alias_method_q = restoreVariableFromDisk("nets_weights_alias_method_q")
    logging.info(
        "... finished: alias_method_q = restoreVariableFromDisk('nets_weights_alias_method_q')"
    )

    amount_neighbours = restoreVariableFromDisk("amount_neighbours")
    logging.info(
        "... finished: amount_neighbours = restoreVariableFromDisk('amount_neighbours')"
    )

    logging.info("Creating RWs...")

    t0 = time()

    walks = deque()

    if workers > num_walks:
        workers = num_walks

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            job = executor.submit(
                exec_ramdom_walks_for_chunck,
                vertices,
                graphs,
                alias_method_j,
                alias_method_q,
                walk_length,
                amount_neighbours,
                probability_distribution,
                softmax_factor,
            )
            futures[job] = walk_iter

        logging.info("Receiving results...")

        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            logging.info("Iteration {} executedxxx.".format(r))
            walks.extend(walk)
            del futures[job]

    t1 = time()
    logging.info("RWs created. Time: {}m".format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")

    save_random_walks(walks)


def save_random_walks(walks):
    with open("random_walks.txt", "w") as file:
        for walk in walks:
            line = ""
            for v in walk:
                line += str(v) + " "
            line += "\n"
            file.write(line)
    return


def prob_moveup(amount_neighbours):
    x = math.log(amount_neighbours + math.e)
    p = x / (x + 1)
    return p


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
