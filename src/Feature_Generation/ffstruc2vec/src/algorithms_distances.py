# -*- coding: utf-8 -*-

import numpy as np
import math
import logging
import os

from time import time
from collections import deque
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.linalg import norm
from utils import (
    restoreVariableFromDisk,
    saveVariableOnDisk,
    partition,
    isPickle,
    returnPathFfstruc2vec
)


limiteDist = 20


def getDegreeListsVertices(
    g,
    pagerank,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    core_number,
    clustering,
    active_feature,
    vertices,
    calcUntilLayer,
):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getDegreeLists(
            g,
            pagerank,
            closeness_centrality,
            betweenness_centrality,
            eigenvector_centrality,
            core_number,
            clustering,
            active_feature,
            v,
            calcUntilLayer,
        )

    return degreeList


def getCompactDegreeListsVertices(g, vertices, maxDegree, calcUntilLayer):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getCompactDegreeLists(g, v, maxDegree, calcUntilLayer)

    return degreeList


def getCompactDegreeLists(g, root, maxDegree, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    length = {}

    # Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = len(g[vertex])
        if d not in length:
            length[d] = 0
        length[d] += 1

        for v in g[vertex]:
            if vetor_marcacao[v] == 0:
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1

        if timeToDepthIncrease == 0:
            list_d = []
            for degree, freq in length.items():
                list_d.append((degree, freq))
            list_d.sort(key=lambda x: x[0])
            listas[depth] = np.array(list_d, dtype=np.int32)

            length = {}

            if calcUntilLayer == depth:
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    t1 = time()
    logging.info("BFS vertex {}. Time: {}s".format(root, (t1 - t0)))

    return listas


def getDegreeLists(
    g,
    pagerank,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    core_number,
    clustering,
    active_feature,
    root,
    calcUntilLayer,
):
    t0 = time()
    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1

    el = deque()

    # Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1
        # xxx
        if active_feature == 0:
            el.append(len(g[vertex]))
        elif active_feature == 1:
            el.append(pagerank[vertex])
        elif active_feature == 2:
            el.append(closeness_centrality[vertex])
        elif active_feature == 3:
            el.append(betweenness_centrality[vertex])
        elif active_feature == 4:
            el.append(eigenvector_centrality[vertex])
        elif active_feature == 5:
            el.append(core_number[vertex])
        elif active_feature == 6:
            el.append(clustering[vertex])

        for v in g[vertex]:
            if vetor_marcacao[v] == 0:
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1

        if timeToDepthIncrease == 0:
            lp = np.array(el, dtype="float")
            lp = np.sort(lp)
            listas[depth] = lp
            el = deque()

            if calcUntilLayer == depth:
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    t1 = time()
    logging.info("BFS vertex {}. Time: {}s".format(root, (t1 - t0)))

    return listas


def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return (m / mi) - 1


def cost_abs(a, b):
    return max(a, b) - min(a, b)


def cost_min(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * min(a[1], b[1])


def cost_max(a, b):
    ep = 0.5
    m = max(a[0], b[0]) + ep
    mi = min(a[0], b[0]) + ep
    return ((m / mi) - 1) * max(a[1], b[1])


def average_of_lists(lists):
    result = []
    for i in range(len(lists[0])):
        sum = 0
        for j in range(len(lists)):
            sum += lists[j][i]
        result.append(sum / len(lists))
        # result.append(sum)
    return result


def subtract_lists(list1, list2):
    result = []
    for a, b in zip(list1, list2):
        result.append(a - b)
    return result


def euclid_to_mean_of_list(a, b):

    a_mean = average_of_lists(a)
    b_mean = average_of_lists(b)

    return norm(subtract_lists(a_mean, b_mean))


def mean_var(a, b, mean_factor, cost_calc):
    if cost_calc == 0:
        return (1 - mean_factor) * cost(np.std(a), np.std(b)) + mean_factor * cost(
            np.mean(a), np.mean(b)
        )
    else:
        return (1 - mean_factor) * cost_abs(
            np.std(a), np.std(b)
        ) + mean_factor * cost_abs(np.mean(a), np.mean(b))


def med_var(a, b, med_factor, cost_calc):
    if cost_calc == 0:
        return (1 - med_factor) * cost(np.std(a), np.std(b)) + med_factor * cost(
            np.median(a), np.median(b)
        )
    else:
        return (1 - med_factor) * cost_abs(
            np.std(a), np.std(b)
        ) + med_factor * cost_abs(np.median(a), np.median(b))


def preprocess_degreeLists():
    logging.info("Recovering degreeList from disk...")
    degreeList = restoreVariableFromDisk("degreeList")

    logging.info("Creating compactDegreeList...")

    dList = {}
    dFrequency = {}
    for v, layers in degreeList.items():
        dFrequency[v] = {}
        for layer, degreeListLayer in layers.items():
            dFrequency[v][layer] = {}
            for degree in degreeListLayer:
                if degree not in dFrequency[v][layer]:
                    dFrequency[v][layer][degree] = 0
                dFrequency[v][layer][degree] += 1
    for v, layers in dFrequency.items():
        dList[v] = {}
        for layer, frequencyList in layers.items():
            list_d = []
            for degree, freq in frequencyList.items():
                list_d.append((degree, freq))
            list_d.sort(key=lambda x: x[0])
            dList[v][layer] = np.array(list_d, dtype="float")

    logging.info("compactDegreeList created!")

    saveVariableOnDisk(dList, "compactDegreeList")


def verifyDegrees(degrees, degree_v_root, degree_a, degree_b):
    if degree_b == -1:
        degree_now = degree_a
    elif degree_a == -1:
        degree_now = degree_b
    elif abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now


def get_vertices(v, degree_v, degrees, a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices, 2)
    vertices = deque()

    try:
        c_v = 0

        for v2 in degrees[degree_v]["vertices"]:
            if v != v2:
                vertices.append(v2)
                c_v += 1
                if c_v > a_vertices_selected:
                    raise StopIteration

        if "before" not in degrees[degree_v]:
            degree_b = -1
        else:
            degree_b = degrees[degree_v]["before"]
        if "after" not in degrees[degree_v]:
            degree_a = -1
        else:
            degree_a = degrees[degree_v]["after"]
        if degree_b == -1 and degree_a == -1:
            raise StopIteration
        degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)

        while True:
            for v2 in degrees[degree_now]["vertices"]:
                if v != v2:
                    vertices.append(v2)
                    c_v += 1
                    if c_v > a_vertices_selected:
                        raise StopIteration

            if degree_now == degree_b:
                if "before" not in degrees[degree_b]:
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]["before"]
            else:
                if "after" not in degrees[degree_a]:
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]["after"]

            if degree_b == -1 and degree_a == -1:
                raise StopIteration

            degree_now = verifyDegrees(degrees, degree_v, degree_a, degree_b)

    except StopIteration:
        # logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def splitDegreeList(part, c, G, compactDegree):
    if compactDegree:
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk("compactDegreeList")
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk("degreeList")

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk("degrees_vector")

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v, len(G[v]), degrees, a_vertices)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices, "split-vertices-" + str(part))
    saveVariableOnDisk(degreeListsSelected, "split-degreeList-" + str(part))


def calc_distances(
    part,
    active_feature,
    mean_factor,
    med_factor,
    cost_calc,
    cost_function=0,
    compactDegree=False,
):
    vertices = restoreVariableFromDisk("split-vertices-" + str(part))
    degreeList = restoreVariableFromDisk("split-degreeList-" + str(part))

    distances = {}

    # wie in calc_distances_all
    if compactDegree:
        dist_func = cost_max
    else:
        if cost_calc == 0:
            dist_func = cost
        else:
            dist_func = cost_abs

    for v1, nbs in vertices.items():
        lists_v1 = degreeList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1), len(lists_v2))
            distances[v1, v2] = {}

            for layer in range(0, max_layer):
                # wie in calc_distances_all
                if active_feature == 8:
                    # Spezialfall: Mean-of-lists-Euclid
                    dist = euclid_to_mean_of_list(
                        lists_v1[layer], lists_v2[layer]
                    )
                else:
                    if cost_function == 0:
                        # DTW
                        dist, path = fastdtw(
                            lists_v1[layer],
                            lists_v2[layer],
                            radius=1,
                            dist=dist_func,
                        )
                    elif cost_function == 1:
                        # mean & variance
                        dist = mean_var(
                            lists_v1[layer],
                            lists_v2[layer],
                            mean_factor,
                            cost_calc,
                        )
                    elif cost_function == 2:
                        # median & variance
                        dist = med_var(
                            lists_v1[layer],
                            lists_v2[layer],
                            med_factor,
                            cost_calc,
                        )
                    else:
                        # Fallback: DTW
                        dist, path = fastdtw(
                            lists_v1[layer],
                            lists_v2[layer],
                            radius=1,
                            dist=dist_func,
                        )

                distances[v1, v2][layer] = dist

            t11 = time()
            logging.info(
                "fastDTW/struct cost between vertices ({}, {}). Time: {}s".format(
                    v1, v2, (t11 - t00)
                )
            )

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances, "distances-" + str(part))
    return



def calc_distances_all(
    active_feature,
    vertices,
    list_vertices,
    degreeList,
    part,
    mean_factor,
    med_factor,
    cost_calc,
    cost_function=0,
    compactDegree=False,
):
    distances = {}
    cont = 0

    if compactDegree:
        dist_func = cost_max
    else:
        if cost_calc == 0:
            dist_func = cost
        else:
            dist_func = cost_abs

    for v1 in vertices:
        lists_v1 = degreeList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1), len(lists_v2))
            distances[v1, v2] = {}

            for layer in range(0, max_layer):
                if active_feature == 8:
                    dist = euclid_to_mean_of_list(lists_v1[layer], lists_v2[layer])
                else:
                    if cost_function == 0:
                        dist, path = fastdtw(
                            lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func
                        )
                    elif cost_function == 1:
                        dist = mean_var(
                            lists_v1[layer], lists_v2[layer], mean_factor, cost_calc
                        )
                    elif cost_function == 2:
                        dist = med_var(
                            lists_v1[layer], lists_v2[layer], med_factor, cost_calc
                        )

                distances[v1, v2][layer] = dist

        cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances, "distances-" + str(part))
    return


def selectVertices(layer, fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recovering distances from disk...")
    distances = restoreVariableFromDisk("distances")

    threshold = calcThresholdDistance(previousLayer, distances, fractionCalcDists)

    logging.info("Selecting vertices...")

    vertices_selected = deque()

    for vertices, layers in distances.items():
        if previousLayer not in layers:
            continue
        if layers[previousLayer] <= threshold:
            vertices_selected.append(vertices)

    distances = {}

    logging.info("Vertices selected.")

    return vertices_selected


def preprocess_consolides_distances(distances, startLayer=1):
    logging.info("Consolidating distances...")

    for vertices, layers in distances.items():
        keys_layers = sorted(layers.keys())
        startLayer = min(len(keys_layers), startLayer)
        for layer in range(0, startLayer):
            keys_layers.pop(0)

        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info("Distances consolidated.")


def exec_bfs_compact(G, workers, calcUntilLayer):
    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    logging.info(
        f"exec_bfs_compact: BFS will be run from {len(vertices)} vertices"
    )
    parts = workers
    chunks = partition(vertices, parts)

    logging.info("Capturing larger degree...")
    maxDegree = 0
    for v in vertices:
        if len(G[v]) > maxDegree:
            maxDegree = len(G[v])
    logging.info("Larger degree captured")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(
                getCompactDegreeListsVertices, G, c, maxDegree, calcUntilLayer
            )
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList, "compactDegreeList")
    t1 = time()
    logging.info("Execution time - BFS: {}m".format((t1 - t0) / 60))

    return


def exec_bfs(
    G,
    pagerank,
    closeness_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    core_number,
    clustering,
    active_feature,
    workers,
    calcUntilLayer,
):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    
    logging.info(
        f"exec_bfs: BFS will be run from {len(vertices)} vertices"
    )
    start_all = time()
    
    parts = workers
    chunks = partition(vertices, parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        part = 1
        total_parts = len(chunks)
        for c in chunks:
            job = executor.submit(
                getDegreeListsVertices,
                G,
                pagerank,
                closeness_centrality,
                betweenness_centrality,
                eigenvector_centrality,
                core_number,
                clustering,
                active_feature,
                c,
                calcUntilLayer,
            )
            futures[job] = part
            part += 1

        done_parts = 0
        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)
            done_parts += 1
            logging.info(
                f"exec_bfs: finished chunk {done_parts}/{total_parts} "
                f"(elapsed {(time() - start_all)/60:.1f} min)"
            )

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList, "degreeList")
    t1 = time()
    logging.info("Execution time - BFS: {}m".format((t1 - t0) / 60))

    return


def generate_distances_network_part1(workers):
    parts = workers
    weights_distances = {}
    for part in range(1, parts + 1):
        logging.info("Executing part {}...".format(part))
        distances = restoreVariableFromDisk("distances-" + str(part))

        for vertices, layers in distances.items():
            for layer, distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if layer not in weights_distances:
                    weights_distances[layer] = {}
                weights_distances[layer][vx, vy] = distance

        logging.info("Part {} executed.".format(part))

    for layer, values in weights_distances.items():
        saveVariableOnDisk(values, "weights_distances-layer-" + str(layer))
    return


def generate_distances_network_part2(workers):
    parts = workers
    graphs = {}
    for part in range(1, parts + 1):
        logging.info("Executing part {}...".format(part))
        distances = restoreVariableFromDisk("distances-" + str(part))

        for vertices, layers in distances.items():
            for layer, distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if layer not in graphs:
                    graphs[layer] = {}
                if vx not in graphs[layer]:
                    graphs[layer][vx] = []
                if vy not in graphs[layer]:
                    graphs[layer][vy] = []
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
        logging.info("Part {} executed.".format(part))

    for layer, values in graphs.items():
        saveVariableOnDisk(values, "graphs-layer-" + str(layer))

    return


def generate_distances_network_part3(
    active_feature, pagerank_scale, weight_transformer
):
    layer = 0
    while isPickle("graphs-layer-" + str(layer)):
        graphs = restoreVariableFromDisk("graphs-layer-" + str(layer))
        weights_distances = restoreVariableFromDisk(
            "weights_distances-layer-" + str(layer)
        )

        logging.info("Executing layer {}...".format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}

        for v, neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0

            for n in neighbors:
                if (v, n) in weights_distances:
                    wd = weights_distances[v, n]
                else:
                    wd = weights_distances[n, v]

                try:
                    w = np.float64(1 / (weight_transformer ** float(wd)))
                except OverflowError:
                    w = 0.0

                if active_feature == 1:  # pagerank
                    w *= pagerank_scale

                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        saveVariableOnDisk(weights, "distances_nets_weights-layer-" + str(layer))
        saveVariableOnDisk(alias_method_j, "alias_method_j-layer-" + str(layer))
        saveVariableOnDisk(alias_method_q, "alias_method_q-layer-" + str(layer))
        logging.info("Layer {} executed.".format(layer))
        layer += 1

    logging.info("Weights created.")

    return


def generate_distances_network_part3_flattened(
    until_layer,
    weight_layer_0,
    weight_layer_1,
    weight_layer_2,
    weight_layer_3,
    weight_layer_4,
    weight_layer_5,
):
    cur_layer = 0

    graphs_layer_0 = restoreVariableFromDisk("graphs-layer-0")

    while (
        isPickle("distances_nets_weights-layer-" + str(cur_layer))
        and cur_layer <= until_layer
    ):
        logging.info("Executing layer {}...".format(cur_layer))

        distances_nets_weights_layer = restoreVariableFromDisk(
            "distances_nets_weights-layer-" + str(cur_layer)
        )

        graphs_layer = restoreVariableFromDisk("graphs-layer-" + str(cur_layer))

        if cur_layer == 0:
            distances_nets_weights_layer_comb = distances_nets_weights_layer
            distances_nets_weights_layer_comb.update(
                (key, [weight_layer_0 * x for x in value])
                for key, value in distances_nets_weights_layer_comb.items()
            )
        elif cur_layer == 1:
            for v, neighbors in graphs_layer_0.items():
                for n in neighbors:
                    if (v in graphs_layer) and (n in graphs_layer[v]):
                        pos_cur_layer = graphs_layer[v].index(n)
                        pos_0 = graphs_layer_0[v].index(n)

                        distances_nets_weights_layer_comb[v][pos_0] += (
                            weight_layer_1
                            * distances_nets_weights_layer[v][pos_cur_layer]
                        )
        elif cur_layer == 2:
            for v, neighbors in graphs_layer_0.items():
                for n in neighbors:
                    if (v in graphs_layer) and (n in graphs_layer[v]):
                        pos_cur_layer = graphs_layer[v].index(n)
                        pos_0 = graphs_layer_0[v].index(n)

                        distances_nets_weights_layer_comb[v][pos_0] += (
                            weight_layer_2
                            * distances_nets_weights_layer[v][pos_cur_layer]
                        )
        elif cur_layer == 3:
            for v, neighbors in graphs_layer_0.items():
                for n in neighbors:
                    if (v in graphs_layer) and (n in graphs_layer[v]):
                        pos_cur_layer = graphs_layer[v].index(n)
                        pos_0 = graphs_layer_0[v].index(n)

                        distances_nets_weights_layer_comb[v][pos_0] += (
                            weight_layer_3
                            * distances_nets_weights_layer[v][pos_cur_layer]
                        )
        elif cur_layer == 4:
            for v, neighbors in graphs_layer_0.items():
                for n in neighbors:
                    if (v in graphs_layer) and (n in graphs_layer[v]):
                        pos_cur_layer = graphs_layer[v].index(n)
                        pos_0 = graphs_layer_0[v].index(n)

                        distances_nets_weights_layer_comb[v][pos_0] += (
                            weight_layer_4
                            * distances_nets_weights_layer[v][pos_cur_layer]
                        )
        else:
            for v, neighbors in graphs_layer_0.items():
                for n in neighbors:
                    if (v in graphs_layer) and (n in graphs_layer[v]):
                        pos_cur_layer = graphs_layer[v].index(n)
                        pos_0 = graphs_layer_0[v].index(n)

                        distances_nets_weights_layer_comb[v][pos_0] += (
                            weight_layer_5
                            * distances_nets_weights_layer[v][pos_cur_layer]
                        )

        cur_layer += 1

    saveVariableOnDisk(
        distances_nets_weights_layer_comb, "distances_nets_weights_layer_comb"
    )
    logging.info("Flattened weights created.")


def generate_distances_network_part4():
    logging.info("Consolidating graphs...")
    graphs_c = {}
    layer = 0
    while isPickle("graphs-layer-" + str(layer)):
        logging.info("Executing layer {}...".format(layer))
        graphs = restoreVariableFromDisk("graphs-layer-" + str(layer))
        graphs_c[layer] = graphs
        logging.info("Layer {} executed.".format(layer))
        layer += 1

    logging.info("Saving distancesNets on disk...")
    saveVariableOnDisk(graphs_c, "distances_nets_graphs")
    logging.info("Graphs consolidated.")
    return


def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while isPickle("alias_method_j-layer-" + str(layer)):
        logging.info("Executing layer {}...".format(layer))
        alias_method_j = restoreVariableFromDisk("alias_method_j-layer-" + str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info("Layer {} executed.".format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    saveVariableOnDisk(alias_method_j_c, "nets_weights_alias_method_j")

    return


def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while isPickle("alias_method_q-layer-" + str(layer)):
        logging.info("Executing layer {}...".format(layer))
        alias_method_q = restoreVariableFromDisk("alias_method_q-layer-" + str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info("Layer {} executed.".format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    saveVariableOnDisk(alias_method_q_c, "nets_weights_alias_method_q")

    return


def generate_distances_network(
    workers,
    until_layer,
    weight_layer_0,
    weight_layer_1,
    weight_layer_2,
    weight_layer_3,
    weight_layer_4,
    weight_layer_5,
    active_feature,
    pagerank_scale,
    weight_transformer,
):
    t0 = time()
    logging.info("Creating distance network...")

    os.system(
        "rm " + returnPathFfstruc2vec() + "/../pickles/weights_distances-layer-*.pickle"
    )
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1, workers)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 1: {}s".format(t))

    t0 = time()
    os.system("rm " + returnPathFfstruc2vec() + "/../pickles/graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2, workers)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 2: {}s".format(t))
    logging.info("distance network created.")

    logging.info("Transforming distances into weights...")

    t0 = time()
    os.system(
        "rm "
        + returnPathFfstruc2vec()
        + "/../pickles/distances_nets_weights-layer-*.pickle"
    )
    os.system(
        "rm " + returnPathFfstruc2vec() + "/../pickles/alias_method_j-layer-*.pickle"
    )
    os.system(
        "rm " + returnPathFfstruc2vec() + "/../pickles/alias_method_q-layer-*.pickle"
    )
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(
            generate_distances_network_part3,
            active_feature,
            pagerank_scale,
            weight_transformer,
        )
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 3: {}s".format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(
            generate_distances_network_part3_flattened,
            until_layer=until_layer,
            weight_layer_0=weight_layer_0,
            weight_layer_1=weight_layer_1,
            weight_layer_2=weight_layer_2,
            weight_layer_3=weight_layer_3,
            weight_layer_4=weight_layer_4,
            weight_layer_5=weight_layer_5,
        )
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 3_flattened: {}s".format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 4: {}s".format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 5: {}s".format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1 - t0
    logging.info("- Time - part 6: {}s".format(t))

    return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q
