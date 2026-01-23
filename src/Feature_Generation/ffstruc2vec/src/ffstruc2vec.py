# -*- coding: utf-8 -*-

import numpy as np
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
from collections import deque

from utils import saveVariableOnDisk, restoreVariableFromDisk, partition
from algorithms import (
    generate_parameters_random_walk,
    generate_random_walks_large_graphs,
    generate_random_walks
)
from algorithms_distances import (
    exec_bfs,
    exec_bfs_compact,
    preprocess_degreeLists,
    calc_distances_all,
    splitDegreeList,
    calc_distances,
    preprocess_consolides_distances,
    generate_distances_network
)


class Graph:
    def __init__(self, g, is_directed, workers, untilLayer=None):
        logging.info(" - Converting graph to dict...")
        self.G = g.gToDict()
        logging.info("Graph converted.")

        self.num_vertices = g.number_of_nodes()
        self.num_edges = g.number_of_edges()
        self.is_directed = is_directed
        self.workers = workers
        self.calcUntilLayer = untilLayer

        (
            self.pagerank,
            self.closeness_centrality,
            self.betweenness_centrality,
            self.eigenvector_centrality,
            self.core_number,
            self.clustering,
        ) = calculate_indicators(self.G)

        logging.info("Graph - Number of vertices: {}".format(self.num_vertices))
        logging.info("Graph - Number of edges: {}".format(self.num_edges))

    def preprocess_neighbors_with_bfs(self, active_feature):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(
                exec_bfs,
                self.G,
                self.pagerank,
                self.closeness_centrality,
                self.betweenness_centrality,
                self.eigenvector_centrality,
                self.core_number,
                self.clustering,
                active_feature,
                self.workers,
                self.calcUntilLayer,
            )
            job.result()

        return

    def preprocess_neighbors_with_bfs_compact(self):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(
                exec_bfs_compact, self.G, self.workers, self.calcUntilLayer
            )

            job.result()

        return

    def preprocess_degree_lists(self):
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            job = executor.submit(preprocess_degreeLists)

            job.result()

        return

    def create_vectors(self):
        logging.info("Creating degree vectors...")
        degrees = {}
        degrees_sorted = set()
        G = self.G
        for v in G.keys():
            degree = len(G[v])
            degrees_sorted.add(degree)
            if degree not in degrees:
                degrees[degree] = {}
                degrees[degree]["vertices"] = deque()
            degrees[degree]["vertices"].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype="int")
        degrees_sorted = np.sort(degrees_sorted)

        length = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if index > 0:
                degrees[degree]["before"] = degrees_sorted[index - 1]
            if index < (length - 1):
                degrees[degree]["after"] = degrees_sorted[index + 1]

        logging.info("Degree vectors created.")
        logging.info("Saving degree vectors...")
        saveVariableOnDisk(degrees, "degrees_vector")

    def calc_distances_all_vertices(
        self,
        active_feature,
        mean_factor,
        med_factor,
        cost_calc,
        cost_function=0,
        compactDegree=False,
    ):
        logging.info("Using compactDegree: {}".format(compactDegree))
        if self.calcUntilLayer:
            logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

        futures = {}

        vertices = list(reversed(sorted(self.G.keys())))

        if compactDegree:
            logging.info("Recovering degreeList from disk...")
            degreeList = restoreVariableFromDisk("compactDegreeList")
        else:
            logging.info("Recovering compactDegreeList from disk...")
            degreeList = restoreVariableFromDisk("degreeList")

        parts = self.workers
        chunks = partition(vertices, parts)

        t0 = time()

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                logging.info("Executing part {}...".format(part))
                list_v = []
                for v in c:
                    list_v.append([vd for vd in degreeList.keys() if vd > v])
                job = executor.submit(
                    calc_distances_all,
                    active_feature,
                    c,
                    list_v,
                    degreeList,
                    part,
                    mean_factor,
                    med_factor,
                    cost_calc,
                    cost_function=cost_function,
                    compactDegree=compactDegree,
                )
                futures[job] = part
                part += 1

            logging.info("Receiving results...")

            for job in as_completed(futures):
                job.result()
                r = futures[job]
                logging.info("Part {} Completed.".format(r))

        logging.info("Distances calculated.")
        t1 = time()
        logging.info("Time : {}m".format((t1 - t0) / 60))

        return

    def calc_distances(
            self,
            active_feature,
            mean_factor,
            med_factor,
            cost_calc,
            cost_function,
            compactDegree=False
    ):
        logging.info("Using compactDegree: {}".format(compactDegree))
        if self.calcUntilLayer:
            logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

        futures = {}

        G = self.G
        vertices = G.keys()

        parts = self.workers
        chunks = partition(vertices, parts)

        with ProcessPoolExecutor(max_workers=1) as executor:
            logging.info("Split degree List...")
            part = 1
            for c in chunks:
                job = executor.submit(splitDegreeList, part, c, G, compactDegree)
                job.result()
                logging.info("degreeList {} completed.".format(part))
                part += 1

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            part = 1
            for c in chunks:
                logging.info("Executing part {}...".format(part))
                job = executor.submit(
                    calc_distances,
                    part,
                    active_feature,
                    mean_factor,
                    med_factor,
                    cost_calc,
                    cost_function,
                    compactDegree=compactDegree,
                )
                futures[job] = part
                part += 1

            logging.info("Receiving results...")
            for job in as_completed(futures):
                job.result()
                r = futures[job]
                logging.info("Part {} completed.".format(r))

        return

    def consolide_distances(self):
        distances = {}

        parts = self.workers
        for part in range(1, parts + 1):
            d = restoreVariableFromDisk("distances-" + str(part))
            preprocess_consolides_distances(distances)
            distances.update(d)

        preprocess_consolides_distances(distances)
        saveVariableOnDisk(distances, "distances")

    def create_distances_network(
        self,
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
        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(
                generate_distances_network,
                self.workers,
                until_layer=until_layer,
                weight_layer_0=weight_layer_0,
                weight_layer_1=weight_layer_1,
                weight_layer_2=weight_layer_2,
                weight_layer_3=weight_layer_3,
                weight_layer_4=weight_layer_4,
                weight_layer_5=weight_layer_5,
                active_feature=active_feature,
                pagerank_scale=pagerank_scale,
                weight_transformer=weight_transformer,
            )

            job.result()

        return

    def preprocess_parameters_random_walk(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generate_parameters_random_walk, self.workers)
            job.result()
        return

    def simulate_walks(
        self,
        num_walks,
        walk_length,
        probability_distribution,
        softmax_factor,
    ):
        # for large graphs, it is serially executed, because of memory use.
        if len(self.G) > 500000:
            with ProcessPoolExecutor(max_workers=1) as executor:
                job = executor.submit(
                    generate_random_walks_large_graphs,
                    num_walks,
                    walk_length,
                    self.workers,
                    list(self.G.keys()),
                    probability_distribution,
                    softmax_factor,
                )

                job.result()

        else:
            with ProcessPoolExecutor(max_workers=1) as executor:
                job = executor.submit(
                    generate_random_walks,
                    num_walks,
                    walk_length,
                    self.workers,
                    list(self.G.keys()),
                    probability_distribution,
                    softmax_factor,
                )

                job.result()

        return


def calculate_indicators(G):
    import networkx as nx
    import logging
    import time

    logger = logging.getLogger(__name__)

    start_total = time.time()
    logger.info("calculate_indicators: START")

    G_x = nx.Graph()

    n_nodes = len(G)
    n_edges_half = sum(len(G[v]) for v in G) // 2
    logger.info(
        "calculate_indicators: building nx.Graph from %d nodes, ~%d edges",
        n_nodes, n_edges_half
    )

    for vertex in G:
        for neighbor in G[vertex]:
            G_x.add_edge(vertex, neighbor)

    logger.info(
        "calculate_indicators: nx.Graph built (nodes=%d, edges=%d) in %.2fs",
        G_x.number_of_nodes(),
        G_x.number_of_edges(),
        time.time() - start_total,
    )

    # --- wirklich berechnen: PageRank ---
    t0 = time.time()
    logger.info("calculate_indicators: computing PageRank ...")
    pr = nx.pagerank(G_x)
    logger.info("calculate_indicators: PageRank done in %.2fs", time.time() - t0)

    nodes = list(G_x.nodes())

    # --- SKIP: closeness_centrality (zu teuer) ---
    logger.info("calculate_indicators: SKIP closeness_centrality – using zeros")
    closeness = {v: 0.0 for v in nodes}

    # --- SKIP: betweenness_centrality (viel zu teuer) ---
    logger.info("calculate_indicators: SKIP betweenness_centrality – using zeros")
    btw = {v: 0.0 for v in nodes}

    # --- SKIP: eigenvector_centrality (ebenfalls teuer) ---
    logger.info("calculate_indicators: SKIP eigenvector_centrality – using zeros")
    eig = {v: 0.0 for v in nodes}

    # --- diese beiden sind relativ günstig, können wir lassen ---
    t0 = time.time()
    logger.info("calculate_indicators: computing core_number ...")
    core = nx.core_number(G_x)
    logger.info("calculate_indicators: core_number done in %.2fs", time.time() - t0)

    t0 = time.time()
    logger.info("calculate_indicators: computing clustering ...")
    clust = nx.clustering(G_x)
    logger.info("calculate_indicators: clustering done in %.2fs", time.time() - t0)

    logger.info("calculate_indicators: FINISHED in %.2fs", time.time() - start_total)

    return pr, closeness, btw, eig, core, clust

