import networkx as nx
import numpy as np
import functools

class Spanner:

    def __init__(self, map, delta):
        self.delta = delta
        self.graph = self.construct_spanner(map, delta)

    def construct_spanner(self, map, delta):

        spanner = nx.Graph()
        spanner.add_nodes_from(map.ids)

        joined_euclidean_distances = functools.reduce(lambda x,y:x+y, map.euclidean_distances.values())
        sorted_joined_euclidean_distances = np.argsort(joined_euclidean_distances)[::-1]
        n_nodes = len(map.euclidean_distances[0])

        def index_to_comb(index):
            return (int(index/n_nodes), index%n_nodes)

        for i, index in enumerate(sorted_joined_euclidean_distances):
            print(f"{i}/{n_nodes*n_nodes}", end="\r")

            comb = index_to_comb(index)
            if nx.has_path(spanner, comb[0], comb[1]):
                shortest_path_distance = nx.dijkstra_path_length(spanner, comb[0], comb[1], weight="weight")
            else:
                shortest_path_distance = float("inf")

            if shortest_path_distance >= delta * map.euclidean_distances[comb[0]][comb[1]]:
                spanner.add_edge(comb[0], comb[1], weight=map.euclidean_distances[comb[0]][comb[1]])

        return spanner
