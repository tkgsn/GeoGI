import numpy as np
import networkx as nx
import os
import osmnx as ox
import joblib
import share
import random
import pathlib

class Map:
    def __init__():
        print("init")

class GridMap(Map):

    def __init__(self, n_x_nodes, n_y_nodes, x_length, y_length):
        self.ids = [i+j*n_x_nodes for j in range(n_y_nodes) for i in range(n_x_nodes)]
        self.coords = {id:np.array([(id%n_x_nodes)*x_length,int(id/n_y_nodes)*y_length]) for id in self.ids}
        self.euclidean_distances = self._compute_Euclidean_distance(self.coords, self.ids)

    def _compute_Euclidean_distance(self, coords, ids):
        euclidean_distances = {id:[np.linalg.norm(coords[id]-coords[id_in]) for id_in in ids] for id in ids}
        return euclidean_distances

class RoadNetworkMap(Map):

    def __init__(self, location_name, n_chosen=0):
        self.graph, self.all_shortest_path_distances = self.load(location_name)
        self.all_nodes = list(self.all_shortest_path_distances.keys())

        if n_chosen==0:
            n_chosen = len(self.all_nodes)
        else:
            n_chosen = n_chosen

        self._initialize(self.all_nodes, n_chosen)

    @classmethod
    def make_map(cls, location_name, lat, lon, distance,  simplify=False, n_chosen=0):
        G = ox.graph_from_point((lat, lon), network_type="walk", dist_type="bbox", dist=distance, simplify=simplify)

        if n_chosen >= len(G):
            n_chosen = 0

        if n_chosen == 0:
            orig_shortest_path_distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
            shortest_path_distances = {str(node):{} for node in G.nodes}
            for node in G.nodes:
                for node_ in G.nodes:
                    shortest_path_distances[str(node)][str(node_)] = orig_shortest_path_distances[node][node_]
        else:
            all_nodes = list(G.nodes)
            chosen_nodes = random.sample(all_nodes, n_chosen)
            orig_shortest_path_distances = {node: nx.single_source_dijkstra_path_length(G, node, weight="length") for node in chosen_nodes}
            shortest_path_distances = {str(node):{} for node in chosen_nodes}
            for node in chosen_nodes:
                for node_ in chosen_nodes:
                    shortest_path_distances[str(node)][str(node_)] = orig_shortest_path_distances[node][node_]

        data_path = pathlib.Path("/") / "data" / "takagi" / "geogi" / "data"
        data_path.mkdir(exist_ok=True)
        graph_path = data_path / "graph"
        graph_path.mkdir(exist_ok=True)
        graph_data_path = data_path / "graph_data"
        graph_data_path.mkdir(exist_ok=True)

        ox.save_graphml(G, graph_path / f"{location_name}.ml")
        joblib.dump(filename=graph_data_path / f"{location_name}_shortest_path_distances.jbl", value=shortest_path_distances)

    def __len__(self):
        return len(self.ids)

    def _initialize(self, nodes, n_chosen):
        nodes = self.random_sampling(nodes, n_chosen)
        self.node_to_id = {node: i for i, node in enumerate(nodes)}
        self.id_to_node = {i: node for i, node in enumerate(nodes)}
        self.ids = list(self.node_to_id.values())
        self.euclidean_distances = self._compute_Euclidean_distance(self.graph, nodes)
        self.shortest_path_distances = self._choose_shortest_path_distance(self.all_shortest_path_distances, nodes)

    def resampling(self, n_chosen):
        self._initialize(self.all_nodes, n_chosen)

    def random_sampling(self, nodes, n_chosen):
        return random.sample(nodes, n_chosen)

    def load(self, location_name):
        graph_dir = pathlib.Path(os.path.dirname(__file__)) / ".." / "data" / "graph"
        data_dir = pathlib.Path(os.path.dirname(__file__)) / ".." / "data" / "graph_data"

        graph = nx.read_graphml(graph_dir / f'{location_name}.ml')
        shortest_path_distances = joblib.load(data_dir / f"{location_name}_shortest_path_distances.jbl")

        return graph, shortest_path_distances

    def _compute_Euclidean_distance(self, graph, nodes):
        euclidean_distances = {self.node_to_id[node]:[share.haversine(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]), float(graph.nodes[node_in]["x"]), float(graph.nodes[node_in]["y"])) for node_in in nodes] for node in nodes}
        return euclidean_distances

    def _choose_shortest_path_distance(self, shortest_path_distances, nodes):
        return {self.node_to_id[node]:[shortest_path_distances[node][node_in]*1e-3 for node_in in nodes] for node in nodes}