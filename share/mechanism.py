import numpy as np
import itertools

class Mechanism():

    def change_map(self, map1, map2):
        assert len(map1) < len(map2)
        assert len(map1) == len(self.distribution)

        distribution = np.zeros((len(map2), len(map2)))
        for input_node in map2.node_to_id:
            input_map2_index = map2.node_to_id[input_node]
            if input_node in map1.node_to_id:
                input_map1_index = map1.node_to_id[input_node]
                for output_node in map2.node_to_id:
                    output_map2_index = map2.node_to_id[output_node]
                    if output_node in map1.node_to_id:
                        output_map1_index = map1.node_to_id[output_node]
                        distribution[input_map2_index][output_map2_index] = self.distribution[input_map1_index][output_map1_index]
                    else:
                        distribution[input_map2_index][output_map2_index] = 0
            else:
                distances = map2.shortest_path_distances[input_map2_index]
                distance = float("inf")
                chosen_node = 0
                for node in map1.node_to_id:
                    index = map2.node_to_id[node]
                    temp_distance = distances[index]
                    if temp_distance < distance:
                        chosen_node = node
                        distance = temp_distance
                chosen_map1_index = map1.node_to_id[chosen_node]

                for output_node in map2.node_to_id:
                    output_map2_index = map2.node_to_id[output_node]
                    distribution[input_map2_index][output_map2_index] = self.distribution[chosen_map1_index][map1.node_to_id[output_node]] if output_node in map1.node_to_id else 0

        self.distribution = distribution

    def normalize(self, distribution):
        sum_dist = np.sum(distribution, axis=1).reshape(-1,1)
        m = distribution / sum_dist
        return m

    def compute_GeoGI_epsilon_at_input_domain(self, map1, map2):
        input_domain = list(map1.node_to_id.keys())
        input_domain = np.array([map2.node_to_id[node] for node in input_domain])
        distances = np.array(list(map2.shortest_path_distances.values()))

        return np.max([np.nanmax(np.abs(np.log(self.distribution[comb[0]]/self.distribution[comb[1]]) / distances[comb[0]][comb[1]])) for comb in itertools.combinations(input_domain, 2)])

    def _compute_average_epsilon(self, distances):
        epsilons = []
        for comb in itertools.combinations(range(len(distances)), 2):
            a = np.abs(np.log(self.distribution[comb[0]]/self.distribution[comb[1]]/ distances[comb[0]][comb[1]]))
            epsilons.append(np.nanmax(a[np.isfinite(a)]))
        return np.average(epsilons)
        #return np.average([np.nanmax(np.abs(np.log(self.distribution[comb[0]]/self.distribution[comb[1]]) / distances[comb[0]][comb[1]])) for comb in itertools.combinations(range(len(distances)), 2)])

    def _compute_epsilon(self, distances):
        epsilons = []
        for comb in itertools.combinations(range(len(distances)), 2):
            nonzero_indices1 = np.where(self.distribution[comb[1]] > 1e-10)[0]
            nonzero_indices0 = np.where(self.distribution[comb[0]] > 1e-10)[0]
            nonzero_indices = np.intersect1d(nonzero_indices0, nonzero_indices1)
            if len(nonzero_indices) == 0:
                continue
            # if len(nonzero_indices) != len(nonzero_indices1):
            #     continue
            epsilons.append(np.nanmax(np.abs(np.log((self.distribution[comb[0]][nonzero_indices]/self.distribution[comb[1]][nonzero_indices])) / distances[comb[0]][comb[1]])))
        return np.max(epsilons)

    def __compute_epsilon(self, distances):
        return np.max([np.nanmax(np.abs(np.log(self.distribution[comb[0]]/self.distribution[comb[1]]) / distances[comb[0]][comb[1]])) for comb in itertools.combinations(range(len(distances)), 2)])

    def compute_GeoI_epsilon(self, map):
        distances = np.array(list(map.euclidean_distances.values()))
        return self._compute_epsilon(distances)

    def compute_GeoGI_average_epsilon(self, map):
        distances = np.array(list(map.shortest_path_distances.values()))
        return self._compute_average_epsilon(distances)

    def compute_GeoGI_epsilon(self, map):
        distances = np.array(list(map.shortest_path_distances.values()))
        return self._compute_epsilon(distances)