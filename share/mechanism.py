import numpy as np
import itertools

class Mechanism():

    def normalize(self, distribution):
        sum_dist = np.sum(distribution, axis=1).reshape(-1,1)
        m = distribution / sum_dist
        return m

    def _compute_epsilon(self, distances):
        epsilon = 0

        def _compute_epsilon(prob_a, prob_b, shortest_path_distance):

            if (prob_a == 0) or (prob_b == 0):
                if not (prob_a == prob_b):
                    return float("inf")
                else:
                    return 0

            a = abs((1/shortest_path_distance) * np.log(prob_a/prob_b))
            b = abs((1/shortest_path_distance) * np.log(prob_b/prob_a))
            return max(a,b)
        
        n_nodes = len(distances)

        for tuple in itertools.combinations(range(n_nodes), 2):
            distance = distances[tuple[0]][tuple[1]]

            for i in range(n_nodes):
                prob_a = self.distribution[tuple[0]][i]
                prob_b = self.distribution[tuple[1]][i]

                temp_epsilon = _compute_epsilon(prob_a, prob_b, distance)
                if temp_epsilon > epsilon:
                    epsilon = temp_epsilon

        return epsilon

    def compute_GeoI_epsilon(self, map):
        distances = map.euclidean_distances
        return self._compute_epsilon(distances)

    def compute_GeoGI_epsilon(self, map):
        distances = map.shortest_path_distances
        return self._compute_epsilon(distances)