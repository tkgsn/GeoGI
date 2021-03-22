import numpy as np
import share

class Mechanism():

    def normalize(self, distribution):
        sum_dist = np.sum(distribution, axis=1).reshape(-1,1)
        m = distribution / sum_dist
        return m

class GraphExponentialMechanism(Mechanism):

    def __init__(self, map, epsilon):
        self.distribution = self.construct_distribution(map, epsilon)

    def construct_distribution(self, map, epsilon):
        return self.normalize(self.exponential_distribution(map, epsilon))

    def exponential_distribution(self, map, epsilon):
        return np.exp((-epsilon/2) * np.array(list(map.shortest_path_distances.values())))


class OptimalGraphExponentialMechanism(GraphExponentialMechanism):

    def __init__(self, map, epsilon, prior):
        self.construct_distribution(map, epsilon, prior)

    def construct_distribution(self, map, epsilon, prior):
        evaluator = share.Evaluator(euclid=False)
        
        removed_nodes = []
        self.distribution = super().construct_distribution(map, epsilon)
        remove_flg = False

        while True:

            initial_sql = evaluator.compute_sql(self, map, prior)
            current_sql = initial_sql

            for id in map.ids:

                if id in removed_nodes:
                    remove_flg = True
                    removed_nodes.remove(id)
                else:
                    remove_flg = False
                    removed_nodes.append(id)

                if len(removed_nodes) == len(map.ids):
                    removed_nodes.remove(id)
                    continue
                
                remove_mat = self._make_remove_mat(len(map.ids), removed_nodes)
                self.distribution = self.normalize(np.dot(self.exponential_distribution(map, epsilon), remove_mat))
                temp_sql = evaluator.compute_sql(self, map, prior)

                if temp_sql >= current_sql:
                    if remove_flg:
                        removed_nodes.append(id)
                    else:
                        removed_nodes.remove(id)
                else:
                    current_sql = temp_sql

            remove_mat = self._make_remove_mat(len(map.ids), removed_nodes)
            self.distribution = self.normalize(np.dot(self.exponential_distribution(map, epsilon), remove_mat))

            if initial_sql == current_sql:
                break

        return self.distribution


    def _make_remove_mat(self, n_nodes, removed_nodes):
        remove_mat = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            if i not in removed_nodes:
                remove_mat[i][i] = 1
        return remove_mat