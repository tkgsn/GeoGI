import numpy as np
import itertools
import share


class GraphExponentialMechanism(share.Mechanism):

    def __init__(self, map, epsilon):
        self.distribution = self.construct_distribution(map, epsilon)

    def construct_distribution(self, map, epsilon):
        return self.normalize(self.exponential_distribution(map, epsilon))

    def exponential_distribution(self, map, epsilon):
        return np.exp((-epsilon/2) * np.array(list(map.shortest_path_distances.values())))

class OptimalGraphExponentialMechanism(GraphExponentialMechanism):

    def __init__(self, map, epsilon, prior):
        self.distribution = self.construct_distribution(map, epsilon, prior)

    def construct_distribution(self, map, epsilon, prior):
        shortest_path_distances = np.array(list(map.shortest_path_distances.values()))
        
        removed_nodes = []
        remove_flg = False

        exp_eps_distance = np.exp( -(epsilon/2) * shortest_path_distances )
        distance_prior_eed = shortest_path_distances * prior * exp_eps_distance

        def difference(mat, index):
            return mat[:, index].reshape(-1,1)

        sum_eed = np.sum(exp_eps_distance, axis=1).reshape(-1,1)
        sum_dpe = np.sum(distance_prior_eed, axis=1).reshape(-1,1)

        def bool_to_sign(bool):
            if bool:
                return 1
            else:
                return -1

        while True:
            initial_sql = np.sum(sum_dpe / sum_eed)
            print("Q_loss", initial_sql)
            current_sql = initial_sql

            for id in map.ids:

                #print(f"{id}/{len(map.ids)}", end="\r")

                if id in removed_nodes:
                    remove_flg = True
                    removed_nodes.remove(id)
                else:
                    remove_flg = False
                    removed_nodes.append(id)

                if len(removed_nodes) == len(map.ids):
                    removed_nodes.remove(id)
                    continue

                sign = bool_to_sign(remove_flg)
                temp_sum_eed = sum_eed + sign * difference(exp_eps_distance, id)
                temp_sum_dpe = sum_dpe + sign * difference(distance_prior_eed, id)
                temp_sql = np.sum(temp_sum_dpe / temp_sum_eed)

                if temp_sql >= current_sql:
                    if remove_flg:
                        removed_nodes.append(id)
                    else:
                        removed_nodes.remove(id)
                else:
                    current_sql = temp_sql
                    sum_eed = temp_sum_eed
                    sum_dpe = temp_sum_dpe

            if initial_sql == current_sql:
                break

        remove_mat = self._make_remove_mat(len(map), removed_nodes)
        self.removed_nodes = [map.id_to_node[id] for id in removed_nodes]
        return self.normalize(np.dot(self.exponential_distribution(map, epsilon), remove_mat))

    def _make_remove_mat(self, n_nodes, removed_nodes):
        remove_mat = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            if i not in removed_nodes:
                remove_mat[i][i] = 1
        return remove_mat

class GivenNodesGraphExponentialMechanism(OptimalGraphExponentialMechanism):

    def __init__(self, map, epsilon, remove_nodes=[]):
        self.distribution = self.construct_distribution(map, epsilon, remove_nodes)

    def construct_distribution(self, map, epsilon, remove_nodes):
        remove_mat = self._make_remove_mat(len(map), [map.node_to_id[node] for node in remove_nodes])
        return self.normalize(np.dot(self.exponential_distribution(map, epsilon), remove_mat))


class OptimalEpsilonGraphExponentialMechanism(GraphExponentialMechanism):

    def __init__(self, map1, map2, epsilon):
        self.distribution = self.construct_distribution(map1, map2, epsilon)

    def construct_distribution(self, map1, map2, epsilon):
        shortest_path_distances = np.array(list(map2.shortest_path_distances.values()))
        
        removed_nodes = []
        remove_flg = False

        exp_eps_distance = np.exp( -(epsilon/2) * shortest_path_distances )
        sum_eed = np.sum(exp_eps_distance, axis=1).reshape(-1,1)

        self.distribution = super().construct_distribution(map2, epsilon)

        def difference(mat, index):
            return mat[:, index].reshape(-1,1)

        def bool_to_sign(bool):
            if bool:
                return 1
            else:
                return -1

        while True:

            #initial_alphaepsilon = np.max(np.abs([np.log(sum_eed[tuple[0]]/sum_eed[tuple[1]])/shortest_path_distances[tuple[0]][tuple[1]] for tuple in itertools.combinations(range(len(map.ids)), 2)]))
            initial_alphaepsilon = self.compute_GeoGI_epsilon_at_input_domain(map1, map2)
            print("initial", initial_alphaepsilon)
            current_alphaepsilon = initial_alphaepsilon

            for id in map2.ids:

                print(f"{id}/{len(map2.ids)}", end="\r")

                distribution = self.distribution
 
                if id in removed_nodes:
                    remove_flg = True
                    removed_nodes.remove(id)
                else:
                    remove_flg = False
                    removed_nodes.append(id)

                if len(removed_nodes) == len(map2.ids):
                    removed_nodes.remove(id)
                    continue

                #sign = bool_to_sign(remove_flg)
                #temp_sum_eed = sum_eed + sign * difference(exp_eps_distance, id)
                #temp_alphaepsilon = np.max(np.abs([np.log(temp_sum_eed[tuple[0]]/temp_sum_eed[tuple[1]])/shortest_path_distances[tuple[0]][tuple[1]] for tuple in itertools.combinations(range(len(map.ids)), 2)]))
                remove_mat = self._make_remove_mat(len(map2.ids), removed_nodes)
                self.distribution = self.normalize(np.dot(self.exponential_distribution(map2, epsilon), remove_mat))
                temp_alphaepsilon = self.compute_GeoGI_epsilon_at_input_domain(map1, map2)
                print(temp_alphaepsilon)

                if temp_alphaepsilon >= current_alphaepsilon:
                    self.distribution = distribution
                    if remove_flg:
                        removed_nodes.append(id)
                    else:
                        removed_nodes.remove(id)
                else:
                    current_alphaepsilon = temp_alphaepsilon
                    #sum_eed = temp_sum_eed

            if initial_alphaepsilon == current_alphaepsilon:
                break

        remove_mat = self._make_remove_mat(len(map2.ids), removed_nodes)
        return self.normalize(np.dot(self.exponential_distribution(map2, epsilon), remove_mat))


    def _make_remove_mat(self, n_nodes, removed_nodes):
        remove_mat = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            if i not in removed_nodes:
                remove_mat[i][i] = 1
        return remove_mat