import numpy as np
import pulp

class Evaluator():
    def __init__(self, euclid=True):
        self.euclid = euclid

    def compute_sql(self, mechanism, map, prior):
        if self.euclid:
            return np.sum(mechanism.distribution * prior * list(map.euclidean_distances.values()))
        else:
            return np.sum(mechanism.distribution * prior * list(map.shortest_path_distances.values()))

    def compute_ae(self, mechanism, map, prior):
        remapping = self._compute_inference_function(mechanism, map, prior)
        prior_distribution = mechanism.distribution * prior
        
        if self.euclid:
            ae = np.sum(remapping * np.dot(prior_distribution.T, list(map.euclidean_distances.values())))
        else:
            ae = np.sum(remapping * np.dot(prior_distribution.T, list(map.shortest_path_distances.values())))
        
        return ae

    def _compute_inference_function(self, mechanism, map, prior):
        remapping = [[pulp.LpVariable((str(perturbed_node)+"_"+str(inf_node)),0,1,'Continuous') for inf_node in map.ids] for perturbed_node in map.ids]
        problem = pulp.LpProblem('AE', pulp.LpMinimize)

        prior_distribution = prior * mechanism.distribution
        if self.euclid:
            problem += pulp.lpSum(remapping * np.dot(prior_distribution.T, list(map.euclidean_distances.values())))
        else:
            problem += pulp.lpSum(remapping * np.dot(prior_distribution.T, list(map.shortest_path_distances.values())))

        for i in range(len(map.ids)):
            problem += pulp.lpSum(remapping[i]) == 1.0

        status = problem.solve(pulp.PULP_CBC_CMD(msg=1))
        if(status):
            f = np.frompyfunc(lambda x:x.value(), 1, 1)
        else:
            print("error")
        return f(remapping)