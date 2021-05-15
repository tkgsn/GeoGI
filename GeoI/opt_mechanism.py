import pulp
import numpy as np
import GeoI
import share
import itertools

class OptMechanism(share.Mechanism):

    def __init__(self, map, epsilon, prior, delta):
        print("construct a spanner")
        spanner = GeoI.Spanner(map)
        print("solving the linear problem")
        self.distribution = self.construct_distribution(map, epsilon, spanner, prior)

    def construct_distribution(self, map, epsilon, spanner, prior):

        delta = spanner.delta

        variables = [[pulp.LpVariable(str(id_in)+"_"+str(id),0,1,'Continuous') for id in map.ids] for id_in in map.ids]
        problem = pulp.LpProblem('optGeoI', pulp.LpMinimize)
        problem += pulp.lpSum(variables * prior * list(map.euclidean_distances.values()))

        for i in map.ids:
            problem += pulp.lpSum(variables[i]) == 1.0

        for variables_ in variables:
            for variable in variables_:
                problem += variable >= 0

        for edge in spanner.graph.edges:
        #for edge in itertools.combinations(map.ids, 2):
            for id in map.ids:
                problem += variables[edge[0]][id] <= np.exp(epsilon * map.euclidean_distances[edge[0]][edge[1]] / delta) * variables[edge[1]][id]
                problem += variables[edge[1]][id] <= np.exp(epsilon * map.euclidean_distances[edge[0]][edge[1]] / delta) * variables[edge[0]][id]
                
        status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
        f = np.frompyfunc(lambda x:x.value(), 1, 1)

        return f(variables).astype(np.float64)