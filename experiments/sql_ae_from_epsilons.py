import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random
import GeoI
import GeoGI
import share
import numpy as np
import pandas as pd
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--delta", default=1.1, type=float)
parser.add_argument("--epsilon", default=1, type=float)
parser.add_argument("--n_chosen", default=50, type=int)
parser.add_argument("--location", default="Akita", type=str)
args = parser.parse_args()

if __name__ == "__main__":

    n_iter = args.n_iter
    epsilon = args.epsilon
    delta = args.delta
    n_chosen = args.n_chosen
    location_name = args.location
    seed = args.seed

    map = share.RoadNetworkMap(location_name, n_chosen=n_chosen)
    evaluator = share.Evaluator(euclid=False)

    prior = np.array([[1/len(map.ids)]*len(map.ids)]).T

    result_path = pathlib.Path(os.path.dirname(__file__)).parent / "results" / location_name / "sql_ae"
    result_path.mkdir(exist_ok=True, parents=True)

    epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]

    for epsilon in epsilons:

        random.seed(seed)
        results = {"plmg_sql":[], "plmg_ae":[], "optgeoi_sql":[], "optgeoi_ae":[], "gem_sql":[], "gem_ae":[], "optgem_sql":[], "optgem_ae":[], "optgem_epsilon":[]}

        for i in range(n_iter):

            map.resampling(n_chosen=n_chosen)

            plmg = GeoI.PlanarLaplaceMechanismOnGraph(map, epsilon)
            optgeoi = GeoI.OptMechanism(map, epsilon, prior, delta)
            gem = GeoGI.GraphExponentialMechanism(map, epsilon)
            optgem = GeoGI.OptimalGraphExponentialMechanism(map, epsilon, prior)

            results["plmg_sql"].append(evaluator.compute_sql(plmg, map, prior))
            results["plmg_ae"].append(evaluator.compute_ae(plmg, map, prior))
            results["optgeoi_sql"].append(evaluator.compute_sql(optgeoi, map, prior))
            results["optgeoi_ae"].append(evaluator.compute_ae(optgeoi, map, prior))
            results["gem_sql"].append(evaluator.compute_sql(gem, map, prior))
            results["gem_ae"].append(evaluator.compute_ae(gem, map, prior))
            results["optgem_sql"].append(evaluator.compute_sql(optgem, map, prior))
            results["optgem_ae"].append(evaluator.compute_ae(optgem, map, prior))
            results["optgem_epsilon"].append(optgem.compute_GeoGI_epsilon(map))

        df = pd.DataFrame(results)
        df.to_csv(result_path / f"epsilon{epsilon}_nchosen{n_chosen}_delta{delta}.csv", index=False)