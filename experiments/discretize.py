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

    map1 = share.RoadNetworkMap(location_name, n_chosen=n_chosen)
    map2 = share.RoadNetworkMap(location_name)
    evaluator = share.Evaluator(euclid=False)

    prior1 = np.array([[1/len(map1.ids)]*len(map1.ids)]).T
    prior2 = np.array([[1/len(map2.ids)]*len(map2.ids)]).T

    result_path = pathlib.Path(os.path.dirname(__file__)).parent / "results" / f"{location_name}_10000" / "discretize"
    result_path.mkdir(exist_ok=True, parents=True)

    epsilons = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]

    for epsilon in epsilons:

        print("epsilon:", epsilon)
        random.seed(seed)
        results = {"continuous_optgeoi_epsilon":[], "optgeoi_sql":[], "optgem_epsilon":[], "optgem_sql":[], "optgem_with_optgeoi_epsilon_sql":[]}
        optgem = GeoGI.OptimalGraphExponentialMechanism(map2, epsilon, prior2)
        optgem_sql = evaluator.compute_sql(optgem, map2, prior2)
        
        for i in range(n_iter):

            print(f"{i}/{n_iter}")

            map1.resampling(n_chosen)
            optgeoi = GeoI.OptMechanism(map1, epsilon, prior1, delta)
            optgeoi.change_map(map1, map2)
            continuous_optgeoi_epsilon = optgeoi.compute_GeoGI_epsilon(map2)

            optgem_epsilon = optgem.compute_GeoGI_epsilon_at_input_domain(map1, map2)

            optgem_with_optgeoi_epsilon = GeoGI.OptimalGraphExponentialMechanism(map2, continuous_optgeoi_epsilon, prior2)
            optgem_with_optgeoi_epsilon_sql = evaluator.compute_sql(optgem_with_optgeoi_epsilon, map2, prior2)

            results["continuous_optgeoi_epsilon"].append(continuous_optgeoi_epsilon)
            results["optgeoi_sql"].append(evaluator.compute_sql(optgeoi, map2, prior2))
            results["optgem_sql"].append(optgem_sql)
            results["optgem_epsilon"].append(optgem_epsilon)
            results["optgem_with_optgeoi_epsilon_sql"].append(optgem_with_optgeoi_epsilon_sql)

        df = pd.DataFrame(results)
        df.to_csv(result_path / f"epsilon{epsilon}_nchosen{n_chosen}_delta{delta}.csv", index=False)