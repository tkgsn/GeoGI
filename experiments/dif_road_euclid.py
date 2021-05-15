import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import share
import argparse
import numpy as np
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--location", default="Akita", type=str)
args = parser.parse_args()


if __name__ == "__main__":

    result_path = pathlib.Path(os.path.dirname(__file__)).parent / "results" / args.location / "dif_road_euc"
    result_path.mkdir(exist_ok=True, parents=True)

    print(result_path)

    map = share.RoadNetworkMap(args.location)
    dif = np.average(np.array(list(map.shortest_path_distances.values())) - np.array(list(map.euclidean_distances.values())))

    with open(result_path / "result.txt", "w") as f:
        f.write(str(dif))