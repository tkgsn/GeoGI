import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import share

if __name__ == "__main__":
    #share.RoadNetworkMap.make_map("Akita", 39.898740, 140.247766, 2000)
    #share.RoadNetworkMap.make_map("Tokyo", 35.698218, 139.853887, 2000)
    share.RoadNetworkMap.make_map("Tokyo2", 35.698218, 139.753887, 2000, True)