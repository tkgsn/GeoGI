from math import radians, cos, sin, asin, sqrt
import osmnx as ox
import os
import joblib
import numpy as np
import pyproj

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


def convert_key_to_str(dic):
    b = {}
    for key, dic_ in dic.items():
        b[str(key)] = {} 
    for key, dic_ in dic.items():
        for key_, value_ in dic_.items():
            b[str(key)][str(key_)] = value_
    return b

EPSG4612 = pyproj.Proj("+init=EPSG:4612")
EPSG2451 = pyproj.Proj("+init=EPSG:2451")

def convert_to_cart(lon, lat):
    x,y = pyproj.transform(EPSG4612, EPSG2451, lon,lat)
    return x,y