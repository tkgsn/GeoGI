import osmnx as ox
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
import share

#垂直二等分線
def per_bi(a_x,a_y,b_x,b_y):
    def f(x):
        if a_x == b_x:
            ret = (b_x**2 - a_x**2 + b_y**2 - a_y**2)/((b_y-a_y)*2)
        else:
            ret = -x*(b_x-a_x)/(b_y-a_y) + (b_x**2 - a_x**2 + b_y**2 - a_y**2)/((b_y-a_y)*2)
        return ret
    return None if b_y==a_y else f

def laplace(x0,y0,epsilon):
    return lambda x,y : (epsilon**2/(2*math.pi))*np.exp(-epsilon*np.sqrt((x0-x)**2+(y0-y)**2))

class PlanarLaplaceMechanismOnGraph(share.Mechanism):
    
    def __init__(self, map, epsilon, graph_type="ox", is_vis=False):
        self.is_vis = is_vis
        self.graph_type = graph_type
        self.distribution = self.construct_distribution(map, epsilon)
    
    def _make_voronoi(self, map):
        latlon_list = []
        for id in map.ids:
            data = map.graph.nodes[map.id_to_node[id]]
            latlon_list.append((data["y"], data["x"]))

        if self.graph_type == "ox":
            cart_list = [share.convert_to_cart(ll[1],ll[0]) for ll in latlon_list]
        else:
            cart_list = [(xy[0]*100,xy[1]*100) for xy in latlon_list]

        x_list, y_list = [cart_tuple[0] for cart_tuple in cart_list], [cart_tuple[1] for cart_tuple in cart_list]
        x_min, y_min = min(x_list), min(y_list)
        points = np.array([((cart_tuple[0] - x_min)*1e-3,(cart_tuple[1] - y_min)*1e-3) for cart_tuple in cart_list])
    
        vor = Voronoi(points)
        if self.is_vis:
            voronoi_plot_2d(vor,show_vertices=False,point_size=1)
            plt.show()
        
        return vor
    
    def _make_vor_equations(self, vor, map):
        eq_dict_dict = {}
        for i in map.ids:
            eq_dict_dict[i] = []

        center = vor.points.mean(axis=0)
        
        for point_tuple,ver_list in vor.ridge_dict.items():
            #方向を求める dirction
            midpoint = vor.points[list(point_tuple)].mean(axis=0)
            t = vor.points[point_tuple[1]] - vor.points[point_tuple[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            direction = np.sign(np.dot(midpoint - center, n)) * n
            x_min,x_max = 0,0
            (point_x0,point_y0),(point_x1,point_y1) = vor.points[point_tuple[0]],vor.points[point_tuple[1]]
            #垂直二等分線を求める.yの方程式で表せない場合、Noneを返す。
            f = per_bi(point_x0,point_y0,point_x1,point_y1)

            #verticeに-1を含む→半平面
            if((ver_list[0] == -1 or ver_list[1] == -1) and f):
                #iは-1が入っていない方
                i = (ver_list[0] == -1)
                ver_x = vor.vertices[ver_list[i]][0]
                #directionが正なら右に無限大
                (x_min,x_max) = (ver_x,float("inf")) if direction[0] > 0 else (-float("inf"),ver_x)
                (y_min,y_max) = (f(x_min),f(x_max)) if f(x_min) < f(x_max) else (f(x_max),f(x_min))
            #verticeに-1を含み、x=定数の形
            elif((ver_list[0] == -1 or ver_list[1] == -1) and f == None):
                #iは-1が入っていない方
                i = (ver_list[0] == -1)
                ver_y = vor.vertices[ver_list[i]][1]
                x_min,x_max = (vor.points[point_tuple[0]][0]+vor.points[point_tuple[1]][0])/2,(vor.points[point_tuple[0]][0]+vor.points[point_tuple[1]][0])/2
                #directionが正なら上むき
                y_min,y_max = (ver_y,float("inf")) if direction[1] > 0 else (-float("inf"),ver_y)

            #fがNoneじゃなくて、verticeに-1がない場合、閉じている→小さい方から大きい方
            #ver_listが-1を含まないならそれは線分 fがNoneならx=定数の線分
            elif(f):
                (x_min,x_max) = (vor.vertices[ver_list[0]][0],vor.vertices[ver_list[1]][0]) if (vor.vertices[ver_list[0]][0] < vor.vertices[ver_list[1]][0]) else (vor.vertices[ver_list[1]][0],vor.vertices[ver_list[0]][0])
                (y_min,y_max) = (f(x_min),f(x_max)) if f(x_min) < f(x_max) else (f(x_max),f(x_min))
            else:
                (x_min,x_max) = (vor.vertices[ver_list[0]][0],vor.vertices[ver_list[1]][0]) if (vor.vertices[ver_list[0]][0] < vor.vertices[ver_list[1]][0]) else (vor.vertices[ver_list[1]][0],vor.vertices[ver_list[0]][0])
                y_min,y_max = (vor.vertices[ver_list[0]][1],vor.vertices[ver_list[1]][1]) if  vor.vertices[ver_list[0]][1] < vor.vertices[ver_list[1]][1] else (vor.vertices[ver_list[1]][1],vor.vertices[ver_list[0]][1])

            if(f):
                for point in point_tuple:
                    eq_dict_dict[point].append({"equation":f,"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max})     

        #eq_dictの各要素をx_min順にsort
        sorted_eq_dict = {region_id: sorted(l,key=lambda x : x['x_min']) for region_id,l in eq_dict_dict.items()}

        #x_minが繋がっていない場合は変を追加していく。
        for region_id,eq_list in sorted_eq_dict.items():
            x_min = eq_list[0]["x_min"]
            flag = False
            for eq in eq_list[1:]:
                if x_min == eq["x_min"]:
                    if eq["equation"] != None:
                        flag = True #flag=1 のとき、追加する必要なし
                        break
                else:
                    break
            if(not flag):
                x_max = eq_list[-1]["x_max"]
                #追加する x_minからxの最大値までの線分（直線）y = inf(-inf)
                for i,eq in enumerate(eq_list):
                    if(eq["equation"]):
                        break
                if eq_list[i]["equation"](vor.points[map.ids.index(region_id)][0]) - vor.points[map.ids.index(region_id)][1] > 0:
                    sorted_eq_dict[region_id].append({"equation":(lambda x : -float("inf")),"x_min":x_min,"x_max":x_max,"y_min":-float("inf"),"y_max":-float("inf")}) 
                else:
                    sorted_eq_dict[region_id].append({"equation":(lambda x : float("inf")),"x_min":x_min,"x_max":x_max,"y_min":float("inf"),"y_max":float("inf")}) 

        #eq_dictの各要素をx_min順にsort
        return {region_id: sorted(l,key=lambda x : x['x_min']) for region_id,l in sorted_eq_dict.items()}

    def cp_dist(self, map, vor, sorted_eq_dict, epsilon):
        dist = {}
        count = 0
        for i,point in enumerate(vor.points):
            print(f"\r{count}/{len(map.ids)}", end="")
            count += 1
            dist_dict = {}
            array = []
            for key,eq_dict in sorted_eq_dict.items():
                suma = 0
                temp_eq = None
                for eq in eq_dict:
                    area = (0,0)
                #temp_eqが設定されるまで回す    
                    if(temp_eq):
                        #最初はeq["x_min"] == temp_x_minのはず
                        #temp_x_maxとeq["x_max"]を比べて小さい方まで積分をする。temp_x_maxの方が小さかったらtempは上書きされる。
                        if (eq["x_max"] > temp_x_max):
                            #print(f"{eq['x_min']}から{temp_x_max}まで積分")
                            area =  dblquad(laplace(point[1],point[0],epsilon),eq["x_min"],temp_x_max,temp_eq,eq["equation"])
                            #print(area)
                            temp_eq = eq["equation"]
                            temp_x_max = eq["x_max"]
                        else:
                            #print(f"{eq['x_min']}から{eq['x_max']}まで積分")
                            area = dblquad(laplace(point[1],point[0],epsilon),eq["x_min"],eq["x_max"],temp_eq,eq["equation"])                
                            #print(area)
                    else:
                        temp_x_max = eq["x_max"]
                        temp_eq = eq["equation"]     
                    suma += abs(area[0])
                dist_dict[key] = suma

            dist[map.ids[i]] = list(dist_dict.values())

        return np.array(list(dist.values()))
    
    
    def construct_distribution(self, map, epsilon):  
        vor = self._make_voronoi(map)
        sorted_eq_dict = self._make_vor_equations(vor, map)
        return self.cp_dist(map, vor, sorted_eq_dict, epsilon)