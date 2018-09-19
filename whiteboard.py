'''
Created on Jul 16, 2018

@author: Atrisha
'''

import sqlite3
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from math import inf

def a():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("SELECT location_px_x,location_px_y,lane_id,vehicle_id FROM tracks")
    d_list = c.fetchall()
    data_array = np.array(d_list)
    x = data_array[np.where( data_array[:,3] == 67 )]
    conn.close()
    
def b():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("SELECT * FROM collision_polygons")
    d_list = c.fetchall()
    collision_polygon_dict = dict()
    for e in d_list:
        if e[0] not in collision_polygon_dict:
            collision_polygon_dict[e[0]] = [(e[1],e[2])]
        else:
            collision_polygon_dict[e[0]].append((e[1],e[2]))  
    print(collision_polygon_dict)  
    
def get_collision_lanes(lane_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select lane_1,lane_2 from lane_collision_map where lane_collision_map.lane_1="+str(lane_id)+" or lane_collision_map.lane_2="+str(lane_id))
    d_list = c.fetchall()
    data_array = np.array(d_list)
    collision_list = data_array[np.where( data_array != lane_id )].flatten().tolist()
    conn.close
    print(collision_list)
    
def c():
    entry_path = [(1,1),(2,2),(3,3),(4,4)]
    a =  list(zip(entry_path[:-1],entry_path[1:]))
    b = [math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][0])**2) for p in a]
    c = sum(b)
    print(c)
    
def dist_2_polygons(path,polygon_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select point_location_px_x,point_location_px_y from collision_polygons where collision_polygon_id = "+str(polygon_id))
    d_list = c.fetchall()
    d_list = [(-1,-1),(1.5,-1),(1.5,1.5),(-1,1.5)]
    polygon = Polygon(d_list)
    

    
    entry_dist,exit_dist = inf,inf
    entry_index,exit_index = None,None
    for indx,pt in enumerate(path):
        point = Point(pt[0],pt[1])
        if polygon.contains(point):
            entry_index = indx
            break
    if entry_index is not None:
        path_for_exit = path[entry_index:]
        for indx,pt in enumerate(path_for_exit):
            point = Point(pt[0],pt[1])
            if not polygon.contains(point):
                exit_index = indx + entry_index
                break
    else:
        exit_index = None
    if entry_index is not None:
        entry_path = path[:entry_index+1]
        entry_dist = sum([math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][0])**2) for p in zip(entry_path[:-1],entry_path[1:])])
    if exit_index is not None:
        exit_path = path[:exit_index+1]
        exit_dist = sum([math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1] - p[0][0])**2) for p in zip(exit_path[:-1],exit_path[1:])])
    return entry_dist,exit_dist
    
#print(dist_2_polygons([(1,1),(2,2),(3,3),(4,4),(5,5)], 4))

def getOverlap(ti_1, ti_2):
    if max(0, min(ti_1[1], ti_2[1]) - max(ti_1[0], ti_2[0])) > 0:
        return max(ti_1[0], ti_2[0])
    else:
        return inf  

#print(getOverlap((10,15), (20,21)))
#print(list(range(67,101)))

def get_next_lane(veh_lane_id,veh_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select distinct lane_id from tracks where vehicle_id="+str(veh_id)+" order by frame_id")
    d_list = c.fetchall()
    d_list = [p[0] for p in d_list]
    indx = d_list.index(veh_lane_id)
    if indx == len(d_list) - 1:
        return None
    else:
        next_lane_indx = indx + 1
        next_lane = d_list[next_lane_indx]
        if next_lane == -1:
            if next_lane_indx == len(d_list) - 1:
                return None
            else:
                next_lane_indx = next_lane_indx + 1
                next_lane = d_list[next_lane_indx]
    conn.close()
    return next_lane
#print(get_next_lane(58, 69))
A,B,C,D,E,F = 0.166,0,1100536.44,0,-0.166,5526424.60
def px_2_utm(pt_px_tuple):
    x,y = pt_px_tuple[0],pt_px_tuple[1]
    x_prime = A*float(x) + B*float(y) + C
    y_prime = D*float(x) + E*float(y) + F
    print((x_prime,y_prime))
    return (x_prime,y_prime)

def calc_dist(pt_px1,pt_px2):
    pt_px1,pt_px2 = px_2_utm(pt_px1),px_2_utm(pt_px2)
    return math.sqrt((float(pt_px2[0])-pt_px1[0])**2 + (float(pt_px2[1])-pt_px1[1])**2)

'''
collision_polygon_dict = dict()
cnb = [((511,-324), (481,-363)),((498,-211) , (507,-260)),((251,-230) , (267,-281)),((282,-351) , (270,-302))]
for i,p in enumerate(cnb):
    dist = calc_dist(p[0], p[1])
    collision_polygon_dict[i+1] = (p[0],dist)
print(collision_polygon_dict)
'''
x_list=[1,2,3,4,5,6,7,8,9,10]
_x = list(range(0,len(x_list)-1,5))
print(_x)