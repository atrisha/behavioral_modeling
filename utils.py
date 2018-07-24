'''
Created on Jul 4, 2018

@author: Atrisha
'''

import sqlite3
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from prompt_toolkit.key_binding.bindings.named_commands import complete
from math import inf
from sumolib.net import lane



A,B,C,D,E,F = 0.166,0,1100536.44,0,-0.166,5526424.60
lane_clear_threshold_mts = 20
lanes_with_conflict_points = [37, 38, 39, 40, 41, 42, 43, 52, 54, 55, 56, 57, 58, 58]
multi_path_lanes = ['52','56']
collision_polygon_dict = {1: ((511, -324), 8.167807294796283), 2: ((498, -211), 8.270066022027406), 3: ((251, -230), 8.872851401890749), 4: ((282, -351), 8.374366841326635)}

def px_2_utm(pt_px_tuple):
    x,y = pt_px_tuple[0],pt_px_tuple[1]
    x_prime = A*float(x) + B*float(y) + C
    y_prime = D*float(x) + E*float(y) + F
    return (x_prime,y_prime)

def calc_dist(pt_px1,pt_px2):
    pt_px1,pt_px2 = px_2_utm(pt_px1),px_2_utm(pt_px2)
    return math.sqrt((float(pt_px2[0])-pt_px1[0])**2 + (float(pt_px2[1])-pt_px1[1])**2)

def get_future_path_in_lane(location,lane_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    if lane_id in multi_path_lanes:
        lane_path_ids = [lane_id+'_1',lane_id+'_2']
        c.execute("select * from paths where lane_id='"+lane_path_ids[0]+"' or lane_id='"+lane_path_ids[1]+"'")
        d_list = c.fetchall()
        path_dict,dist_to_complete_path_points,val_idx = dict(),dict(),dict()
        for d in d_list:
            if d[0] not in path_dict:
                path_dict[d[0]] = [(float(d[1]),float(d[2]))]
            else:
                path_dict[d[0]].append((float(d[1]),float(d[2])))
        for k,v in path_dict.items():
            dist_to_complete_path_points[k] = list(map(lambda x : calc_dist(location,x),v))
            val_idx[k] = min((val, idx) for (idx, val) in enumerate(dist_to_complete_path_points[k]))
        closest_lane_path_id = min(val_idx, key=val_idx.get)
        future_path = [location] + path_dict[closest_lane_path_id][val_idx[closest_lane_path_id][1]:]
        return future_path
    else:
        c.execute("SELECT path_pt_px_x,path_pt_px_y FROM paths where lane_id="+str(lane_id)+"")
        d_list = c.fetchall()
        data_array = np.array(d_list).astype(np.float16)
        complete_path = list(zip(data_array[:,0],data_array[:,1]))
        dist_to_complete_path_points = list(map(lambda x : calc_dist(location,x),complete_path))
        val, idx = min((val, idx) for (idx, val) in enumerate(dist_to_complete_path_points))
        future_path = [location] + complete_path[idx:]
        return future_path

def get_lane_path(lane_id):
    if str(lane_id) in multi_path_lanes:
        lane_id = str(lane_id)+'_1'
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("SELECT path_pt_px_x,path_pt_px_y FROM paths where lane_id='"+str(lane_id)+"'")
    d_list = c.fetchall()
    data_array = np.array(d_list).astype(np.float16)
    complete_path = list(zip(data_array[:,0],data_array[:,1]))
    return complete_path
    
    
def get_lane(location_tuple_px):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("SELECT * FROM lanes ORDER BY lane_id,point_index")
    d_list = c.fetchall()
    data_array = np.array(d_list)
    all_lane_ids = range(37,59)
    lane_id = -1
    for l in all_lane_ids:
        curr_lane = list(zip(data_array[np.where(data_array[:,0]==l)][:,2],data_array[np.where(data_array[:,0]==l)][:,3]))
        polygon = Polygon(curr_lane)
        if polygon.contains(Point(float(location_tuple_px[0]),float(location_tuple_px[1]))):
            lane_id = l
            break
    conn.close()
    return lane_id

def get_complete_path(vehicle_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("SELECT location_px_x,location_px_y FROM tracks where vehicle_id='"+str(vehicle_id)+"'")
    d_list = c.fetchall()
    data_array = np.array(d_list)
    complete_path = list(zip(data_array[:,0],data_array[:,1]))
    return complete_path

def get_future_lanes(vehicle_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select distinct lane_id from tracks where tracks.vehicle_id="+str(vehicle_id)+" order by tracks.frame_id")
    d_list = c.fetchall()
    conn.close()
    d_list = [d[0][0] for d in d_list]
    return d_list
'''
Return in meters. Input path is in pixel co_ordinate
'''
def calc_path_length_mts(path_px):
    path_px = [px_2_utm((float(i[0]),float(i[1]))) for i in path_px]
    length = sum([math.sqrt((p[1][0] - p[0][0])**2 + (p[1][1]-p[0][1])**2) for p in zip(path_px[:-1],path_px[1:])])
    return length

def calc_dist_along_path(pt1_px,pt2_px,path_px):
    pt1_px = (float(pt1_px[0]),float(pt1_px[1]))
    pt2_px = (float(pt2_px[0]),float(pt2_px[1]))
    path_px = [(float(p[0]),float(p[1])) for p in path_px]
    dist_to_complete_path_points = list(map(lambda x : calc_dist(pt1_px,x),path_px))
    val, idx = min((val, idx) for (idx, val) in enumerate(dist_to_complete_path_points))
    index_to_closest_point_from_1 = idx
    dist_to_complete_path_points = list(map(lambda x : calc_dist(pt2_px,x),path_px))
    val, idx = min((val, idx) for (idx, val) in enumerate(dist_to_complete_path_points))
    index_to_closest_point_from_2 = idx
    if index_to_closest_point_from_2 > index_to_closest_point_from_1:
        return calc_path_length_mts(path_px[index_to_closest_point_from_1:index_to_closest_point_from_2])
    elif index_to_closest_point_from_2 < index_to_closest_point_from_1:
        return calc_path_length_mts(path_px[index_to_closest_point_from_2:index_to_closest_point_from_1])*-1
    else:
        return 0
             
def get_next_lane(veh_lane_id,veh_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select distinct lane_id from tracks where vehicle_id="+str(veh_id)+" order by frame_id")
    d_list = c.fetchall()
    d_list = [p[0] for p in d_list]
    indx = d_list.index(int(veh_lane_id))
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
    
def calc_leading_ttc(dist,sub_veh_vel,leading_vehicle_vel):
    ttc = None
    rel_vel = max(0,sub_veh_vel - leading_vehicle_vel)
    if rel_vel > 0:
        ttc = dist/rel_vel
    return ttc

''' Also calculates ttc if the lane is not clear '''
def is_lane_clear(vehicle_info,env_info):
    veh_location = vehicle_info['location']
    veh_lane_id = vehicle_info['lane_id']
    veh_velocity = vehicle_info['velocity']
    future_path_in_lane = get_future_path_in_lane(veh_location, veh_lane_id)
    length_of_future_path_mt = calc_path_length_mts(future_path_in_lane)
    next_lane_id = get_next_lane(veh_lane_id,vehicle_info['id'])
    if next_lane_id is not None:
        future_path_in_next_lane = get_lane_path(next_lane_id)
        all_future_path = future_path_in_lane + future_path_in_next_lane
    else:
        all_future_path = future_path_in_lane
    for k,v in env_info.items():
        if v['lane_id'] == veh_lane_id or v['lane_id'] == next_lane_id:
            g=5
            dist_to_vehicle = calc_dist_along_path(veh_location,v['location'],all_future_path)
            if 0 < dist_to_vehicle <= lane_clear_threshold_mts :
                return False,calc_leading_ttc(dist_to_vehicle,veh_velocity,v['velocity'])
    return True,None

def get_collision_lanes_from_lane(lane_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select lane_1,lane_2 from lane_collision_map where lane_collision_map.lane_1="+str(lane_id)+" or lane_collision_map.lane_2="+str(lane_id))
    d_list = c.fetchall()
    data_array = np.array(d_list)
    collision_list = data_array[np.where( data_array != lane_id )].flatten().tolist()
    conn.close()
    return collision_list

def get_collision_lanes_from_polygon(polygon_id):
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select lane_1,lane_2 from lane_collision_map where collision_polygon_id="+str(polygon_id))
    d_list = c.fetchall()
    data_array = np.array(d_list)
    collision_list = data_array.flatten().tolist()
    conn.close()
    return collision_list

def contains(pt_px,collision_polygon_dict):
    for k,c in collision_polygon_dict.items():
        if calc_dist(pt_px, c[0]) < c[1]:
            return k
    return None

def get_conflict_polygon_for_point(path,collision_polygon_dict):
    for point_px in path:
        polygon_id = contains(point_px,collision_polygon_dict)
        if polygon_id is not None:
            return polygon_id
    return None
    '''for point_px in path:
        for k,v in collision_polygon_dict.items():
            if Polygon(v).contains(Point(point_px[0],point_px[1])):
                return k
    return None'''


def dist_2_polygons(path,polygon_id):
    '''
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select point_location_px_x,point_location_px_y from collision_polygons where collision_polygon_id = "+str(polygon_id))
    d_list = c.fetchall()
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
        entry_dist = sum([calc_dist(p[0],p[1]) for p in zip(entry_path[:-1],entry_path[1:])])
    if exit_index is not None:
        exit_path = path[:exit_index+1]
        exit_dist = sum([calc_dist(p[0],p[1]) for p in zip(exit_path[:-1],exit_path[1:])])
    return entry_dist,exit_dist
    '''
    entry_dist,exit_dist = inf,inf
    entry_index,exit_index = None,None
    for indx,pt in enumerate(path):
        if calc_dist(pt, collision_polygon_dict[polygon_id][0]) < collision_polygon_dict[polygon_id][1]:
            entry_index = indx
            break
    if entry_index is not None:
        path_for_exit = path[entry_index:]
        for indx,pt in enumerate(path_for_exit):
            if not calc_dist(pt, collision_polygon_dict[polygon_id][0]) < collision_polygon_dict[polygon_id][1]:
                exit_index = indx + entry_index
                break
    else:
        exit_index = None
    if entry_index is not None:
        entry_path = path[:entry_index+1]
        entry_dist = sum([calc_dist(p[0],p[1]) for p in zip(entry_path[:-1],entry_path[1:])])
    if exit_index is not None:
        exit_path = path[:exit_index+1]
        exit_dist = sum([calc_dist(p[0],p[1]) for p in zip(exit_path[:-1],exit_path[1:])])
    return entry_dist,exit_dist
    
    
def get_overlap(ti_1, ti_2):
    if max(0, min(ti_1[1], ti_2[1]) - max(ti_1[0], ti_2[0])) > 0:
        return max(ti_1[0], ti_2[0])
    else:
        return inf    

def calc_ttc(vehicle_info,env_info):
    ttc_dict = dict()
    dist_gap_dict = dict()
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    veh_location = vehicle_info['location']
    veh_lane_id = vehicle_info['lane_id']
    veh_velocity = float(vehicle_info['velocity'])
    veh_id = int(vehicle_info['id'])
    c.execute("SELECT * FROM collision_polygons")
    d_list = c.fetchall()
    
    '''
    changes for polygon to circle:
    for e in d_list:
        if e[0] not in collision_polygon_dict:
            collision_polygon_dict[e[0]] = [(e[1],e[2])]
        else:
            collision_polygon_dict[e[0]].append((e[1],e[2]))
    '''
    c.execute("SELECT location_px_x,location_px_y,lane_id,vehicle_id FROM tracks")
    d_list = c.fetchall()
    data_array = np.array(d_list)
    ''' truncate future path only from current locations '''
    sub_veh_data_array = data_array[np.where( data_array[:,3] == veh_id )]
    complete_path = list(zip(sub_veh_data_array[:,0],sub_veh_data_array[:,1]))
    dist_to_complete_path_points = list(map(lambda x : calc_dist(veh_location,x),complete_path))
    val, idx = min((val, idx) for (idx, val) in enumerate(dist_to_complete_path_points))
    sub_veh_data_array = sub_veh_data_array[idx:]
    
    
    future_path_and_lanes = zip(list(zip(sub_veh_data_array[:,0],sub_veh_data_array[:,1])),sub_veh_data_array[:,2])
    future_lanes = sub_veh_data_array[:,2]
    
    '''
    First check if the future lanes (including the current one) the vehicle would be has any conflict point or not.
    '''
    future_conflict_lanes = list(set(future_lanes) and set(lanes_with_conflict_points))
    '''
    1. Find the next conflict polygon the vehicle would be in (assuming that the conflict polygons are mutually disjoint)
    2. Find the lanes (other than the current lane vehicle is in) that conflict with this polygon
    3. For all other vehicles in the environment check if any vehicle is in those lanes or not
    4. For those vehicles check if their future path would fall in the polygon from 1
    5. If so, then calculate the time of entry and exit with respect to the polygon
    '''
    for p in future_path_and_lanes:
        if p[1] in future_conflict_lanes:
            sub_veh_future_path = [p[0] for p in future_path_and_lanes]
            ''' 1 '''
            conflict_polygon_id = get_conflict_polygon_for_point(sub_veh_future_path,collision_polygon_dict)
            if conflict_polygon_id is not None:
                
                sub_veh_entry_dist,sub_veh_exit_dist = dist_2_polygons(sub_veh_future_path,conflict_polygon_id)
                sub_veh_entry_time,sub_vehicle_exit_time = sub_veh_entry_dist/float(veh_velocity),sub_veh_exit_dist/float(veh_velocity)
                ''' 2 '''
                collision_lanes = get_collision_lanes_from_polygon(conflict_polygon_id)
                collision_lanes = list(filter(lambda a: a != int(p[1]), collision_lanes))
                ''' 3 '''    
                for k,v in env_info.items():
                    oth_veh_id = k
                    oth_veh_location = v['location']
                    oth_veh_data_array = data_array[np.where( data_array[:,3] == float(oth_veh_id) )]
                    complete_path = list(zip(oth_veh_data_array[:,0],oth_veh_data_array[:,1]))
                    dist_to_complete_path_points = list(map(lambda x : calc_dist(oth_veh_location,x),complete_path))
                    val, idx = min((val, idx) for (idx, val) in enumerate(dist_to_complete_path_points))
                    oth_veh_data_array = oth_veh_data_array[idx:]
                    oth_veh_future_path_and_lanes = list(zip(list(zip(oth_veh_data_array[:,0],oth_veh_data_array[:,1])),oth_veh_data_array[:,2]))
                    oth_veh_future_path = [p[0] for p in oth_veh_future_path_and_lanes]
                    for indx,f_p_l in enumerate(oth_veh_future_path_and_lanes):
                        ''' 3 '''
                        if f_p_l[1] in collision_lanes:
                            oth_veh_conflict_polygon = get_conflict_polygon_for_point(oth_veh_future_path, collision_polygon_dict)
                            ''' 4 '''
                            if oth_veh_conflict_polygon == conflict_polygon_id:
                               
                                '''
                                5. calc_time to entry and exit
                                '''
                                oth_veh_entry_dist,oth_veh_exit_dist = dist_2_polygons(oth_veh_future_path,conflict_polygon_id)
                                oth_veh_entry_time,oth_vehicle_exit_time = oth_veh_entry_dist/v['velocity'],oth_veh_exit_dist/v['velocity']
                                '''
                                if this time to entry and exit intersect the other time to entry and exit, then that intersection is the ttc value for this other vehicle.
                                '''
                                veh_ttc = get_overlap((sub_veh_entry_time,sub_vehicle_exit_time), (oth_veh_entry_time,oth_vehicle_exit_time))
                                ttc_dict[oth_veh_id] = veh_ttc
                                dist_gap_dict[oth_veh_id] = oth_veh_entry_dist + sub_veh_entry_dist
            return ttc_dict,dist_gap_dict                            
    conn.close()
    return None,None
    
    
    
def dist_2_regulatory_stop(vehicle_info,env_info,meta_info):    
    veh_location = vehicle_info['location']
    veh_lane_id = vehicle_info['lane_id']
    veh_velocity = float(vehicle_info['velocity'])
    veh_id = int(vehicle_info['id'])
    entry_lanes = [37,38,42,43,39,40,41]
    yielding_precendence = {37:[56],38:[56],42:[57],43:[58],39:[52],40:[52],41:[52,54]}
    exit_lanes = [44,45,46,48,49,47]
    on_roundabout = [56,57,58,52,54,55]
    stopping_point_dict = {37:1,38:1,42:2,43:2,39:3,40:3,41:4}
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    dist_to_reg_stops,dist_to_roundabout = None,None
    if int(veh_lane_id) in yielding_precendence:
        for k,v, in env_info.items():
            pt1_px = veh_location
            pt2_px = collision_polygon_dict[stopping_point_dict[int(veh_lane_id)]][0]
            radius = collision_polygon_dict[stopping_point_dict[int(veh_lane_id)]][1]
            if str(veh_lane_id) in multi_path_lanes:
                veh_lane_id = str(veh_lane_id) + "_1"
            c.execute("SELECT location_px_x,location_px_y FROM tracks where vehicle_id="+str(veh_id)+" order by frame_id")
            d_list = c.fetchall()
            dist_to_roundabout = calc_dist_along_path(pt1_px, pt2_px, d_list) - radius
            if int(v['lane_id']) in yielding_precendence[int(veh_lane_id)] and meta_info['ttc'][veh_id][str(v['id'])] < 7:   
                dist_to_reg_stops = dist_to_roundabout
    conn.close()
    if int(veh_lane_id) in exit_lanes:
        dist_to_roundabout = -2
    if int(veh_lane_id) in on_roundabout:
        dist_to_roundabout = -1
    return (dist_to_reg_stops,dist_to_roundabout)
                
                
            
    
                
       
