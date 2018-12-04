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

'''
x_list=[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7]
_x = list(range(0,len(x_list),4))
print(_x)'''
'''
Created on Sep 28, 2018

@author: atrisha
'''

import glob,os
import sqlite3
import mysql.connector
import math
import datetime
import numpy as np
from sympy.functions.elementary.piecewise import ExprCondPair

def rename_files():
    count = 1
    for infile in sorted(glob.glob('raw_csv_data/*')):
        print("Current File Being Processed is: " + infile + ' renamed to '+str(count))
        os.rename(infile,'raw_csv_data/'+str(count)+'.csv')
        count = count+1


def create_tables():
    beg = 201
    last = 5926
    start_num = beg
    end_num = start_num+99
    while end_num < last:
        print(str(start_num)+'_'+str(end_num))
        create_table = """ CREATE TABLE IF NOT EXISTS '"""+str(start_num)+'_'+str(end_num)+"""' (
        Vehicle_ID NUMERIC,
        Frame_ID NUMERIC,
        Total_Frames NUMERIC,
        Global_Time NUMERIC,
        Local_X NUMERIC,
        Local_Y NUMERIC,
        Global_X NUMERIC,
        Global_Y NUMERIC,
        v_length NUMERIC,
        v_Width NUMERIC,
        v_Class NUMERIC,
        v_Vel NUMERIC,
        v_Acc NUMERIC,
        Lane_ID NUMERIC,
        O_Zone TEXT,
        D_Zone TEXT,
        Int_ID TEXT,
        Section_ID TEXT,
        Direction TEXT,
        Movement TEXT,
        Preceding NUMERIC,
        Following NUMERIC,
        Space_Headway NUMERIC,
        Time_Headway NUMERIC,
        Location TEXT
        );"""
        start_num = end_num + 1
        end_num = start_num + 99
        
        conn = sqlite3.connect('db/ngsim.db')
        c = conn.cursor()
        c.execute(create_table)
        conn.commit()
        conn.close()


mydb = mysql.connector.connect(
  host="localhost",
  user="data_admin",
  passwd="password"
)

def max_ttc_diff(range,range_rate):
    max_dec = 9
    init_ttc = min(range/abs(range_rate),10)
    print(init_ttc)
    return init_ttc - min(int((range - ((range_rate * .1) + (0.5 * max_dec * .01))) / (range_rate + (max_dec * .1))),10)

import csv
import matplotlib.pyplot as plt
def show_convergence():
    last_rate = 0
    conv_list = []
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/mc_results.out', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            rate = round(float(row[1])/float(row[0]),6)
            if last_rate != 0:
                diff = abs(rate - last_rate)
                conv_list.append(diff)
            last_rate = rate
    
    plt.plot(np.arange(99),conv_list,)
    plt.show()



def print_first100_interac():
    count = 0
    min = 0
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/interac_seq_data.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] != '':
                if float(row[2]) < min:
                    min = float(row[2])
                print(row)
                if count == 100:
                    break
            count = count + 1
    print(min)

def util_dist(dist_m,param):
    thresh = param
    x = dist_m
    return 1.5/(1+math.exp(-1 * (x-thresh))) - 0.5
    
    
def util_ttc(ttc_sec,param):
    thresh = param
    x = ttc_sec
    return 1.5/(1+math.exp(-1 * (x-thresh))) - 0.5
    
    
    
def util_progress(vel_mps):
    x = vel_mps
    return 1.5/(1+math.exp(-1 * (x-1))) - 0.5 if x<=30 else -0.05*x + 2.5


def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def plot_all_utils(): 
    from mpl_toolkits.mplot3d import Axes3D   
    vel_s = 20
    
    X,Y,Z = [],[],[]
    
    for e in zip(np.arange(0,50,5),np.arange(0,100,10)):
        vel_lc,range = e[0],e[1]
        if vel_lc != vel_s:
            util_ttc_param = 2.5
            X.append(util_progress(vel_lc))
            Y.append(util_ttc(range/(vel_s-vel_lc),util_ttc_param))
            util_dist_param = 50
            Z.append(util_dist(range,util_dist_param))
    plt.plot(np.arange(0,50,.5),[util_progress(x) for x in np.arange(0,50,.5)])        
    pareto_array = np.asarray(list(zip(X,Y,Z)))
    pareto_array.reshape((len(X),3))
    pareto_points = is_pareto_efficient(pareto_array)
    X_pareto,Y_pareto,Z_pareto = [],[],[]
    for i in enumerate(X):
        if pareto_points[i[0],]:
            X_pareto.append(X[i[0]])
            Y_pareto.append(Y[i[0]])
            Z_pareto.append(Z[i[0]])
            
            
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('utilvel_lc')
    ax.set_ylabel('util_ttc')
    ax.set_zlabel('util_range')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.scatter3D(X,Y,Z)
    ax.scatter3D(X_pareto,Y_pareto,Z_pareto,color='green')
    plt.show()

print_first100_interac()


'''
Y = np.arange(-1000,1000)
for y in Y:
    soln = np.exp(0.5*y)/(np.exp(y)/y - 1/y)
    print(y,soln)
'''
