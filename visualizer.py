'''
Created on Jul 25, 2018

@author: Atrisha
'''
import sqlite3
import matplotlib.pyplot as plt
import random
from scipy import interpolate
import numpy as np
import math
from urllib.request import parse_keqv_list

fps = 30


def detect_vel_changepoints(vel_list):
    indx_list = []
    for k,v in enumerate(vel_list):
        if k+30 < len(vel_list):
            delta = max(vel_list[k:k+30]) - min(vel_list[k:k+30])
        else:
            delta = max(vel_list[k:]) - min(vel_list[k:])
        if abs(delta) > 1.5:
            indx_list.append(k)
    return indx_list

def mps_to_kph(v_list):
    return [v * 3.6 for v in v_list] 
    
def plot_accelerations():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select vehicle_id,frame_id,velocity from tracks")
    d_list = c.fetchall()
    acc_list,vel_list = dict(),dict()
    for indx,d in enumerate(d_list):
        veh_id = d[0]
        if veh_id not in acc_list:
            acc_list[veh_id] = []
            vel_list[veh_id] = []
        if indx < len(d_list)-1:
            acceleration = (float(d_list[indx+1][2]) - float(d[2]))/(1/fps)
            acc_list[veh_id].append(acceleration)
        else:
            acc_list[veh_id].append(acc_list[veh_id][-1])
        vel_list[veh_id].append(float(d[2]))
    for k,v in vel_list.items():
        if k == 86:
            indx_list = detect_vel_changepoints(v)
            x_vals = range(len(v))
            rgb = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]
            tck = interpolate.splrep(x_vals, v, s=0.5, k =5)
            vnew = interpolate.splev(x_vals, tck, der=0)
            plt.plot(x_vals,mps_to_kph(v),'b')
            #for i in indx_list:
            #    plt.plot(x_vals[i],v[i],'bx')
    plt.show()
    conn.close()
plot_accelerations()   


        

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def plot_path():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    c.execute("select vehicle_id,frame_id,location_utm_x,location_utm_y from tracks")
    d_list = c.fetchall()
    x_list,y_list= [],[]
    for indx,d in enumerate(d_list):
        veh_id = d[0]
        if int(veh_id) == 99:
            x_list.append(d[2])
            y_list.append(d[3])
    tck = interpolate.splrep(x_list, y_list, s=0, k =2)
    x,y = x_list,y_list
    xnew = x_list    
    ynew = interpolate.splev(xnew, tck, der=0)
    plt.figure()
    plt.plot(x, y, 'bx')
    plt.plot(xnew,ynew,color='r')
    curv_list = []
    for _x in xnew:
        y_2prime = interpolate.splev(_x,tck,der=2)
        y_prime = interpolate.splev(_x,tck,der=1)
        curvature = y_2prime / ( math.pow( (1+(y_prime**2)) , (3/2) ))
        curv_list.append(1/curvature)
    _x = range(0,len(x_list)-1,100)
    _curve_chuck_sum = list(chunks(curv_list,100))
    for i,j in enumerate(list(zip([xnew[i] for i in _x],[ynew[i] for i in _x]))):
        v_2 = 9.8 * (sum(_curve_chuck_sum[i])/len(_curve_chuck_sum[i]))
        plt.text(j[0],j[1],math.sqrt(abs(v_2)))
    plt.show()
    
    conn.close()
#plot_path()
    
    