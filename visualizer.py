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
from matplotlib.collections import LineCollection


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

def add_maneuvers(f_l,v_l_mps):
    section_indxs = list(range(0,len(f_l),60))
    last_frame= 0 
    maneuver_list = []
    for start_frame, end_frame in zip(section_indxs, section_indxs[1:]):
        last_frame = end_frame
        current_vel_list = v_l_mps[start_frame:end_frame]
        if current_vel_list[-1] - current_vel_list[0] > 1:
            maneuver_list.append((start_frame,1))
        elif current_vel_list[-1] - current_vel_list[0] < -1:
            maneuver_list.append((start_frame,-1))
        else:
            maneuver_list.append((start_frame,0))
    
    current_vel_list = v_l_mps[last_frame:]
    if current_vel_list[-1] - current_vel_list[0] > 1:
        maneuver_list.append((last_frame,1))
    elif current_vel_list[-1] - current_vel_list[0] < -1:
        maneuver_list.append((last_frame,-1))
    else:
        maneuver_list.append((last_frame,0))
    return maneuver_list
        
def get_color(m):
    if m == -1:
        return 'g'
    elif m == 0:
        return 'b'
    else:
        return 'r'        
    
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
        #if k == 86:
            indx_list = detect_vel_changepoints(v)
            x_vals = range(len(v))
            rgb = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]
            tck = interpolate.splrep(x_vals, v, s=0.5, k =5)
            vnew = interpolate.splev(x_vals, tck, der=0)
            v_kph_list = mps_to_kph(v)
            maneuver_list = add_maneuvers(x_vals, v)
            lines,color_list = [],[]
            for indx,m in enumerate(maneuver_list):
                if indx < len(maneuver_list)-1:
                    curr_xs = x_vals[m[0]:maneuver_list[indx+1][0]]
                    curr_ys = v_kph_list[m[0]:maneuver_list[indx+1][0]]
                    line_segment = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(curr_xs[:-1], curr_ys[:-1], curr_xs[1:], curr_ys[1:])]
                    lines = lines + line_segment
                    color_list = color_list + [get_color(m[1])]*len(line_segment)
                else:
                    curr_xs = x_vals[m[0]:]
                    curr_ys = v_kph_list[m[0]:]
                    line_segment = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(curr_xs[:-1], curr_ys[:-1], curr_xs[1:], curr_ys[1:])]
                    lines = lines + line_segment
                    color_list = color_list + [get_color(m[1])]*len(line_segment)
            colored_lines = LineCollection(lines, colors=color_list, linewidths=(2,))
            fig, ax = plt.subplots(1)
            ax.add_collection(colored_lines)
            ax.autoscale_view()
            #for i in indx_list:
            #    plt.plot(x_vals[i],v[i],'bx')
            plt.show()
    conn.close()
#plot_accelerations()   


        

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
    
    