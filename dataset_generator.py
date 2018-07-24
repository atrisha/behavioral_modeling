'''
Created on Jul 19, 2018

@author: Atrisha
'''
import utils
import numpy as np
import sqlite3
from utils import is_lane_clear, calc_ttc, dist_2_regulatory_stop
import os, shutil
import csv

def wrap_none(val):
    if val is None:
        return 'NA'
    else:
        return val

def clear_data_folder():
    folder = 'data'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def generate_dataset():
    conn = sqlite3.connect('db/trajectories.db')
    c = conn.cursor()
    vehicle_ids = range(67,101)
    c.execute("select * from tracks order by frame_id")
    d_list = c.fetchall()
    data_array = np.array(d_list)
    other_info = {'ttc':{}}
    clear_data_folder()
    for vehicle in vehicle_ids:
        other_info['ttc'][vehicle] = dict()
        curr_vehicle_data = data_array[np.where( data_array[:,1] == str(vehicle) )]
        frame_ids = curr_vehicle_data[:,4].flatten().tolist()
        with open('DATA/'+str(vehicle)+'.csv', 'w', newline='') as csvfile:
            data_writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for current_frame in frame_ids:
                
                vehicle_info = dict()
                vehicle_info['id'] = str(vehicle)
                vehicle_info['location'] = (np.asscalar(curr_vehicle_data[np.where( curr_vehicle_data[:,4] == str(current_frame) )][:,2]),np.asscalar(curr_vehicle_data[np.where( curr_vehicle_data[:,4] == str(current_frame) )][:,5]))
                vehicle_info['lane_id'] = np.asscalar(curr_vehicle_data[np.where( curr_vehicle_data[:,4] == str(current_frame) )][:,7])
                vehicle_info['velocity'] = float(np.asscalar(curr_vehicle_data[np.where( curr_vehicle_data[:,4] == str(current_frame) )][:,8]))
                env_data = data_array[np.where( (data_array[:,1] != str(vehicle)) & (data_array[:,4] == str(current_frame)))].tolist()
                env_info = dict()
                for v in env_data:
                    env_info[v[1]] = dict()
                    env_info[v[1]]['id'] = v[1]
                    env_info[v[1]]['location'] = (v[2],v[5])
                    env_info[v[1]]['lane_id'] = v[7]
                    env_info[v[1]]['velocity'] = float(v[8])
                if vehicle_info['id'] == '69':
                    g=5
                is_lane,leading_ttc = is_lane_clear(vehicle_info,env_info)
                ttc_dict,dist_gap_dict = calc_ttc(vehicle_info,env_info)
                other_info['ttc'][vehicle] = ttc_dict
                min_ttc = None if not bool(ttc_dict) else min(list(ttc_dict.values()))
                min_dist_gap = None if not bool(dist_gap_dict) else min(list(dist_gap_dict.values()))
                dist_tuples = dist_2_regulatory_stop(vehicle_info,env_info,other_info)
                print(vehicle,',',vehicle_info['lane_id'],',',current_frame,':',is_lane,leading_ttc,min_ttc,min_dist_gap,dist_tuples[0],dist_tuples[1])
                data_writer.writerow([current_frame,wrap_none(is_lane),wrap_none(leading_ttc),wrap_none(min_ttc),wrap_none(min_dist_gap),wrap_none(dist_tuples[0]),wrap_none(dist_tuples[1])])
generate_dataset()       