'''
Created on Oct 25, 2018

@author: atrisha
'''

import mysql.connector
from os import listdir
from os.path import isfile, join
import csv
import time
import datetime
import numpy as np
from utils import root_path
from utils import data_path

def flush_to_file(data_buffer,dir_path):
    for key,val in data_buffer.items():
        out_file_name = key+'.csv'
        with open(dir_path+out_file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for row in val:
                    writer.writerow(list(row))
    data_buffer = dict()
    return data_buffer
    

def generate_device_list_files():
    dir_path = root_path
    file = 'device_trip_list.csv'
    line_count = 0
    device_trip_list = []
    with open(dir_path+str(file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_count !=0:
                device_trip_list.append(row)
            line_count = line_count + 1
    dir_path = data_path+'DataLane/csv_files/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    data_buffer = dict()
    num_records = 0
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100)
        line_count = 0
        with open(dir_path+str(file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:   
                if line_count !=0:
                    if str(row[0])+'-'+str(row[1]) in data_buffer.keys():
                        data_buffer[str(row[0])+'-'+str(row[1])].append(row)
                    else:
                        data_buffer[str(row[0])+'-'+str(row[1])] = [row]
                line_count = line_count + 1
        data_buffer = flush_to_file(data_buffer,root_path+'device_trip_data/')

def generate_wsu_devise_files():
    dir_path = root_path
    file = 'device_trip_list.csv'
    line_count = 0
    device_trip_list = []
    with open(dir_path+str(file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_count !=0:
                device_trip_list.append(row)
            line_count = line_count + 1
    dir_path = data_path+'DataWsu/csv_files/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    data_buffer = dict()
    num_records = 0
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100)
        line_count = 0
        with open(dir_path+str(file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:   
                if line_count !=0:
                    if str(row[0])+'-'+str(row[1]) in data_buffer.keys():
                        data_buffer[str(row[0])+'-'+str(row[1])].append(row)
                    else:
                        data_buffer[str(row[0])+'-'+str(row[1])] = [row]
                line_count = line_count + 1
        data_buffer = flush_to_file(data_buffer,root_path+'wsu_data/')


def generate_front_target_devise_files():
    dir_path = root_path
    file = 'device_trip_list.csv'
    line_count = 0
    device_trip_list = []
    with open(dir_path+str(file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_count !=0:
                device_trip_list.append(row)
            line_count = line_count + 1
    dir_path = data_path+'DataFrontTargets/csv_files/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    data_buffer = dict()
    num_records = 0
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100)
        line_count = 0
        with open(dir_path+str(file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:   
                if line_count !=0:
                    if str(row[0])+'-'+str(row[1]) in data_buffer.keys():
                        data_buffer[str(row[0])+'-'+str(row[1])].append(row)
                    else:
                        data_buffer[str(row[0])+'-'+str(row[1])] = [row]
                line_count = line_count + 1
        data_buffer = flush_to_file(data_buffer,root_path+'front_target_data/')


def check_for_lane_change(data_window,for_left_side):
    one_sec_array = np.asarray(data_window, dtype=np.float)
    if for_left_side:
        if not np.all(one_sec_array[:,7]>0, axis=0):
            return False,None
    else:
        if not np.all(one_sec_array[:,6]>0, axis=0):
            return False,None
    if for_left_side:
        lane_distances = [abs(x) for x in list(one_sec_array[:,3])]
    else:
        lane_distances = [abs(x) for x in list(one_sec_array[:,3])]
    lane_change = False
    cross_time = None
    ''' if the change of lane is between 2 and 4 meters'''
    if 2 <= abs(max(lane_distances) - min(lane_distances)) <= 4:
        lane_change = True
        cross_time = one_sec_array[lane_distances.index(abs(min(lane_distances))),2]
    return lane_change,cross_time

def flush_to_lan_change_file(lane_change_buffer):
    dir_path = root_path
    out_file_name = 'lane_change_event.csv'
    with open(dir_path+out_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in lane_change_buffer:
                writer.writerow(list(row))
                

def detect_lane_change_events():
    dir_path = root_path+'device_trip_data/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    
    num_records = 0
    two_wheel_lane_change = False
    lane_change_buffer = []
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100,len(lane_change_buffer))
        data_buffer = []
        with open(dir_path+str(file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data_buffer.append(row)
        if len(data_buffer) >=10:
            cross_time_l,cross_time_r = None,None
            for ind in np.arange(len(data_buffer)-10):
                one_second_data = data_buffer[ind:ind+10]
                one_wheel_change,time_1 = check_for_lane_change(one_second_data,True)
                if one_wheel_change:
                    two_sec_data = data_buffer[max(0,ind-10):min(len(data_buffer)-1,ind+10)]
                    two_wheel_change,time_2 = check_for_lane_change(two_sec_data,False)
                    if two_wheel_change and abs(time_1 - time_2) >=1:
                        cross_time = time_1
                        line_entry = [data_buffer[ind][0],data_buffer[ind][1],cross_time]
                        if line_entry not in lane_change_buffer:
                            lane_change_buffer.append(line_entry)
                            if len(lane_change_buffer) > 5000:
                                flush_to_lan_change_file(lane_change_buffer)
                                lane_change_buffer = []
                                
                            #print(line_entry)
                        
        
    dir_path = root_path
    out_file_name = 'lane_change_event.csv'
    with open(dir_path+out_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in lane_change_buffer:
                writer.writerow(list(row))


def get_lane_change_list():
    lane_change_dict = dict()
    dir_path = root_path
    file_name = 'lane_change_event.csv'
    with open(dir_path+str(file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] not in lane_change_dict.keys():
                lane_change_dict[row[0]+'-'+row[1]] = [int(float(row[2]))]
            else:
                if all(abs((int(float(row[2]))) - e ) > 200 for e in lane_change_dict[row[0]+'-'+row[1]]):
                    lane_change_dict[row[0]+'-'+row[1]].append(int(float(row[2])))
    lane_change_list = []
    for key,val in lane_change_dict.items():
        for v in val:
            temp = key+'-'+str(v)
            lane_change_list.append(temp)
    return lane_change_list
            
        
            
def flush_cut_ins(cut_in_list):  
    dir_path = root_path
    out_file_name = 'cut_in_events.csv'
    with open(dir_path+out_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in cut_in_list:
                writer.writerow(list(row))
    return []             
            
            
    
def detect_cut_ins():
    lane_change_list = get_lane_change_list()
    dir_path = data_path+'DataFrontTargets/csv_files/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    cut_in_list = []
    c_id = 0
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100,c_id)
        data_buffer_dict = dict()
        line_num = 0
        with open(dir_path+str(file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if line_num > 0:
                    if row[0]+'-'+row[1]+'-'+row[2] not in data_buffer_dict.keys():
                        data_buffer_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
                    else:
                        data_buffer_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
                line_num = line_num + 1
        for key,val in data_buffer_dict.items():
            if key not in lane_change_list:
                rows_t = val
                rows_t_minus_1 = None
                device,trip,time = key.split('-')
                t_minus_1 = int(time) - 10
                if device+'-'+trip+'-'+str(t_minus_1) in data_buffer_dict.keys():
                    rows_t_minus_1 = data_buffer_dict[device+'-'+trip+'-'+str(t_minus_1)]
                    for r_t in rows_t:
                        for r_t_1 in rows_t_minus_1:
                            if r_t[4] == r_t_1[4] and r_t[10] is '1' and r_t_1[10] is '0':
                                c_id = c_id + 1
                                cut_in_list.append(r_t)
                                if len(cut_in_list) >= 5000:
                                    cut_in_list = flush_cut_ins(cut_in_list)
            
                                
    dir_path = root_path
    out_file_name = 'cut_in_events.csv'
    with open(dir_path+out_file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in cut_in_list:
                writer.writerow(list(row))
    return cut_in_list            
                             
def remove_pedestrians_from_all_cut_ins():
    dir_path = root_path
    in_file_name = 'cut_in_events.csv'
    out_file_name = 'vehicle_cut_in_events.csv'
    with open(dir_path+in_file_name, 'r', newline='') as inp, \
     open(dir_path+out_file_name, 'w', newline='') as outp:
        writer = csv.writer(outp, delimiter=',')
        for row in csv.reader(inp, delimiter=','):
            if row[8] != '3' or row[8] != '4':
                writer.writerow(row)
                
def flush_to_new_file(file_path,data_list):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in data_list:
                writer.writerow(list(row))
    

def create_data_wsu_for_cut_ins():
    dir_path = root_path
    file_name = 'cut_in_events.csv'
    cut_in_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in cut_in_dict:
                cut_in_dict[row[0]+'-'+row[1]].append(row[2])
            else:
                cut_in_dict[row[0]+'-'+row[1]] = [row[2]]
            
    
    dir_path = root_path+'wsu_data/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    wsu_cut_in_list = []
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100)
        if str(file)[:-4] in cut_in_dict.keys():
            data_buffer = []
            with open(dir_path+str(file)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    data_buffer.append(row)
            for entry in data_buffer:
                if entry[2] in cut_in_dict[str(file)[:-4]]:
                    wsu_cut_in_list.append(entry)
    flush_to_new_file(root_path+'wsu_cut_in_list.csv',wsu_cut_in_list)

def create_wsu_sequence_for_cutins():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    time_seq_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in time_seq_dict:
                time_seq_dict[row[0]+'-'+row[1]].append(int(row[2]))
            else:
                time_seq_dict[row[0]+'-'+row[1]] = [int(row[2])]
    ''' add 5 seconds of past and future to the time sequence dict '''
    for k,v in time_seq_dict.items():
        seq_with_added_time = list(v)
        for _time in v:
            for _next_time in np.arange(_time-500,_time+510,10):
                seq_with_added_time.append(_next_time)
        time_seq_dict[k] = sorted(seq_with_added_time)
    dir_path = root_path+'wsu_data/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100)
        if str(file)[:-4] in time_seq_dict.keys():
            wsu_cut_in_list = []
            data_buffer = []
            with open(dir_path+str(file)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    data_buffer.append(row)
            for entry in data_buffer:
                if int(entry[2]) in time_seq_dict[str(file)[:-4]]:
                    wsu_cut_in_list.append(entry)
            flush_to_new_file(root_path+'wsu_seq_data_for_cutins/'+str(file),wsu_cut_in_list)


def create_front_target_sequence_for_cutins():
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    time_seq_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in time_seq_dict:
                time_seq_dict[row[0]+'-'+row[1]].append(int(row[2]))
            else:
                time_seq_dict[row[0]+'-'+row[1]] = [int(row[2])]
    ''' add 5 seconds of past and future to the time sequence dict '''
    for k,v in time_seq_dict.items():
        seq_with_added_time = list(v)
        for _time in v:
            for _next_time in np.arange(_time-500,_time+510,10):
                seq_with_added_time.append(_next_time)
        time_seq_dict[k] = sorted(seq_with_added_time)
    dir_path = root_path+'front_target_data/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    
    for ct,file in enumerate(filename_list):
        print('processing file ',str(file),', complete:',(ct/file_count)*100)
        if str(file)[:-4] in time_seq_dict.keys():
            front_target_cut_in_list = []
            data_buffer = []
            with open(dir_path+str(file)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    data_buffer.append(row)
            for entry in data_buffer:
                if int(entry[2]) in time_seq_dict[str(file)[:-4]]:
                    front_target_cut_in_list.append(entry)
            flush_to_new_file(root_path+'front_target_seq_for_cutins/'+str(file),front_target_cut_in_list)


''' all runs below '''

                    
                

                        
    
                
        
                           
                    
                
                
                 
            
                
    
                  
                
                
