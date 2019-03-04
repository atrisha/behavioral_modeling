'''
Created on Oct 30, 2018

@author: atrisha
'''
import csv
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import operator
import json
import os.path
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
import pickle
import time
import datetime
from utils import *


def count_negative_range_rate():
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    positive_count,negative_count = 0,0
    count = 0
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if count == 0:
                print(row)
                print(row[6])
            if float(row[6]) < 0:
                negative_count = negative_count + 1
            else:
                positive_count = positive_count + 1
            count = count + 1
    print('Negative count:',negative_count)
    print('Positive count:',positive_count)
    
def plot_vel_at_cut_ins():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    gps_vel_list_mps,can_vel_list_kph = [],[]
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[9]) > 0.3:
                gps_vel_list_mps.append(float(row[9]))
            if float(row[16]) > 1:
                can_vel_list_kph.append(float(row[16]))
  
    plt.hist(can_vel_list_kph, bins=100,density=True, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
    plt.show()
    print(len(can_vel_list_kph))
  
    
def plot_ttc():
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    ttc_secs = []
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[6]) != 0:
                ttc = float(row[5]) / float(row[6])
                if -50 <= ttc <= 50:
                    ttc_secs.append(ttc * -1)
    #print(min(ttc_secs),max(ttc_secs))
    plt.hist(ttc_secs, bins=100,density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
    plt.show()
    
    
def store_hist_objects(key,val,store):
    file_name = root_path+'histogram_distr.json'
    if os.path.isfile(file_name):
        js = open(file_name,'r')
        hist_dict = json.load(js)
    else:
        hist_dict = dict()
    if store:
        hist_dict[key] = val
        with open(file_name,'w') as file:
            file.write(json.dumps(hist_dict))
    return hist_dict
    

def plot_ttc_vel():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    ttc_secs,vel_kph = [],[]
    
    vel_5_15,vel_15_25,vel_25_35 = [],[],[]
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0 :
                ttc = float(row[5]) / -float(row[6])
                
                if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                    vel_mps = int(round(float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16]) / 3.6))
                    if 5 <= vel_mps < 15:
                        vel_5_15.append((vel_mps,1/ttc))
                    elif 15 <= vel_mps < 25:
                        vel_15_25.append((vel_mps,1/ttc))
                    elif 25 <= vel_mps < 35:
                        vel_25_35.append((vel_mps,1/ttc))
    
    
    lambda_5_15 = np.mean([x[0] for x in vel_5_15])
    lambda_15_25 = np.mean([x[0] for x in vel_15_25])
    lambda_25_35 = np.mean([x[0] for x in vel_25_35])
    
    ttc_1 = [x[1] for x in vel_5_15]
    ttc_2 = [x[1] for x in vel_15_25]
    ttc_3 = [x[1] for x in vel_25_35]
    
    all_ttc_inv = ttc_1 + ttc_2 + ttc_3
    print('min,max ttc',min(all_ttc_inv),max(all_ttc_inv))
    
    #print(lambda_15_25)
    #print(lambda_25_35)
    
    def _exp(X,lambda_param,a):
        return [(a*1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    def _pareto(X,param):
        return [(param*math.pow(.1,param))/(math.pow(x, param+1)) if x>=.1 else 0 for x in X]
    
    '''
    hist = np.histogram(ttc_1,bins = np.arange(0,math.ceil(max(ttc_1))+.01,.01),density=True)
    X = [float(x/100) for x in np.arange(0,len(hist[0]))]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    #store_hist_objects('ttc_inv_5_15', dict_val, True)
    
    
    
    hist = np.histogram(ttc_2,bins = np.arange(0,math.ceil(max(ttc_2)),.01),density=True)
    X = [float(x/100) for x in np.arange(0,len(hist[0]))]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    #store_hist_objects('ttc_inv_15_25', dict_val, True)
    
    hist = np.histogram(ttc_3,bins = np.arange(0,math.ceil(max(ttc_3)),.01),density=True)
    X = [float(x/100) for x in np.arange(0,len(hist[0]))]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    #store_hist_objects('ttc_inv_25_35', dict_val, True)
    
    '''
    '''
    ttc_1 = [1/x[1] for x in vel_5_15]
    ttc_2 = [1/x[1] for x in vel_15_25]
    ttc_3 = [1/x[1] for x in vel_25_35]
    '''
    print(min(ttc_1),max(ttc_1))
    print(min(ttc_2),max(ttc_2))
    print(min(ttc_3),max(ttc_3))
    
    bins = np.arange(.1,math.ceil(max(ttc_1))+.01,.1)
    hist = np.histogram(ttc_1,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    
    
    
    bins = np.arange(.1,math.ceil(max(ttc_2))+.01,.1)
    hist = np.histogram(ttc_2,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    
    bins = np.arange(.1,math.ceil(max(ttc_3))+.01,.1)
    hist = np.histogram(ttc_3,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    
    
    plt.show()
    


    

def plot_range_vel():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    ttc_secs,vel_kph = [],[]
    vel_5_15,vel_15_25,vel_25_35 = [],[],[]
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            range_inv = 1/float(row[5])
            if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                vel_mps = int(round(float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16]) / 3.6))
                if 5 <= vel_mps < 15:
                    vel_5_15.append((vel_mps,range_inv))
                elif 15 <= vel_mps < 25:
                    vel_15_25.append((vel_mps,range_inv))
                elif 25 <= vel_mps < 35:
                    vel_25_35.append((vel_mps,range_inv))
    
    
    
    
    range_inv_1 = [x[1] for x in vel_5_15]
    range_inv_2 = [x[1] for x in vel_15_25]
    range_inv_3 = [x[1] for x in vel_25_35]
    
    all_range_inv = range_inv_1 + range_inv_2 + range_inv_3
    print(min(all_range_inv),max(all_range_inv))
    
    
    def _exp(X,lambda_param,a):
        return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    def _pareto(X,param):
        x_m = .01
        return [1 if x<x_m else math.pow((x_m/x),param) for x in X]
    
    
    
    hist = np.histogram(all_range_inv,bins = np.arange(0,math.ceil(max(all_range_inv)),.01),density=True)
    X = [x/100 for x in np.arange(0,len(hist[0]))]
    Y = [x/100 for x in hist[0]]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    '''
    
    hist = np.histogram(range_inv_1,bins = np.arange(0,math.ceil(max(range_inv_1)),.05),density=True)
    X = np.arange(0,math.ceil(max(range_inv_1)),.05)[1:]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    dict_val = list(zip(X,Y))
    print(popt)
    #store_hist_objects('range_inv_5_15', dict_val, True)
    
    
    hist = np.histogram(range_inv_2,bins = np.arange(0,math.ceil(max(range_inv_2)),.05),density=True)
    X = np.arange(0,math.ceil(max(range_inv_1)),.05)[1:]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    #store_hist_objects('range_inv_15_25', dict_val, True)
    
    hist = np.histogram(range_inv_3,bins = np.arange(0,math.ceil(max(range_inv_3)),.05),density=True)
    X = np.arange(0,math.ceil(max(range_inv_1)),.05)[1:]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_exp, X, Y)
    plt.plot(X, _exp(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    #store_hist_objects('range_inv_25_35', dict_val, True)
    '''
    '''
    import operator
    bins = np.arange(10,math.ceil(max(range_inv_1))+10,10)
    hist = np.histogram(range_inv_1,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    index,value = max(enumerate(Y),key = operator.itemgetter(1))
    vel_mean = print(index,value,X[index])
    
    
    bins = np.arange(10,math.ceil(max(range_inv_2))+10,10)
    hist = np.histogram(range_inv_2,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    index,value = max(enumerate(Y),key = operator.itemgetter(1))
    vel_mean = print(index,value,X[index])
    
    bins = np.arange(10,math.ceil(max(range_inv_3))+10,10)
    hist = np.histogram(range_inv_3,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    index,value = max(enumerate(Y),key = operator.itemgetter(1))
    vel_mean = print(index,value,X[index])
    '''
    plt.show()
    
def calc_util(x,curve_type):
    if x >= 10:
        return 1
    elif x<= -10:
        return 0
    else:
        if curve_type is 'sigmoidal':
            return 1/(1+math.exp(-1 * (x-4)))
        else:
            ''' linear '''
            return (0.2*x) - 1
    
def plot_utilities():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    ttc_secs,vel_kph = [],[]
    low_speed_u,med_speed_u,high_speed_u = [],[],[]
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0:
                ttc = float(row[5]) / -float(row[6])
                if 0 <= ttc <= 10:
                    if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                        vel = float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16])
                        if 0 < vel < 30:
                            low_speed_u.append(calc_util(ttc))
                        elif 30 <= vel < 80:
                            med_speed_u.append(calc_util(ttc))
                        elif vel >= 80:
                            high_speed_u.append(calc_util(ttc))
    
    plt.subplot(3, 1, 1)
    plt.hist(low_speed_u, bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,.07,.08,.09,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
    plt.xlabel('Low speed (0-30 kph)')
    plt.ylabel('count')
    
    plt.subplot(3, 1, 2)
    plt.hist(med_speed_u, bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,.07,.08,.09,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
    plt.xlabel('Medium speed (30-80 kph)')
    plt.ylabel('count')
    
    plt.subplot(3, 1, 3)
    plt.hist(high_speed_u, bins=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,.07,.08,.09,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
    plt.xlabel('High speed (>80 kph)')
    plt.ylabel('count')
    
    plt.show()
    
def get_ttc_list():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    ttc_secs,vel_kph = [],[]
    low_speed_ttc,med_speed_ttc,high_speed_ttc = [],[],[]
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0 :
                ttc = float(row[5]) / -float(row[6])
                if 0 <= ttc:
                    if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                        vel = float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16])
                        if 0 < vel < 30:
                            low_speed_ttc.append(ttc)
                        elif 30 <= vel < 80:
                            med_speed_ttc.append(ttc)
                        elif vel >= 80:
                            high_speed_ttc.append(ttc)
    ttc_dict = {'low_speed':low_speed_ttc,'med_speed':med_speed_ttc,'high_speed':high_speed_ttc}
    return ttc_dict

def reject_outliers(data, m=2):
    filtered_list = []
    med = np.median(data)
    std = np.std(data)
    for d in data:
        if abs(d-med) < m * std:
            filtered_list.append(d)
        else:
            if len(filtered_list) > 0:
                filtered_list.append(filtered_list[-1])
    return filtered_list  
    
def get_ttc_list_with_time():
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    wsu_seq_dict = dict()
    
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    ttc_secs,vel_kph = [],[]
    line_count = 0
    max_diff_l = []
    low_speed_ttc,med_speed_ttc,high_speed_ttc = [],[],[]
    file_cache = dict()
    inst_id = 0
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0 :
                inst_id = inst_id + 1
                seq_entries = []
                dir_path_wsu = root_path+'wsu_seq_data_for_cutins/'
                file_name_wsu = row[0]+'-'+row[1]+'.csv'
                if os.path.isfile(dir_path_wsu+file_name_wsu):
                    with open(dir_path_wsu+file_name_wsu, 'r', newline='') as csv_file_wsu:
                        csv_reader_wsu = csv.reader(csv_file_wsu, delimiter=',')
                        for row_wsu in csv_reader_wsu:
                            wsu_seq_dict[row_wsu[0]+'-'+row_wsu[1]+'-'+row_wsu[2]] = row_wsu
                print('processed',round((line_count / 74450)*100,3),'%')
                range_x,range_rate = float(row[5]) , -float(row[6])
                ttc = float(row[5]) / -float(row[6])
                obstacle_id = row[4]
                ''' code to get next 5 secs sequence '''
                next_five_sec_timestamp = np.arange(int(row[2])+10,int(row[2])+510,10)
                seq_file_name = root_path+'front_target_seq_for_cutins/'+row[0]+'-'+row[1]+'.csv'
                seq_list = []
                if row[0]+'-'+row[1] in file_cache.keys():
                    for seq_row in file_cache[row[0]+'-'+row[1]]:
                        if int(seq_row[2]) in next_five_sec_timestamp and seq_row[4] == obstacle_id:
                            seq_list.append(seq_row)
                else:
                    file_cache[row[0]+'-'+row[1]] = []         
                    with open(seq_file_name, 'r', newline='') as seq_csv_file:
                        seq_csv_reader = csv.reader(seq_csv_file, delimiter=',')
                        for seq_row in seq_csv_reader:
                            if int(seq_row[2]) in next_five_sec_timestamp and seq_row[4] == obstacle_id:
                                seq_list.append(seq_row)
                            file_cache[row[0]+'-'+row[1]].append(seq_row)
                ''' code to get next 5 sec sequence is in seq_list'''
                next_five_sec_ttc,next_five_sec_v_s = [],[]
                next_five_sec_range,next_five_sec_range_rate = [],[]
                next_five_sec_v_lc = []
                t_id = 1
                frame_id = []
                for f_seq in seq_list:
                    if float(f_seq[6]) >= 0:
                        next_five_sec_ttc.append(30)
                    else:
                        next_five_sec_ttc.append(float(f_seq[5]) / -float(f_seq[6]))            
                    if f_seq[0]+'-'+f_seq[1]+'-'+f_seq[2] in wsu_seq_dict.keys():
                        vel = float(wsu_seq_dict[f_seq[0]+'-'+f_seq[1]+'-'+f_seq[2]][16]) / 3.6
                        vel_lc = float(f_seq[6]) + vel
                        next_five_sec_v_s.append(vel)
                        next_five_sec_range.append(float(f_seq[5]))
                        next_five_sec_range_rate.append(float(f_seq[6]))
                        next_five_sec_v_lc.append(vel_lc)
                        frame_id.append(t_id)
                    t_id = t_id + 1
                ttc_entry = [ttc] + next_five_sec_ttc
                #ttc_entry_s = reject_outliers(ttc_entry, m=2)
                #plt.plot(np.arange(len(ttc_entry)),ttc_entry)
                #ttc_entry_s = reject_outliers(ttc_entry, m=2)
                '''if len(ttc_entry_s) > 5:
                    z = np.polyfit(np.arange(len(ttc_entry_s)),ttc_entry_s,4)
                    p = np.poly1d(z)
                    #plt.plot(np.arange(len(ttc_entry_s)),ttc_entry_s)
                    x_fit = [p(x) for x in np.arange(len(ttc_entry_s))]
                    ttc_entry = x_fit'''
                #print(max_diff)
                #plt.plot(np.arange(len(ttc_entry_s)),x_fit,'x')
                #plt.show()
                vel = None
                if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                    vel = float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16]) 
                    if 0 < vel < 30:
                        low_speed_ttc.append(ttc_entry)
                    elif 30 <= vel < 80:
                        med_speed_ttc.append(ttc_entry)
                    elif vel >= 80:
                        high_speed_ttc.append(ttc_entry)
                if vel is not None:
                    vel_lc = (vel / 3.6) + (-1 *range_rate)
                    vel_mps = vel / 3.6
                else:
                    vel_lc = None
                    vel_mps = None
                   
                with open(root_path+'interac_seq_data.csv', 'a', newline='') as csvfile_int_seq:
                    writer_seq = csv.writer(csvfile_int_seq, delimiter=',')
                    writer_seq.writerow([inst_id,0,vel_mps,vel_lc,range_x,range_rate])
                    for row in list(zip(frame_id,next_five_sec_v_s,next_five_sec_v_lc,next_five_sec_range,next_five_sec_range_rate)):
                        entry_seq = [inst_id] + list(row)
                        writer_seq.writerow(entry_seq)
                
            line_count = line_count + 1
            '''if line_count % 1000 == 0:
                bins = [0,10,50,100]
                plt.hist(max_diff_l,bins=bins)
                plt.show()'''
    ttc_dict = {'low_speed':low_speed_ttc,'med_speed':med_speed_ttc,'high_speed':high_speed_ttc}
    '''with open(root_path+'all_cutin_ttc.json','w') as file:
        file.write(json.dumps(ttc_dict))'''
    return ttc_dict



def generate_ttc_dist_list(ttc_time_list,with_time):
    if not with_time:
        ttc_dist = dict()
        intervals = np.arange(0,10.1,0.1)
        for i in intervals:
            ttc_dist[round(i,1)] = 0
        for ttc in ttc_time_list:
            if ttc > 10:
                rounded_ttc = 10.0
            else:
                rounded_ttc = round(ttc,1)
            ttc_dist[rounded_ttc] = (ttc_dist[rounded_ttc] + 1)
        for k,v in ttc_dist.items():
            ttc_dist[k] = v / len(ttc_time_list)
        ttc_dist_list = [(k,v) for k,v in ttc_dist.items()]
        return ttc_dist_list
    else:
        ttc_dist = dict()
        max_time_length = 50
        for _time in np.arange(max_time_length+1):
            ttc_dist[_time] = dict()
            intervals = np.arange(0,10.1,0.1)
            for i in intervals:
                ttc_dist[_time][round(i,1)] = 0
        horizon_count = dict()
        for ttc_list in ttc_time_list:
            for indx,_ttc_time in enumerate(ttc_list):
                if _ttc_time > 10:
                    rounded_ttc = 10.0
                else:
                    rounded_ttc = round(_ttc_time,1)
                ttc_dist[indx][rounded_ttc] = ttc_dist[indx][rounded_ttc] + 1
                if indx in horizon_count.keys():
                    horizon_count[indx] = horizon_count[indx] + 1
                else:
                    horizon_count[indx] = 1
        for _t_k,_t_v in ttc_dist.items():
            for _ttc_k,_ttc_v in _t_v.items():
                ttc_dist[_t_k][_ttc_k] = ttc_dist[_t_k][_ttc_k] / horizon_count[_t_k]
        return ttc_dist
                
                
                
                
                
            
def find_min_range_value_in_cutins():
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    range_list = []
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            range_list.append(float(row[5]))
    print(min(range_list),max(range_list))
    

            
                  
            
            
def find_min_range_value_in_all():
    dir_path = root_path+'front_target_data/' 
    filename_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    file_count = len(filename_list)
    filename_list.sort()
    range_list = []
    stop_logging = False
    for ct,file in enumerate(filename_list):
        range_list_current_file = []
        with open(dir_path+str(file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if int(row[8]) < 3:
                    range_list_current_file.append(float(row[5]))
                    '''if float(row[5]) < 0.001:
                        print(file)
                        print(row)
                        stop_logging = True
                        break'''
        if len(range_list_current_file) > 0:
            min_range = min(range_list_current_file)
            range_list.append(min_range)
        print('complete:',(ct/file_count)*100,'min range_x',min(range_list))
    print(min(range_list))
            
            
        


def round_off_ttcs(ttc_list_wrt_time):
    for _t_list_a in ttc_list_wrt_time:
        for ttc in _t_list_a:
            if ttc > 10:
                ttc = 10.0
            else:
                ttc = round(ttc,1)
    return ttc_list_wrt_time
            
    
            
def add_util_values(ttc_dist,curve_type,is_dict):
    if not is_dict:
        for ind,ttc in enumerate(ttc_dist):
            ttc_dist[ind] = (ttc[0],ttc[1],calc_util(ttc[0],curve_type))
        return ttc_dist
    else:
        for k,v in ttc_dist.items():
            for k_i,v_i in v.items():
                ttc_dist[k][k_i] = (v_i,calc_util(k_i,curve_type))
        return ttc_dist
                
                
        
    

def get_max_likelihood_estimate(p_o,u_opt,u_sub_opt):
    if p_o == 0:
        ''' this means lambda is -infinity'''
        return -30000
    else:
        return math.log((1/p_o)-1) / (u_sub_opt - u_opt)
    

def create_ttc_dist_with_time():
    if os.path.isfile(root_path+'all_cutin_ttc.json'):
        js = open(root_path+'all_cutin_ttc.json')
        ttc_dict = json.load(js)
    else:    
        ttc_dict = get_ttc_list_with_time()
    lambda_dict = dict()
    ttc_dist_dict = dict()
    for speed_level,ttc_list_wrt_time in ttc_dict.items():
        ttc_dist_dict[speed_level] = generate_ttc_dist_list(ttc_list_wrt_time,True)
        ttc_dist_dict[speed_level] = add_util_values(ttc_dist_dict[speed_level],'sigmoidal',True)
    for speed_level,time_indexed_ttc in ttc_dist_dict.items(): 
        lambda_dict[speed_level] = dict()  
        for time_index,ttc_det in time_indexed_ttc.items():
            print('processing',speed_level,time_index)
            flat_list = [(k,v[0],v[1]) for k,v in ttc_det.items()]
            flat_list.sort(key=operator.itemgetter(2),reverse=True)
            for level in np.arange(1,len(ttc_det.keys())+1):
                if len(flat_list) > 1:
                    p_o = flat_list[0][1]
                    u_opt = flat_list[0][2]
                    u_sub_opt = flat_list[1][2]
                    lambda_val = get_max_likelihood_estimate(p_o,u_opt,u_sub_opt)
                    if time_index in lambda_dict[speed_level].keys():
                        lambda_dict[speed_level][time_index][level] = lambda_val
                    else:
                        lambda_dict[speed_level][time_index] = dict()
                        lambda_dict[speed_level][time_index][level] = lambda_val
                    flat_list = flat_list[1:]
                else:
                    break
    with open(root_path+'lambda_dist.json','w') as file:
        file.write(json.dumps(lambda_dict))
    return ttc_dict
    
    
def create_dist():
    ttc_dict = get_ttc_list()
    ttc_dist = dict()
    lambda_dict = dict()
    for speed_level,ttc_list in ttc_dict.items():
        ttc_dist_list = generate_ttc_dist_list(ttc_list,False)
        ttc_dist_list = add_util_values(ttc_dist_list,'sigmoidal',False)
        max_level = len(ttc_dist_list)+1
        for level in np.arange(1,max_level):
            ttc_dist_list.sort(key=operator.itemgetter(2),reverse=True)
            if len(ttc_dist_list) > 1:
                p_o = ttc_dist_list[0][1]
                u_opt = ttc_dist_list[0][2]
                u_sub_opt = ttc_dist_list[1][2]
                lambda_val = get_max_likelihood_estimate(p_o,u_opt,u_sub_opt)
                lambda_dict[level] = lambda_val
                ttc_dist_list = ttc_dist_list[1:]
            else:
                break
            
        plt.plot(lambda_dict.keys(),lambda_dict.values(),'.')
        plt.show()
        
def plot_range_ttc():
    data_dict = {'0-15':[],'15-30':[],'30-45':[],'45-60':[],'60-75':[]}
    with open(root_path+'vehicle_cut_in_events.csv','r') as csvfile:
        csv_reader = csv.reader(csvfile,delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0:
                range_x = float(row[5])
                ttc = range_x / -float(row[6])
                for k,v in data_dict.items():
                    low,high = k.split('-')
                    if int(low) <= range_x < int(high):
                        data_dict[k].append(int(round(ttc)))
                        break
    for k,v in data_dict.items():
        hist = np.histogram(v,bins=np.arange(350),density=True)
        plt.plot(np.arange(349),hist[0])
    plt.show()
    
    
    
        

def smooth_velocity_curves():
    with open(root_path+'interac_seq_data.csv','r') as csvfile:
        c_0,c_1,V_s,V_lc,c_4,c_5 = [],[],[],[],[],[]
        csv_reader = csv.reader(csvfile,delimiter=',')
        count = 0
        for row in csv_reader:
            print('processing: '+str(count/1755166 * 100)+'%')
            if int(row[1]) == 0:
                if len(V_s) == 0 and len(V_lc) == 0:
                    if row[2] != '':
                        V_s.append(float(row[2]))
                    if row[3] != '':
                        V_lc.append(float(row[3]))
                    
                    c_0.append(row[0])
                    c_1.append(row[1])
                    c_4.append(row[4])
                    c_5.append(row[5])
                else:
                    if len(V_lc) > 10 and len(V_s) > 10:
                        
                        z = np.polyfit(np.arange(len(V_s)),V_s,4)
                        p = np.poly1d(z)
                        x_fit = [p(x) for x in np.arange(len(V_s))]
                        V_s = x_fit
                        
                        z_1 = np.polyfit(np.arange(len(V_lc)),V_lc,4)
                        p_1 = np.poly1d(z_1)
                        x_fit_1 = [p_1(x) for x in np.arange(len(V_lc))]
                        V_lc = x_fit_1
                        
                    
                    '''plt.plot(np.arange(len(V_s)),V_s)
                    plt.plot(np.arange(len(V_lc)),V_lc)
                    plt.show()'''
                    
                    with open(root_path+'interac_seq_data_smooth.csv', 'a', newline='') as csvfile_int_seq:
                        writer_seq = csv.writer(csvfile_int_seq, delimiter=',')
                        for row in list(zip(c_0,c_1,V_s,V_lc,c_4,c_5)):
                            writer_seq.writerow(row)
                            #print(row)
                    '''
                    for row in list(zip(c_0,c_1,V_s,V_lc,c_4,c_5)):
                        print(row)
                    '''
                    c_0,c_1,V_s,V_lc,c_4,c_5 = [],[],[],[],[],[]
                    if row[2] != '':
                        V_s.append(float(row[2]))
                    if row[3] != '':
                        V_lc.append(float(row[3]))
                    c_0.append(row[0])
                    c_1.append(row[1])
                    c_4.append(row[4])
                    c_5.append(row[5])
                    
            else:
                if row[2] != '':
                        V_s.append(float(row[2]))
                else:
                    V_s.append(row[2])
                if row[3] != '':
                    V_lc.append(float(row[3]))
                else:
                    V_lc.append(row[3])
                c_0.append(row[0])
                c_1.append(row[1])
                c_4.append(row[4])
                c_5.append(row[5])
            count = count + 1
            
                
       
            
            
    
def plot_lc_veh_vel():
    lc_veh_vel_mps = []
    dir_path = root_path
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = root_path
    file_name = 'vehicle_cut_in_events.csv'
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0 :
                range_rate = -float(row[6])
                if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                    vel_s = int(round(float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16]) / 3.6))
                    vel_lc = vel_s - range_rate
                    lc_veh_vel_mps.append(vel_lc)
    
    
    
    def _gauss(x,mu,sigma,A):
        #A = 1 / (2*math.pi*sigma**2)**0.5
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
        return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]
    
    
    max_vel = max(lc_veh_vel_mps)
    bins = np.arange(.5,max_vel+.5,0.5)
    hist = np.histogram(lc_veh_vel_mps,bins = bins,density=True)
    X = bins[:-1]
    Y = hist[0]
    dict_val = list(zip(X.astype(float),Y.astype(float)))
    #store_hist_objects('veh_dist', dict_val, True)
    #params,cov=curve_fit(_bimodal,X,Y)
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    #print(params)
    plt.plot(X,_bimodal(X,*vel_bimodal_params))
    axes = plt.gca()
    axes.set_xlim([0,max_vel])
    axes.set_ylim([0,.1])
    plt.plot(X,Y)
    plt.show()        
    
    
'''
def util_dist(dist_m,thresh):
    x = dist_m
    return 1.5/(1+math.exp(-1 * (x-thresh))) - 0.5
    
    
def util_ttc(ttc_sec,thresh):
    if ttc_sec > 100:
        return 1
    x = ttc_sec
    return 1.5/(1+math.exp(-1 * (x-thresh))) - 0.5
 
def util_progress(vel_mps,thresh=0):
    x = vel_mps
    return np.tanh(1*(x-2.5))
'''
'''
def util_dist(x,scale=.4,thresh=10):
    return np.tanh(scale*(x-thresh)) 
    
def util_ttc(x,scale=2,thresh=.1):
    return np.tanh(scale*(x-thresh))
'''
def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
    return is_efficient
   
def get_action_probability(state_info):
    ttc_inv,vel_lc,range_inv = 1/state_info[0],state_info[1],1/state_info[2]
    
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    def _gauss(x,mu,sigma,A):
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return _gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2)
    
    range_inv_pareto_params = 2.25843986
    def _pareto(x,param):
        return (param*math.pow(.1,param))/(math.pow(x, param+1)) if x>=.1 else 0
    
    ttc_inv_5_15_pareto_params = 0.00586138
    ttc_inv_15_25_pareto_params = 0.00622818
    ttc_inv_25_35_pareto_params = 0.00707637
    def _exp(X,lambda_param,a):
        return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    prob_vel_lc = _bimodal(vel_lc,*vel_bimodal_params)
    ttc_dist_param = ttc_inv_5_15_pareto_params if 0<=vel_lc<15 else ttc_inv_15_25_pareto_params if 15<=vel_lc<25 else ttc_inv_25_35_pareto_params
    prob_ttc_inv = _pareto(ttc_inv,ttc_dist_param)
    prob_range_inv = _pareto(range_inv,range_inv_pareto_params)
    
    '''
    hist_dict = store_hist_objects(None,None,False)
    vel_hist = hist_dict['veh_dist']
    ttc_inv_5_15_hist = hist_dict['ttc_inv_5_15']
    ttc_inv_15_25_hist = hist_dict['ttc_inv_15_25']
    ttc_inv_25_35_hist = hist_dict['ttc_inv_25_35']
    range_inv_5_15_hist = hist_dict['range_inv_5_15']
    range_inv_15_25_hist = hist_dict['range_inv_15_25']
    range_inv_25_35_hist = hist_dict['range_inv_25_35']
    
    def _get_prob(hist_list_tuples,key):
        if key < hist_list_tuples[0][0]:
            return hist_list_tuples[0][1]
        elif key > hist_list_tuples[-1][0]:
            return hist_list_tuples[-1][1]
        else:
            keys = [x[0] for x in hist_list_tuples]
            for indx,item in enumerate(list(zip(keys[:-1],keys[1:]))):
                if item[0] <= key <= item[1]:
                    return hist_list_tuples[indx][1]
    
    prob_vel_lc = _get_prob(vel_hist, vel_lc)
    
    ttc_inv_hist = ttc_inv_5_15_hist if 0<=vel_lc<15 else ttc_inv_15_25_hist if 15<=vel_lc<25 else ttc_inv_25_35_hist
    prob_ttc_inv = _get_prob(ttc_inv_hist, ttc_inv)
    
    range_inv_hist = range_inv_5_15_hist if 0<=vel_lc<15 else range_inv_15_25_hist if 15<=vel_lc<25 else range_inv_25_35_hist
    prob_range_inv = _get_prob(range_inv_hist, range_inv)
    '''
    #print('probabilities:',prob_vel_lc,prob_ttc_inv,prob_range_inv)
    return prob_vel_lc*prob_ttc_inv*prob_range_inv


def solve_lambda(delta_u,p_a,integral_dict):
    low,high = -10000,10000
    lambda_list = np.arange(low,high)
    min = 1
    lambda_estimate = None
    level_count = 0
    while True:
        for l in lambda_list:
            integral_delta_u_a = integral_dict[l]
            p_a_calculated = np.exp(l*delta_u) / integral_delta_u_a
            diff = abs(p_a_calculated - p_a)
            if diff < min:
                min = diff
                lambda_estimate = l
        if lambda_estimate == low:
            high = low
            low = high - 10000
        elif lambda_estimate == high:
            low = high
            high = low + 10000
        else:
            break
        lambda_list = np.arange(low,high)
        level_count = level_count + 1
        if level_count >= 1:
            break
    return lambda_estimate
            
        

def generate_lambda_values():
    X,Y,Z = [],[],[]
    X_pareto,Y_pareto,Z_pareto = [],[],[]
    count = 0
    state_data = []
    continuos = True
    if not os.path.isfile(root_path+'cache_data/state_data.dmp'):
        dir_path = root_path
        file_name = 'wsu_cut_in_list.csv'
        wsu_dict = dict()
        with open(dir_path+file_name, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0]+'-'+row[1] in wsu_dict.keys():
                    wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
                else:
                    wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
        dir_path = root_path
        file_name = 'vehicle_cut_in_events.csv'
        ttc_secs,vel_kph = [],[]
        
        vel_5_15,vel_15_25,vel_25_35 = [],[],[]
        with open(dir_path+file_name, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                print('processed',count)
                if float(row[6]) < 0 :
                    range_x = float(row[5])
                    vel_s = None
                    if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                        vel_s = int(round(float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16]) / 3.6))
                    else:
                        continue
                    range_rate = -float(row[6])
                    vel_lc = vel_s - range_rate
                    if vel_lc != vel_s:
                        util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
                        X.append(util_progress(vel_lc))
                        Y.append(util_ttc(range_x/(vel_s-vel_lc),util_ttc_param))
                        util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
                        Z.append(util_dist(range_x,util_dist_param))
                        state_data.append((vel_s,vel_lc,range_x))
                count = count + 1
        '''            
        with open(root_path+'interac_seq_data.csv','r') as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=',')
            for row in csv_reader:
                print('processed',count/1755165)
                if int(row[1]) == 0 and row[3] != '':
                    vel_lc = float(row[3])
                    vel_s = float(row[2])
                    range_x = float(row[4])
                    if vel_lc != vel_s:
                        util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
                        X.append(util_progress(vel_lc))
                        Y.append(util_ttc(range_x/(vel_s-vel_lc),util_ttc_param))
                        util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
                        Z.append(util_dist(range_x,util_dist_param))
                        state_data.append((vel_s,vel_lc,range_x))
                count = count + 1
        '''
            
        
        pareto_array = np.asarray(list(zip(X,Y,Z)))
        pareto_array.reshape((len(X),3))
        print('calculating pareto front...')
        pareto_points = is_pareto_efficient(pareto_array)
        
                
        file_pi = open(root_path+'cache_data/state_data.dmp','wb')
        pickle.dump(state_data,file_pi)
        file_pi = open(root_path+'cache_data/X.dmp','wb')
        pickle.dump(X,file_pi)
        file_pi = open(root_path+'cache_data/Y.dmp','wb')
        pickle.dump(Y,file_pi)
        file_pi = open(root_path+'cache_data/Z.dmp','wb')
        pickle.dump(Z,file_pi)
        file_pi = open(root_path+'cache_data/pareto_points.dmp','wb')
        pickle.dump(pareto_points,file_pi)
                
    else:
        file_pi = open(root_path+'cache_data/state_data.dmp','rb')
        state_data = pickle.load(file_pi)
        file_pi = open(root_path+'cache_data/X.dmp','rb')
        X = pickle.load(file_pi)
        file_pi = open(root_path+'cache_data/Y.dmp','rb')
        Y = pickle.load(file_pi)
        file_pi = open(root_path+'cache_data/Z.dmp','rb')
        Z = pickle.load(file_pi)
        file_pi = open(root_path+'cache_data/pareto_points.dmp','rb')
        pareto_points = pickle.load(file_pi)
    '''        
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('utilvel_lc')
    ax.set_ylabel('util_ttc')
    ax.set_zlabel('util_range')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.scatter3D(X,Y,Z)
    #plt.show()
    print(total_count,len(X_pareto))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('utilvel_lc')
    ax.set_ylabel('util_ttc')
    ax.set_zlabel('util_range')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.scatter3D(X_pareto,Y_pareto,Z_pareto)
    plt.show()
    '''
    total_count = len(X)
    X_pareto,Y_pareto,Z_pareto = [],[],[]
    for i in enumerate(X):
        print('processed',i[0]/total_count)
        if pareto_points[i[0],]:
            X_pareto.append(X[i[0]])
            Y_pareto.append(Y[i[0]])
            Z_pareto.append(Z[i[0]])
    pareto_front = list(zip(X_pareto,Y_pareto,Z_pareto))
    data_points = list(zip(X,Y,Z))
    prob__delta_util_list = []
    print('setting up integral dict')
    integral_dict = dict()
    for l in np.arange(-10000,10000):
        print(l)
        integral_dict[l] = sum([np.exp(l*x) for x in np.arange(0,1.74,.01)])
    start_time = time.time()
    for idx,data in enumerate(data_points):
        dist_to_all_paretos = [np.linalg.norm([data[0]-x[0],data[1]-x[1],data[2]-x[2]]) for x in pareto_front]
        min_delta_to_pareto = min(dist_to_all_paretos)
        vel_s,vel_lc,range_x = state_data[idx]
        ttc = range_x/(vel_s-vel_lc)
        #print('state data:',idx,state_data[idx],ttc)
        prob_action = get_action_probability((ttc,vel_lc,range_x))
        #print(prob_action,min_delta_to_pareto)
        lambda_est = solve_lambda(min_delta_to_pareto,prob_action,integral_dict)
        prob__delta_util_list.append((prob_action,min_delta_to_pareto,lambda_est))
        if idx > 0:
            time_diff = ((time.time() - start_time) / idx ) * (total_count - idx)
        else:
            time_diff = 0
            
        print(str(datetime.timedelta(seconds=time_diff)),idx,round(idx/total_count,4),lambda_est)
        #print(vel_s,vel_lc,range_x,ttc,prob_action)
    file_pi = open(root_path+'cache_data/lambda_vals.dmp','wb')
    pickle.dump(prob__delta_util_list,file_pi)
        
    
    
def plot_lambda():
    file_pi = open(root_path+'cache_data/lambda_vals.dmp','rb')
    prob__delta_util_list = pickle.load(file_pi)   
    lambdas = [x[0] for x in prob__delta_util_list]
    min_l,max_l = min(lambdas),max(lambdas)
    bins = 10
    hist = np.histogram(lambdas,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(hist[1][1:],hist[1][:-1])]
    sum_y = sum(hist[0])
    Y = [x/sum_y for x in hist[0]]
    plt.plot(X,Y)
    plt.show()
    
    
        
        
        
def plot_u_prime_distr():
    file_pi = open(root_path+'cache_data/state_data.dmp','rb')
    state_data = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/X.dmp','rb')
    X = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/Y.dmp','rb')
    Y = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/Z.dmp','rb')
    Z = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/pareto_points.dmp','rb')
    pareto_points = pickle.load(file_pi)
    vel_s = 20
    count = 0
    X_sim,Y_sim,Z_sim =[],[],[]
    for vel_lc in np.arange(0,20,.1):
        for range_x in np.arange(0,100,.1):
            print('processed',count/200000)
            if vel_s != vel_lc:
                util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5 
                X_sim.append(util_progress(vel_lc))
                Y_sim.append(util_ttc(range_x/(vel_s-vel_lc),util_ttc_param))
                util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
                Z_sim.append(util_dist(range_x,util_dist_param))
            count = count + 1
    total_count = len(X_sim)
    X_pareto,Y_pareto,Z_pareto = [],[],[]
    for i in enumerate(X):
        if pareto_points[i[0],]:
            X_pareto.append(X[i[0]])
            Y_pareto.append(Y[i[0]])
            Z_pareto.append(Z[i[0]])
    pareto_front = list(zip(X_pareto,Y_pareto,Z_pareto))
    data_points = list(zip(X_sim,Y_sim,Z_sim))
    U_prime = []
    
    for idx,data in enumerate(data_points):
        print('processed',idx/total_count)
        dist_to_all_paretos = [np.linalg.norm([data[0]-x[0],data[1]-x[1],data[2]-x[2]]) for x in pareto_front]
        min_delta_to_pareto = min(dist_to_all_paretos)
        U_prime.append(min_delta_to_pareto)
        
    bins = np.arange(0,1.74,.01)
    hist = np.histogram(U_prime,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    plt.plot(X,Y)
    plt.show()
    
def plot_pareto_front():
    file_pi = open(root_path+'cache_data/state_data.dmp','rb')
    state_data = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/X.dmp','rb')
    X = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/Y.dmp','rb')
    Y = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/Z.dmp','rb')
    Z = pickle.load(file_pi)
    file_pi = open(root_path+'cache_data/pareto_points.dmp','rb')
    pareto_points = pickle.load(file_pi)
    
    total_count = len(X)
    X_pareto,Y_pareto,Z_pareto = [],[],[]
    for i in enumerate(X):
        print('processed',i[0]/total_count)
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
    plt.show()
    print(total_count,len(X_pareto))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('utilvel_lc')
    ax.set_ylabel('util_ttc')
    ax.set_zlabel('util_range')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.scatter3D(X_pareto,Y_pareto,Z_pareto,color='black')
    plt.show()
    return list(zip(X_pareto,Y_pareto,Z_pareto))
    
            
def get_optimal_action(vel_s):
    util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5 
    util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
    for vel_lc in np.arange(30,0,-.5):
        if vel_lc >= vel_s:
            util_d = round(util_dist(util_dist_param+10, util_dist_param),4)
            util_t = round(util_ttc(100, util_ttc_param),4)
            util_p = round(util_progress(vel_lc),4)
            return (util_p,util_t,util_d)
        else:
            for range_x in np.arange(util_dist_param,util_dist_param+200,.5):
                ttc = range_x/(vel_s - vel_lc)
                if ttc > util_ttc_param*2:
                    util_d = round(util_dist(range_x+10, util_dist_param),4)
                    util_t = round(util_ttc(ttc, util_ttc_param),4)
                    util_p = round(util_progress(vel_lc),4)
                    return (util_p,util_t,util_d)
    for vel_lc in np.arange(30,60,.5):
        if vel_lc >= vel_s:
            util_d = round(util_dist(util_dist_param+10, util_dist_param),4)
            util_t = round(util_ttc(100, util_ttc_param),4)
            util_p = round(util_progress(vel_lc),4)
            return (util_p,util_t,util_d)
        else:
            for range_x in np.arange(util_dist_param,util_dist_param+200,.5):
                ttc = range_x/(vel_s - vel_lc)
                if ttc > util_ttc_param*2:
                    util_d = round(util_dist(range_x+10, util_dist_param),4)
                    util_t = round(util_ttc(ttc, util_ttc_param),4)
                    util_p = round(util_progress(vel_lc),4)
                    return (util_p,util_t,util_d)
    return None 
    
      
                
def process_results():
    import ast
    import matplotlib.cm as cm
    cmc = False
    X_cmc,Y_cmc,X_lam,Y_lam = [],[],[],[]
    if cmc:
        wfile = open(root_path+'cross_entr_res.json', 'r')  
        dict_in_file = json.load(wfile)
        print(len(dict_in_file))    
    else:
        file_name = root_path+'lambda_opt_res.json'
        js = open(file_name,'r')
        dict_in_file = json.load(js)
        speed_s_list,speed_lc_list,range_list,react_time_list = [],[],[],[]
        ttc_list,reac_lambda_list = [],[]
        max_crashes = 0
        l_dict = dict()
        for k,v in dict_in_file.items():
            _k = ast.literal_eval(k)
            _k = str(_k[:-1])
            if _k in l_dict:
                l_dict[_k] = max(l_dict[_k],len(v))
            else:
                l_dict[_k] = len(v)
            for val in v:
                speed_s_list.append(float(val[0])*3.6)
                speed_lc_list.append(float(val[1])*3.6)
                range_list.append(val[2])
                ttc_list.append(float(val[2])/(float(val[0])-float(val[1])))
                react_time_list.append(float(val[3]))
                reac_lambda_list.append(ast.literal_eval(k)[3])
        print(min(range_list),max(range_list),np.mean(range_list))
        plt.scatter(speed_s_list,speed_lc_list,c=range_list,cmap='Greens',vmin=.5, vmax=10)
        plt.show()
        opt_l = None
        sorted_l = sorted(l_dict.items(), key=operator.itemgetter(1),reverse=True)
        print(sorted_l)
        '''
        plt.hist(speed_s_list, bins=100,density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
        plt.show()
        plt.hist(speed_lc_list, bins=100,density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
        plt.show()
        plt.hist(range_list, bins=100,density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
        plt.show()
        plt.hist(ttc_list, bins=100,density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
        plt.show()   
        plt.hist(react_time_list, bins=100,density=False, alpha = 0.5, histtype='stepfilled', color='steelblue', edgecolor = 'none')
        plt.show()     
        '''
            
        
        
def show_matrix():
    res_dict = {0: {'id': 'c_1', 'lambdas_tried': [[-20, -20, -20, -1.8], [-71, -41, -95, -2.6], [-49, -60, -28, -3.0], [-91, -58, -2, -3.2], [-50, -42, -27, -0.7], [-16, -33, -25, -3.2], [-89, -83, -71, -1.1], [-41, -43, -64, -1.7], [-74, -31, -92, -2.2], [-64, -77, -72, -2.1], [-42, -49, -35, -3.5]], 'constraints': [(-100, 0), (-100, 0), (-100, 0)], 'crash_probability': [0.007, 0.006, 0.021, 0.0, 0.008, 0.023, 0.009, 0.011, 0.012, 0.017, 0.013]}, 1: {'id': 'la_0', 'lambdas_tried': [[-20, -20, 20, -1.8], [-24, -72, 57, -1.6], [-19, -55, 29, -3.8], [-83, -73, 55, -2.6], [-96, -29, 9, -2.4], [-12, -40, -9, -4.1], [-46, -90, 91, -3.2], [-42, -72, 22, -4.0], [-68, -73, 62, -4.1], [-17, -17, 25, -4.0], [-20, -52, 68, -2.9], [-98, -17, 61, -4.0], [-48, -7, 19, -3.7], [-61, -98, 2, -4.9], [-89, -23, 85, -3.7], [-95, -67, 89, -2.7]], 'constraints': [(-100, 0), (-100, 0), (0, 100)], 'crash_probability': [0.02, 0.023, 0.017, 0.019, 0.0, 0.02, 0.023, 0.019, 0.022, 0.018, 0.021, 0.0, 0.001, 0.021, 0.0, 0.0]}, 2: {'id': 'lc_0', 'lambdas_tried': [[-20, 20, -20, -1.8], [-106, 18, -47, -0.8], [-41, 38, -77, -1.4], [-84, 79, -9, -1.0], [-83, 4, -27, -2.2], [-71, 64, -40, -4.1]], 'constraints': [(-100, 0), (0, 100), (-100, 0)], 'crash_probability': [0.009, 0.0, 0.009, 0.0, 0.0, 0.0]}, 3: {'id': 'a_0', 'lambdas_tried': [[-20, 20, 20, -1.8], [-51, 14, -9, -2.9], [-11, 83, 90, -1.7], [-74, 38, 48, -3.6], [-58, 43, 42, -4.1], [-78, 49, 60, -1.3]], 'constraints': [(-100, 0), (0, 100), (0, 100)], 'crash_probability': [0.013, 0.004, 0.004, 0.001, 0.002, 0.0]}, 4: {'id': 'c_0', 'lambdas_tried': [[20, -20, -20, -1.8], [55, -26, -62, -2.8], [52, -10, -46, -1.0], [-2, -23, -85, -1.8], [83, -20, -55, -0.9], [57, -100, -78, -0.7], [68, -56, -90, -1.8], [60, -18, -73, -4.5], [-4, -93, -49, -1.8], [69, -18, -77, -3.2], [9, -34, -22, -3.7], [86, -99, -44, -3.3], [65, -57, -83, -1.5], [86, -19, -80, -2.7], [90, -26, -43, -2.1], [21, -41, -29, -2.7], [88, -49, -46, -3.2], [93, -80, -84, -4.4], [36, -52, -75, -0.4], [67, -43, -78, -3.6], [85, -26, -82, -1.1], [65, -92, -83, -3.8], [93, -28, -31, -1.1], [94, -41, -12, -1.1], [6, -91, -30, -4.0], [78, -79, -60, -1.3], [77, -14, -72, -1.6], [72, -63, -80, -2.8], [61, -103, -35, -1.6], [27, -80, -67, -3.3], [59, -18, -57, -3.7], [32, -11, -23, -0.9], [86, -64, -23, -3.3], [-6, -10, -26, -0.4], [65, -32, -30, -0.4], [47, -79, -96, -3.8]], 'constraints': [(0, 100), (-100, 0), (-100, 0)], 'crash_probability': [0.011, 0.012, 0.008, 0.008, 0.006, 0.02, 0.008, 0.013, 0.019, 0.008, 0.011, 0.021, 0.01, 0.006, 0.008, 0.025, 0.02, 0.006, 0.008, 0.013, 0.01, 0.022, 0.011, 0.014, 0.025, 0.026, 0.009, 0.005, 0.02, 0.02, 0.009, 0.009, 0.013, 0.006, 0.02, 0.008]}, 5: {'id': 'lc_1', 'lambdas_tried': [[20, -20, 20, -1.8]], 'constraints': [(0, 100), (-100, 0), (0, 100)], 'crash_probability': [0.014]}, 6: {'id': 'lc_2', 'lambdas_tried': [[20, 20, -20, -1.8], [28, 85, -15, -4.0], [42, 67, -46, -3.0], [35, 37, -93, -2.0], [48, 28, -11, -3.2], [87, 26, -41, -2.6], [86, 84, -83, -3.1], [0, 61, -109, -1.8], [38, 79, -14, -2.0], [-7, 57, -96, -0.19999999999999996], [50, 2, -87, -3.3], [82, 32, -40, -1.2], [56, 71, -65, -1.8], [44, 44, -109, -4.4], [28, 46, -50, -1.0], [23, 68, -64, -3.7], [59, 70, -77, -2.6], [16, 40, -27, -2.6], [97, 29, -100, -1.8], [89, 56, -87, -2.1], [24, 49, -14, -1.0], [16, 78, -79, -1.1], [25, 70, -61, -1.5], [56, 34, -57, -4.3], [20, 16, -44, -0.5], [39, 3, -83, -2.6]], 'constraints': [(0, 100), (0, 100), (-100, 0)], 'crash_probability': [0.01, 0.008, 0.006, 0.008, 0.009, 0.013, 0.005, 0.006, 0.012, 0.007, 0.007, 0.004, 0.006, 0.006, 0.008, 0.004, 0.007, 0.014, 0.012, 0.007, 0.009, 0.003, 0.008, 0.005, 0.006, 0.007]}, 7: {'id': 'a_2', 'lambdas_tried': [[20, 20, 20, -1.8], [19, 51, 64, -0.5], [73, 64, 82, -2.9], [7, 26, 3, -0.09999999999999998], [66, 54, 30, -4.5], [11, 54, 70, -2.6]], 'constraints': [(0, 100), (0, 100), (0, 100)], 'crash_probability': [0.0, 0.007, 0.0, 0.002, 0.002, 0.009]}}
    crash_prob = []
    res_matrix = dict()
    for k,v in res_dict.items():
        crash_prob.append((round(np.mean(v['crash_probability']),3),v['id']))
        st_id = v['id']
        for idx,ls in enumerate(v['lambdas_tried']):
            cr_prob = v['crash_probability'][idx]
            if (st_id,ls[3]) in res_matrix:
                ct = res_matrix[(st_id,ls[3])][1] + 1
                res_matrix[(st_id,ls[3])] =  (res_matrix[(st_id,ls[3])][0] + cr_prob/ct, ct)
            else:
                res_matrix[(st_id,ls[3])] = (cr_prob,1)
    res_matrix = {k: v for k, v in sorted(res_matrix.items(), key=lambda x: x[1],reverse=True)}
    plot_dict = dict()
    for k,v in res_matrix.items():
        print(k,':',round(v[0],4),v[1])
        if k[0] in plot_dict:
            plot_dict[k[0]].append((k[1],v[0]))
        else:
            plot_dict[k[0]] = [(k[1],v[0])]
    '''tr_labs = {'lc_0':'c_1','la_0':'la_0','c_2':'lc_0','a_0':'a_0',\
               'c_0':'c_0','a_1':'lc_1','la_1':'lc_2','c_1':'a_2'}'''
    for k,v in plot_dict.items():
        _sorted_res = sorted(v, key=lambda x: x[0])
        X = [x[0] for x in _sorted_res]
        Y = [x[1] for x in _sorted_res]
        plt.plot(X,Y,'-',label=k)
        print(k,len(Y))
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.show()
        
def show_raw_results():
    res_dict = {0: {'id': 'c_1', 'lambdas_tried': [[-10, -10, -10, -1.8], [-110, -60, -59, -3.9], [-87, -16, -16, -4.0], [-4, -68, -110, -3.2], [-65, -12, -73, -3.6], [-25, -82, -12, -1.9]], 'constraints': [(-100, 0), (-100, 0), (-100, 0)], 'crash_probability': [0.007, 0.0, 0.0, 0.008, 0.001, 0.033]}, 1: {'id': 'la_0', 'lambdas_tried': [[-10, -10, 10, -1.8], [-65, -66, 77, -2.1], [-45, -44, 5, -2.4], [-103, -12, 14, -3.2], [-31, -105, 71, -0.6], [-54, -87, 95, -4.4]], 'constraints': [(-100, 0), (-100, 0), (0, 100)], 'crash_probability': [0.004, 0.001, 0.004, 0.0, 0.044, 0.046]}, 2: {'id': 'lc_0', 'lambdas_tried': [[-10, 10, -10, -1.8], [-98, 10, -14, -4.4], [-34, 5, -24, -4.3], [-30, 22, -24, -0.30000000000000004], [-11, 34, -35, -2.8], [-52, 61, -32, -3.4], [-17, 24, -36, -0.7], [-39, 34, -57, -2.2], [-38, 27, -65, -3.1], [-52, 98, -57, -4.2], [-80, 28, -4, -3.8], [-75, 49, -99, -0.7], [-32, 59, -104, -3.4], [-69, 55, -33, -2.7], [-3, 14, -14, -0.19999999999999996], [-103, 55, -7, -1.5]], 'constraints': [(-100, 0), (0, 100), (-100, 0)], 'crash_probability': [0.005, 0.001, 0.002, 0.006, 0.015, 0.003, 0.013, 0.01, 0.004, 0.001, 0.004, 0.004, 0.009, 0.002, 0.007, 0.0]}, 3: {'id': 'a_0', 'lambdas_tried': [[-10, 10, 10, -1.8], [-30, 70, 95, -2.3], [-43, 9, 70, -2.4], [-18, 12, 73, -1.2], [-105, 98, 30, -1.8], [-15, 55, 44, -2.7]], 'constraints': [(-100, 0), (0, 100), (0, 100)], 'crash_probability': [0.005, 0.003, 0.006, 0.011, 0.0, 0.009]}, 4: {'id': 'c_0', 'lambdas_tried': [[10, -10, -10, -1.8], [67, -93, -56, -3.2], [90, -22, -98, -3.5], [64, -70, -53, -3.2], [49, -69, -25, -2.2], [87, -72, -86, -4.2], [38, -33, -106, -2.0], [-8, -16, -103, -0.9], [22, -17, -49, -1.8], [68, -91, -51, -2.8], [35, -42, -75, -3.1], [27, -46, -82, -3.5], [15, -11, -94, -4.2], [-7, -79, -33, -1.3], [26, -49, -81, -3.0], [14, -108, -20, -0.09999999999999998], [88, -45, -94, -1.9], [7, -92, -22, -0.09999999999999998], [12, -73, -62, -3.1], [64, -21, -76, -1.3], [11, -10, -52, -2.8], [68, -13, -58, -2.6], [5, -85, -71, -2.0], [-1, -72, -19, -3.5], [52, -41, -46, -3.0], [-7, -47, -42, -4.9]], 'constraints': [(0, 100), (-100, 0), (-100, 0)], 'crash_probability': [0.011, 0.045, 0.006, 0.03, 0.034, 0.017, 0.004, 0.011, 0.01, 0.042, 0.009, 0.006, 0.009, 0.042, 0.008, 0.041, 0.006, 0.037, 0.046, 0.004, 0.01, 0.009, 0.036, 0.042, 0.006, 0.032]}, 5: {'id': 'lc_1', 'lambdas_tried': [[100, -10, 10, -1.8], [88, -79, 57, -1.7], [83, -44, -9, -1.2], [33, -92, 59, -2.3], [52, -28, 89, -2.6], [29, -57, 23, -4.5], [29, -76, 62, -3.9], [51, -27, 42, -4.4], [66, -75, 59, -3.6], [68, -98, 11, -3.8], [73, -67, 3, -4.2], [46, -52, 9, -3.9], [36, -105, 56, -1.4], [56, -7, 29, -4.2], [34, -33, 15, -0.7], [36, -92, 69, -1.1]], 'constraints': [(0, 100), (-100, 0), (0, 100)], 'crash_probability': [0.025, 0.035, 0.034, 0.039, 0.031, 0.033, 0.033, 0.024, 0.041, 0.041, 0.044, 0.026, 0.036, 0.024, 0.04, 0.05]}, 6: {'id': 'lc_2', 'lambdas_tried': [[10, 10, -10, -1.8], [79, 33, -3, -4.3], [75, 37, -61, -0.6], [67, 84, -66, -0.30000000000000004], [37, 64, -45, -2.8], [-10, 70, -13, -2.1], [19, 17, -67, -4.6], [24, 86, -30, -4.4], [13, 43, -46, -4.2], [95, 43, -52, -4.2], [93, 74, -70, -2.6], [16, 77, -54, -1.8], [62, 28, -95, -2.4], [90, 32, -18, -0.8], [33, 83, -71, -4.2], [39, 12, -19, -1.6]], 'constraints': [(0, 100), (0, 100), (-100, 0)], 'crash_probability': [0.006, 0.005, 0.007, 0.006, 0.006, 0.005, 0.004, 0.009, 0.005, 0.007, 0.006, 0.005, 0.008, 0.009, 0.003, 0.007]}, 7: {'id': 'a_2', 'lambdas_tried': [[10, 10, 10, -1.8], [86, 36, 14, -2.2], [91, 79, 62, -3.6], [30, 98, 95, -0.8], [88, 66, 7, -4.1], [45, 19, 24, -1.7], [16, 52, 47, -1.5], [1, 31, 57, -1.3], [64, 62, 0, -1.9], [97, 80, 86, -3.7], [89, 18, 26, -2.7], [14, 46, 30, -1.7], [83, 39, 23, -2.8], [-5, 12, 93, -0.7], [11, 86, 20, -3.4], [40, 11, 18, -2.0]], 'constraints': [(0, 100), (0, 100), (0, 100)], 'crash_probability': [0.002, 0.0, 0.0, 0.002, 0.0, 0.0, 0.008, 0.008, 0.005, 0.0, 0.0, 0.007, 0.0, 0.006, 0.006, 0.0]}}
    crash_prob = []
    for k,v in res_dict.items():
        crash_prob.append((round(max(v['crash_probability']),3),v['id']))
    crash_prob.sort(key=lambda tup:tup[0], reverse = True)
    print(crash_prob)


def temp_process_ce_result_file():
    file_name = root_path+'sumo_ce_results_final.out'
    res_dict = dict()
    ctr = 0
    with open(file_name) as f:
        for line in f:
            ctr = ctr + 1
            print('read',round(ctr/48540330,5),'%')
            _iter,_level = line[line.rfind('(')+1:line.index(')')].split(',')
            nums = int(line[line.index('/')+1:line.rfind("'")])
            crashes = int(line[line.rfind(':')+1:line.rfind(',')])
            res_dict[(int(_iter),int(_level))] = (crashes,nums)
            
    Y = []
    for k,v in res_dict.items():
        _sum_c,_sum_n = 0,0
        if int(k[1]) == 2:
            _sum_c =  _sum_c + v[0]
            _sum_n =  _sum_n + v[1]
            Y.append(_sum_c/_sum_n)
            _sum_c,_sum_n = 0,0
    plt.plot(np.arange(0,100),Y)
    plt.show()
            
def final_results_plot():
    import ast
    cmc,br_2,ce = None,None,None
    f = root_path+'cmc_res_final.list'
    with open(f) as fp:  
        line = fp.readline()
        cmc = ast.literal_eval(line)
    f = root_path+'lambda_opt_res_final_prob.list'
    with open(f) as fp:  
        line = fp.readline()
        br_2 = ast.literal_eval(line)
    f = root_path+'ce_opt_res_final_prob.list'
    with open(f) as fp:  
        line = fp.readline()
        ce = ast.literal_eval(line)

    X = [0,1,2,3]
    Y = [np.mean([x[1] for x in cmc]),np.mean([x[1] for x in br_2]),np.mean([x[1] for x in ce])]
    cmc_b = [x[1] for x in cmc]
    br_2_b = [x[1] for x in br_2]
    ce_b = [x[1] for x in ce]
    print(np.mean(cmc_b),np.mean(br_2_b),np.mean(ce_b))
    #plt.boxplot([x[1] for x in cmc],meanline=True,showmeans=True)
    #print(np.mean([x[1] for x in br_1]))
    #print(np.mean([x[1] for x in br_2]))
    #print(np.mean([x[1] for x in ce]))
    data = [cmc_b,br_2_b,ce_b]
    fig, ax = plt.subplots()
    ax.boxplot(data,meanline=True,showmeans=True)
    plt.xticks([1, 2, 3], ['CMC \n $ \mu = 5.6 x 10^{-4}$', 'BR \n $\mu=4.07 x 10^{-2}$', 'CE \n $\mu=3.36 x 10^{-2}$'])
    plt.ylabel('$p_{\epsilon}$')
    plt.show()
    
    fig, ax = plt.subplots()
    plt.plot(np.arange(1,101),cmc_b,linestyle = '-', label='CMC')
    plt.plot(np.arange(1,101),br_2_b,linestyle = '--',label = 'BR')
    plt.plot(np.arange(1,101),ce_b,linestyle = ':',label='CE')
    plt.xlabel('iterations')
    plt.ylabel('$p_{\epsilon}$')
    plt.legend(bbox_to_anchor=(1.05, 1.15), ncol=3)
    plt.show()
    cmc_var_dict = dict()
    tot_crashes = 0
    for idx,cr_rate in enumerate(cmc_b):
        if cr_rate > 0:
            num_crashes = int(round(cr_rate * 1000))
            tot_crashes = tot_crashes + num_crashes
    cmc_var = [1]*tot_crashes + [0]*(100000 - tot_crashes)
    print(np.var(cmc_var))

    
def final_result_var_plot():
    import ast
    import matplotlib.ticker as ticker
    def _eval_exp(x,lambda_param,a):
        return a*(1/lambda_param) * np.exp(-1*x/lambda_param)
    def _eval_vel_s(vel_s):
        vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
        def _gauss(x,mu,sigma,A):
            return A*math.exp(-(x-mu)**2/2/sigma**2)
    
        def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
            return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]
        return _bimodal([vel_s],*vel_bimodal_params)[0]
    f = root_path+'lambda_opt_res_final.list'
    data_str = None  
    with open(f) as fp:  
        line = fp.readline()
        data_str = ast.literal_eval(line)
    f = root_path+'ce_opt_res_final.list'
    ce_data_str = None  
    with open(f) as fp:  
        line = fp.readline()
        ce_data_str = ast.literal_eval(line)
    cmc_var_dict = {3: 0.000999, 7: 0.000999, 8: 0.001996, 12: 0.000999, 13: 0.000999, 15: 0.000999, 17: 0.001996, 18: 0.000999, 21: 0.000999, 22: 0.000999, 30: 0.000999, 33: 0.000999, 34: 0.000999, 35: 0.000999, 36: 0.000999, 37: 0.000999, 39: 0.001996, 40: 0.000999, 43: 0.001996, 46: 0.0029909999999999997, 48: 0.000999, 51: 0.000999, 52: 0.000999, 53: 0.000999, 55: 0.001996, 56: 0.000999, 57: 0.000999, 58: 0.000999, 59: 0.000999, 63: 0.001996, 66: 0.000999, 67: 0.000999, 69: 0.000999, 75: 0.003984, 79: 0.001996, 83: 0.000999, 85: 0.000999, 87: 0.000999, 88: 0.000999, 89: 0.000999, 91: 0.000999, 94: 0.000999, 97: 0.000999, 99: 0.000999}
    cmc_var = 0.00056
    br_w_u_list = []
    for i in data_str:
        iter = i[0]
        iter_num = i[0]
        iter_dets = i[1]
        w_u_list_iter_br = []
        for vals in iter_dets:
            q_u = vals[0]
            p_u = vals[1]
            w_u = p_u/q_u
            if w_u != 0:
                w_u_list_iter_br.append(w_u)
        br_w_u_list.append(w_u_list_iter_br)        
    
    ce_w_u_list = []
    for i in ce_data_str:
        iter = i[0]
        iter_num = i[0]
        iter_dets = i[1]
        w_u_list_iter_ce = []
        for vals in iter_dets:
            q_u = vals[0]
            p_u = vals[1]
            w_u = p_u/q_u
            if w_u != 0:
                w_u_list_iter_ce.append(w_u)
        ce_w_u_list.append(w_u_list_iter_ce)        
    X,Y,y_all = [],[],[]
    num_crashes = 0
    
    
    [y_all.append(x) for x in br_w_u_list]
    for iter,entry in enumerate(br_w_u_list):
        if iter in cmc_var_dict:
            X.append(iter)
            Y.append(np.var(entry))
            #print('br',iter,np.min(entry),np.max(entry),np.mean(entry))
        
    X_2,Y_2,y2_all = [],[],[]
    [y2_all.append(x) for x in ce_w_u_list]
    for iter,entry in enumerate(ce_w_u_list):
        if iter in cmc_var_dict:
            X_2.append(iter)
            Y_2.append(np.var(entry))
            #print('ce',iter,np.min(entry),np.max(entry),np.mean(entry))
    print('br',Y)
    print(min(Y),max(Y),np.mean(Y))
    print('ce',Y_2)
    print(min(Y_2),max(Y_2),np.mean(Y_2))
    print('% reduction', (np.mean(Y_2) - np.mean(Y))/np.mean(Y_2))
    def myticks(y,pos):
        if y == 0: return "$0$"    
        if y > 0:
            exponent = int(np.log10(y))
            coeff = y/10**exponent
    
            return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)
        else:
            return y
    
    
    f, (ax, ax2) = plt.subplots(1,2)
    #ax.scatter(X,Y,color='b',label='BR',marker='s')
    #ax2.scatter(X,Y_2,color='gray',marker='o',label='CE')
    
    ax.boxplot(Y,meanline=True,showfliers=False)
    ax2.boxplot(Y_2,meanline=True,showfliers=False)
    
    #ax.set_ylim(0,.01)  # outliers only
    #ax2.set_ylim(0, 10000)  # most of the data
    '''
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    '''
    ax2.legend(loc='upper right', ncol=2)
    ax.legend(loc='upper right', ncol=2)
    ax.set_xlabel('BR')
    ax2.set_xlabel('CE')
    ax.set_ylabel('$\sigma^{2}$')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    #ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
    plt.show()
    
    '''
    data = [Y,Y_2]
    fig, ax = plt.subplots()
    ax.boxplot(data,meanline=True,showmeans=True)
    plt.xticks([1, 2], ['BR', 'CE'])
    plt.ylabel('$\sigma^{2}$ ratio')
    plt.show()
    '''
''' all runs below '''      


final_results_plot()


