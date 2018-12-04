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


def count_negative_range_rate():
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    
  
    
def plot_ttc():
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    file_name = '/media/atrisha/Data/datasets/SPMD/processing_lists/histogram_distr.json'
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    
    print(min(ttc_1),max(ttc_1))
    print(min(ttc_2),max(ttc_2))
    print(min(ttc_3),max(ttc_3))
    
    #print(lambda_15_25)
    #print(lambda_25_35)
    
    def _exp(X,lambda_param,a):
        return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    def _pareto(X,param):
        return [(param*math.pow(.1,param))/(math.pow(x, param+1)) if x>=.1 else 0 for x in X]
    
    
    hist = np.histogram(ttc_1,bins = np.arange(0,math.ceil(max(ttc_1))+.01,.01),density=True)
    X = [float(x/100) for x in np.arange(0,len(hist[0]))]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    store_hist_objects('ttc_inv_5_15', dict_val, True)
    
    
    
    hist = np.histogram(ttc_2,bins = np.arange(0,math.ceil(max(ttc_2)),.01),density=True)
    X = [float(x/100) for x in np.arange(0,len(hist[0]))]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    store_hist_objects('ttc_inv_15_25', dict_val, True)
    
    hist = np.histogram(ttc_3,bins = np.arange(0,math.ceil(max(ttc_3)),.01),density=True)
    X = [float(x/100) for x in np.arange(0,len(hist[0]))]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    print(popt)
    dict_val = list(zip(X,Y))
    store_hist_objects('ttc_inv_25_35', dict_val, True)
    
    '''
    ttc_1 = [1/x[1] for x in vel_5_15]
    ttc_2 = [1/x[1] for x in vel_15_25]
    ttc_3 = [1/x[1] for x in vel_25_35]
    
    
    bins = np.arange(.1,math.ceil(max(ttc_1))+.01,.1)
    hist = np.histogram(ttc_1,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    index,value = max(enumerate(Y),key = operator.itemgetter(1))
    print(index,value,X[index])
    
    
    
    bins = np.arange(.1,math.ceil(max(ttc_2))+.01,.1)
    hist = np.histogram(ttc_2,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    index,value = max(enumerate(Y),key = operator.itemgetter(1))
    print(index,value,X[index])
    
    bins = np.arange(.1,math.ceil(max(ttc_3))+.01,.1)
    hist = np.histogram(ttc_3,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [x/100 for x in hist[0]]
    sum_Y = sum(Y)
    Y = [x/sum_Y for x in Y]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    index,value = max(enumerate(Y),key = operator.itemgetter(1))
    print(index,value,X[index])
    
    '''
    plt.show()
    


    

def plot_range_vel():
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    
    
    
    def _exp(X,lambda_param,a):
        return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    def _pareto(X,param):
        x_m = .01
        return [1 if x<x_m else math.pow((x_m/x),param) for x in X]
    
    
    
    hist = np.histogram(all_range_inv,bins = np.arange(0,math.ceil(max(all_range_inv)),.01),density=True)
    X = [x/100 for x in np.arange(0,len(hist[0]))]
    Y = [x/100 for x in hist[0]]
    #popt, pcov = curve_fit(_pareto, X, Y)
    #plt.plot(X, _pareto(X, *popt))
    #plt.plot(X,Y,'.')
    #print(popt)
    '''
    '''
    hist = np.histogram(range_inv_1,bins = np.arange(0,math.ceil(max(range_inv_1)),.05),density=True)
    X = np.arange(0,math.ceil(max(range_inv_1)),.05)[1:]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    dict_val = list(zip(X,Y))
    store_hist_objects('range_inv_5_15', dict_val, True)
    
    
    hist = np.histogram(range_inv_2,bins = np.arange(0,math.ceil(max(range_inv_2)),.05),density=True)
    X = np.arange(0,math.ceil(max(range_inv_1)),.05)[1:]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    dict_val = list(zip(X,Y))
    store_hist_objects('range_inv_15_25', dict_val, True)
    
    hist = np.histogram(range_inv_3,bins = np.arange(0,math.ceil(max(range_inv_3)),.05),density=True)
    X = np.arange(0,math.ceil(max(range_inv_1)),.05)[1:]
    Y = [float(x/100) for x in hist[0]]
    popt, pcov = curve_fit(_pareto, X, Y)
    plt.plot(X, _pareto(X, *popt))
    plt.plot(X,Y,'.')
    dict_val = list(zip(X,Y))
    store_hist_objects('range_inv_25_35', dict_val, True)
    
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
                dir_path_wsu = '/media/atrisha/Data/datasets/SPMD/processing_lists/wsu_seq_data_for_cutins/'
                file_name_wsu = row[0]+'-'+row[1]+'.csv'
                if os.path.isfile(dir_path_wsu+file_name_wsu):
                    with open(dir_path_wsu+file_name_wsu, 'r', newline='') as csv_file_wsu:
                        csv_reader_wsu = csv.reader(csv_file_wsu, delimiter=',')
                        for row_wsu in csv_reader_wsu:
                            wsu_seq_dict[row_wsu[0]+'-'+row_wsu[1]+'-'+row_wsu[2]] = row_wsu
                print('processed',round((line_count / 74450)*100,3),'%')
                range,range_rate = float(row[5]) , -float(row[6])
                ttc = float(row[5]) / -float(row[6])
                obstacle_id = row[4]
                ''' code to get next 5 secs sequence '''
                next_five_sec_timestamp = np.arange(int(row[2])+10,int(row[2])+510,10)
                seq_file_name = '/media/atrisha/Data/datasets/SPMD/processing_lists/front_target_seq_for_cutins/'+row[0]+'-'+row[1]+'.csv'
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
                   
                with open('/media/atrisha/Data/datasets/SPMD/processing_lists/interac_seq_data.csv', 'a', newline='') as csvfile_int_seq:
                    writer_seq = csv.writer(csvfile_int_seq, delimiter=',')
                    writer_seq.writerow([inst_id,0,vel_mps,vel_lc,range,range_rate])
                    for row in list(zip(frame_id,next_five_sec_v_s,next_five_sec_v_lc,next_five_sec_range,next_five_sec_range_rate)):
                        entry_seq = [inst_id] + list(row)
                        writer_seq.writerow(entry_seq)
                
            line_count = line_count + 1
            '''if line_count % 1000 == 0:
                bins = [0,10,50,100]
                plt.hist(max_diff_l,bins=bins)
                plt.show()'''
    ttc_dict = {'low_speed':low_speed_ttc,'med_speed':med_speed_ttc,'high_speed':high_speed_ttc}
    '''with open('/media/atrisha/Data/datasets/SPMD/processing_lists/all_cutin_ttc.json','w') as file:
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
        for _time in range(max_time_length+1):
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'vehicle_cut_in_events.csv'
    range_list = []
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            range_list.append(float(row[5]))
    print(min(range_list),max(range_list))
    

            
                  
            
            
def find_min_range_value_in_all():
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/front_target_data/' 
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
        print('complete:',(ct/file_count)*100,'min range',min(range_list))
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
    if os.path.isfile('/media/atrisha/Data/datasets/SPMD/processing_lists/all_cutin_ttc.json'):
        js = open('/media/atrisha/Data/datasets/SPMD/processing_lists/all_cutin_ttc.json')
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
            for level in range(1,len(ttc_det.keys())+1):
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
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/lambda_dist.json','w') as file:
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
        for level in range(1,max_level):
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
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/vehicle_cut_in_events.csv','r') as csvfile:
        csv_reader = csv.reader(csvfile,delimiter=',')
        for row in csv_reader:
            if float(row[6]) < 0:
                range = float(row[5])
                ttc = range / -float(row[6])
                for k,v in data_dict.items():
                    low,high = k.split('-')
                    if int(low) <= range < int(high):
                        data_dict[k].append(int(round(ttc)))
                        break
    for k,v in data_dict.items():
        hist = np.histogram(v,bins=np.arange(350),density=True)
        plt.plot(np.arange(349),hist[0])
    plt.show()
    
    
    
        

def smooth_velocity_curves():
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/interac_seq_data.csv','r') as csvfile:
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
                    
                    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/interac_seq_data_smooth.csv', 'a', newline='') as csvfile_int_seq:
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
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'wsu_cut_in_list.csv'
    wsu_dict = dict()
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0]+'-'+row[1] in wsu_dict.keys():
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
            else:
                wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
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
    store_hist_objects('veh_dist', dict_val, True)
    #params,cov=curve_fit(_bimodal,X,Y)
    #print(params)
    #plt.plot(X,_bimodal(X,*params))
    axes = plt.gca()
    axes.set_xlim([0,max_vel])
    axes.set_ylim([0,.1])
    plt.plot(X,Y)
    plt.show()        
    
    
 
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
    return 1.5/(1+math.exp(-1 * (x-1))) - 0.5 if x<30 else -0.05*x + 2.5


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
    '''
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
    
    #print('probabilities:',prob_vel_lc,prob_ttc_inv,prob_range_inv)
    return prob_vel_lc*prob_ttc_inv*prob_range_inv


def solve_lambda(delta_u,p_a):
    low,high = -10000,10000
    lambda_list = np.arange(low,high)
    min = 1
    lambda_estimate = None
    level_count = 0
    while True:
        for l in lambda_list:
            p_a_calculated = np.exp(1.73205080756888*l)/l - 1/l if l!=0 else 1.73205080756888
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
        if level_count >= 5:
            break
    return lambda_estimate
            
        

def plot_pareto_front():
    X,Y,Z = [],[],[]
    X_pareto,Y_pareto,Z_pareto = [],[],[]
    count = 0
    state_data = []
    if not os.path.isfile('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/state_data.dmp'):
        dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
        file_name = 'wsu_cut_in_list.csv'
        wsu_dict = dict()
        with open(dir_path+file_name, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0]+'-'+row[1] in wsu_dict.keys():
                    wsu_dict[row[0]+'-'+row[1]+'-'+row[2]].append(row)
                else:
                    wsu_dict[row[0]+'-'+row[1]+'-'+row[2]] = [row]
        dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
        file_name = 'vehicle_cut_in_events.csv'
        ttc_secs,vel_kph = [],[]
        
        vel_5_15,vel_15_25,vel_25_35 = [],[],[]
        with open(dir_path+file_name, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                print('processed',count)
                if float(row[6]) < 0 :
                    range = float(row[5])
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
                        Y.append(util_ttc(range/(vel_s-vel_lc),util_ttc_param))
                        util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
                        Z.append(util_dist(range,util_dist_param))
                        state_data.append((vel_s,vel_lc,range))
                count = count + 1
        '''            
        with open('/media/atrisha/Data/datasets/SPMD/processing_lists/interac_seq_data.csv','r') as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=',')
            for row in csv_reader:
                print('processed',count/1755165)
                if int(row[1]) == 0 and row[3] != '':
                    vel_lc = float(row[3])
                    vel_s = float(row[2])
                    range = float(row[4])
                    if vel_lc != vel_s:
                        util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
                        X.append(util_progress(vel_lc))
                        Y.append(util_ttc(range/(vel_s-vel_lc),util_ttc_param))
                        util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
                        Z.append(util_dist(range,util_dist_param))
                        state_data.append((vel_s,vel_lc,range))
                count = count + 1
        '''
            
        
        pareto_array = np.asarray(list(zip(X,Y,Z)))
        pareto_array.reshape((len(X),3))
        print('calculating pareto front...')
        pareto_points = is_pareto_efficient(pareto_array)
        
                
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/state_data.dmp','wb')
        pickle.dump(state_data,file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/X.dmp','wb')
        pickle.dump(X,file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/Y.dmp','wb')
        pickle.dump(Y,file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/Z.dmp','wb')
        pickle.dump(Z,file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/pareto_points.dmp','wb')
        pickle.dump(pareto_points,file_pi)
                
    else:
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/state_data.dmp','rb')
        state_data = pickle.load(file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/X.dmp','rb')
        X = pickle.load(file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/Y.dmp','rb')
        Y = pickle.load(file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/Z.dmp','rb')
        Z = pickle.load(file_pi)
        file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/pareto_points.dmp','rb')
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
    start_time = time.time()
    for idx,data in enumerate(data_points):
        dist_to_all_paretos = [np.linalg.norm([data[0]-x[0],data[1]-x[1],data[2]-x[2]]) for x in pareto_front]
        min_delta_to_pareto = min(dist_to_all_paretos)
        vel_s,vel_lc,range = state_data[idx]
        ttc = range/(vel_s-vel_lc)
        #print('state data:',idx,state_data[idx],ttc)
        prob_action = get_action_probability((ttc,vel_lc,range))
        #print(prob_action,min_delta_to_pareto)
        lambda_est = solve_lambda(min_delta_to_pareto,prob_action)
        prob__delta_util_list.append((prob_action,min_delta_to_pareto,lambda_est))
        if idx > 0:
            time_diff = ((time.time() - start_time) / idx ) * (total_count - idx)
        else:
            time_diff = 0
            
        print(str(datetime.timedelta(seconds=time_diff)),idx,round(idx/total_count,4),lambda_est)
    file_pi = open('/media/atrisha/Data/datasets/SPMD/processing_lists/cache_data/lambda_vals.dmp','wb')
    pickle.dump(prob__delta_util_list,file_pi)
        
    
    
        
        
        
        
    
                
    

    
    
''' all runs below '''

plot_pareto_front()


                