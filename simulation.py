'''
Created on Nov 6, 2018

@author: atrisha
'''
import random
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from data_analysis import calc_util
import math
import scipy.stats as st
from sklearn.svm import SVR
import itertools
from mpl_toolkits.mplot3d import axes3d
import pickle
import os.path


def get_markov_chain_trainsition_matrix():
    file_name = '/media/atrisha/Data/datasets/SPMD/processing_lists/ttc_transition_models.dmp'
    if os.path.isfile(file_name):
        print('models found')
        filehandler = open(file_name,'rb')
        clf = pickle.load(filehandler)
        return clf
    else:
        print('models need to be built :(')
        js = open('/media/atrisha/Data/datasets/SPMD/processing_lists/all_cutin_ttc.json')
        ttc_dict = json.load(js)
        transition_matrix_dict = dict()
        total_count = dict()
        data_dict = dict()
        print('preprocessing...')
        for speed_level,ttc_list in ttc_dict.items():
            transition_matrix_dict[speed_level] = np.zeros((101,101), dtype=float)
            total_count[speed_level] = dict()
            for ttc_temporal_list in ttc_list:
                for ttc in list(zip(ttc_temporal_list[:-1],ttc_temporal_list[1:])):
                    if ttc[0] and ttc[1] >= 0:
                        rounded_ttc = tuple(ttc)
                        if ttc[0] >= 10:
                            rounded_ttc = (10,rounded_ttc[1])
                        else:
                            rounded_ttc = (round(ttc[0],1),ttc[1])
                        if ttc[1] >= 10:
                            rounded_ttc = (rounded_ttc[0],10)
                        else:
                            rounded_ttc = (rounded_ttc[0],round(ttc[1],1))
                        if int(rounded_ttc[0]*10) in total_count[speed_level].keys():
                            total_count[speed_level][int(rounded_ttc[0]*10)] = total_count[speed_level][int(rounded_ttc[0]*10)] + 1
                        else:
                            total_count[speed_level][int(rounded_ttc[0]*10)] = 1
                        transition_matrix_dict[speed_level][int(rounded_ttc[0]*10),int(rounded_ttc[1]*10)] = (transition_matrix_dict[speed_level][int(rounded_ttc[0]*10),int(rounded_ttc[1]*10)] + 1)
        for k,v in transition_matrix_dict.items():
            for i in range(0,101):
                if i in total_count[k].keys():
                    transition_matrix_dict[k][i,:] = transition_matrix_dict[k][i,:] / total_count[k][i]
        print('building model...')
        data_x = list(itertools.product(np.arange(101),np.arange(101)))
        data_x_dict,data_y_dict = dict(),dict()
        clf = dict()
        for speed_level,transition_array in transition_matrix_dict.items():
            data_y_dict[speed_level] = []
            data_x_dict[speed_level] = []
            for x in data_x:
                if transition_array[x[0],x[1]] >= 0:
                    data_x_dict[speed_level].append(x)
                    data_y_dict[speed_level].append(float(transition_array[x[0],x[1]]))
            clf[speed_level] = SVR(gamma=0.001, C=10.0, epsilon=0.0001)
            clf[speed_level].fit(np.asarray(data_x_dict[speed_level]),np.asarray(data_y_dict[speed_level]))
            print('model built for',speed_level)
        print('model building done')
        file_pi = open(file_name,'wb')
        pickle.dump(clf,file_pi)
        return clf                


def calculate_covariance(data,mean):
    data_list = data.tolist()
    sum = 0
    for d in data_list:
        sum = sum + (d[0]-mean[0,])*(d[0]-mean[0,])
    cov_X_X = sum / (len(data_list) - 1)
    sum=0
    for d in data_list:
        sum = sum + (d[0]-mean[0,])*(d[1]-mean[1,])
    cov_X_Y = sum / (len(data_list) - 1)
    return cov_X_X,cov_X_Y                 
            
def get_range_and_range_rate_distr(vel_kph):
    vel = None
    if 0 < vel_kph < 30:
        vel = 'low'
    elif 30 <= vel_kph < 80:
        vel = 'med'
    elif vel_kph >= 80:
        vel = 'high'
    wsu_dict = dict()
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/wsu_cut_in_list.csv', 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            wsu_dict[(row[0],row[1],row[2])] = row[16]
    dir_path = '/media/atrisha/Data/datasets/SPMD/processing_lists/'
    file_name = 'vehicle_cut_in_events.csv'
    data_low,data_med,data_high = [],[],[]
    with open(dir_path+file_name, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if (row[0],row[1],row[2]) in wsu_dict.keys():
                vel_kph = float(wsu_dict[(row[0],row[1],row[2])])
                if 0 < vel_kph < 30:
                    data_low.append((float(row[5]),float(row[6])))
                elif 30 <= vel_kph < 80:
                    data_med.append((float(row[5]),float(row[6])))
                elif vel_kph >= 80:
                    data_high.append((float(row[5]),float(row[6])))
    '''data = np.array(data,dtype=float)
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    #calculate_covariance(data,mean)
    X, Y = np.random.multivariate_normal(mean, cov, 1).T
    return list(zip(X,Y))'''
    data_dict = {'low':data_low,'med':data_med,'high':data_high}
    d = data_dict[vel]
    return_real_data = False
    if return_real_data:
        indx = np.random.randint(len(d))
        return d[indx]
    else:
        X = [x[0] for x in d]
        Y = [x[1] for x in d]
        X_Y = np.column_stack((X,Y))
        mu = np.asarray([np.mean(X),np.mean(Y)])
        Sigma = np.cov(X_Y,rowvar=False)
        _x,_y = np.random.multivariate_normal(mu,Sigma,5000).T
        #plt.plot(_x,_y,'.')
        #plt.show()
        return (_x[0],_y[0])   
            
def max_ttc_diff(range,range_rate):
    max_dec = 9
    if range_rate == 0:
        init_ttc = 10
    else:
        init_ttc = min(range/abs(range_rate),10)*10
    return init_ttc - (min(int((range - ((range_rate * .1) + (0.5 * max_dec * .01))) / (range_rate + (max_dec * .1))),10)*10)
                
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def sample_ttc_from_transition(ttc_t_minus_1,ttc_transition_model,range,range_rate):
    ttc_list = [ttc_t_minus_1]
    for t in np.arange(51):
        population = np.arange(0,10.1,.1)
        access_indx = int(ttc_list[-1]*10)
        predictions = []
        max_ttc_delta = max_ttc_diff(range,range_rate)
        for x in np.arange(101):
            if abs((ttc_t_minus_1*10) - x) < max_ttc_delta:
                pred_ttc = max(0,ttc_transition_model.predict([(access_indx,x)])[0])
                predictions.append(pred_ttc)
            else:
                predictions.append(0)
        sum_predictions = sum(predictions)
        if sum_predictions == 0:
            next_ttc = int(ttc_list[-1]*10)
        else:
            weights = [x/sum_predictions for x in predictions]
            total_sum = sum(weights)
            if abs(total_sum - 1) > 0.0001:
                #print(ttc_t_minus_1,access_indx,total_sum)
                return ttc_t_minus_1
            custm = stats.rv_discrete(name='custm', values = (population,weights))
            #next_ttc = custm.rvs(size=10)
            next_ttcs = np.random.choice(population,10,p=list(weights))
            next_ttc = list(filter(lambda x:x>0,next_ttcs))[0]
        ttc_list.append(next_ttc)
    return running_mean(ttc_list[1:],3).tolist() + ttc_list[-2:]
    


def show_average():
    samples = []
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/monte_carlo_results.csv', 'w', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            samples = row
    print(np.mean(samples))
    

def monte_carlo_sampler_calc_prob():
    vel_kph = np.random.randint(0,120)
    vel = None
    if 0 < vel_kph < 30:
        vel = 'low_speed'
    elif 30 <= vel_kph < 80:
        vel = 'med_speed'
    elif vel_kph >= 80:
        vel = 'high_speed'
    ttc_transition_model = get_markov_chain_trainsition_matrix()[vel]
    failed_count = 0
    sample_count = 0
    num_trial_samples = 0
    num_success_samples = []
    
    range_range_rate_sample = get_range_and_range_rate_distr(vel_kph)
    range = abs(range_range_rate_sample[0])
    range_rate = range_range_rate_sample[1]
    ttc_init = min(range/abs(range_rate),10)
    range_init,range_rate_init = range,range_rate 
    ttc_sample_next_five_sec = sample_ttc_from_transition(ttc_init,ttc_transition_model,range_init,range_rate_init)
    ttc_t_minus_1 = ttc_init
    num_trials = 0
    time_thresh = 50
    crash_count = 0
    range_init,range_rate_init = range,range_rate
    trial_count = 0
    result_array,weight_array = [],[]
    range_rate_arr = []
    while True:
        range_range_rate_sample = get_range_and_range_rate_distr(vel_kph)
        range = abs(range_range_rate_sample[0]) 
        range_rate = range_range_rate_sample[1]
        time_ticker = 0
        if range_rate == 0:
            ttc_init = 10
        else:
            ttc_init = range/-range_rate
        if ttc_init < 0:
            ttc_init = 10
        else:
            ttc_init = min(ttc_init,10)
        ttc_sample_next_five_sec = sample_ttc_from_transition(ttc_init,ttc_transition_model,range,range_rate)
        #p_distribution_prob = get_p_distr_prob(ttc_transition_model,ttc_sample_next_five_sec)
        ttc_t_minus_1 = ttc_init
        range_init,range_rate_init = range,range_rate
        dec = []
        while time_ticker < time_thresh :
            time_ticker = time_ticker + 1
            range_prime = range + (range_rate * .1)
            if range_prime < 0.01:
                crash_count = crash_count + 1
                result_array.append(1)
                if len(dec) > 0:
                    range_rate_arr.append((range_init,range_rate_init*3.6,round(max(dec),2)))
                else:
                    range_rate_arr.append((range_init,range_rate_init*3.6,None))
                with open('/media/atrisha/Data/datasets/SPMD/processing_lists/mc_results.out', 'a') as wfile:
                    str_line = str(num_trials) + ',' +str(crash_count) + '\n'
                    wfile.write(str_line)
                break
            ttc_t = ttc_sample_next_five_sec[time_ticker]
            old_range_rate = range_rate
            range_rate = -1 * (range_prime / ttc_t) 
            if abs(range_rate-old_range_rate) > 100:
                f=6
            dec.append(abs(range_rate-old_range_rate)*10)
            range = range_prime
            ttc_t_minus_1 = ttc_t
        #print(range_rate_arr)
        if range_prime >= 0.01:
            result_array.append(0)
        num_trials = num_trials + 1
        print(num_trials,round(range_prime,2),crash_count)
        if crash_count >= 100:
            break
    intvl = st.t.interval(0.95,len(result_array)-1,loc = np.mean(result_array), scale = st.sem(result_array))
    if crash_count != 0:
        print('crash prob based on MC is '+ str(np.mean(result_array)))
        print(range_rate_arr)
        
        
        
def calc_ttc_from_lambda(param_lambda,ttc_opt,ttc_subopt):
    u_opt = calc_util(ttc_opt, 'sigmoidal')
    u_subopt = calc_util(ttc_subopt, 'sigmoidal')
    p_opt,p_subopt = 0.5,0.5
    try:
        if (math.exp(param_lambda*u_opt) +  math.exp(param_lambda*u_subopt)) < math.pow(10, -10):
            p_opt,p_subopt = 0.5,0.5
            #p_opt = math.exp(param_lambda*u_opt) / (  math.exp(param_lambda*u_opt) +  math.exp(param_lambda*u_subopt) )
            #p_subopt = 1- p_opt 
        else:
            p_opt = math.exp(param_lambda*u_opt) / (  math.exp(param_lambda*u_opt) +  math.exp(param_lambda*u_subopt) )
            p_subopt = 1- p_opt
    except:
        print(param_lambda,u_opt,u_subopt,ttc_opt,ttc_subopt)
    ttc_choice = np.random.choice([ttc_opt,ttc_subopt],p=[p_opt,p_subopt])
    prob = [p_opt,p_subopt][[ttc_opt,ttc_subopt].index(ttc_choice)]
    return ttc_choice,prob
    
    
    
            
def sample_ttc_from_lambda(lambda_dict,ttc_init,vel):
    lambda_array = []
    curr_level = int(100 - round(ttc_init,1)*10)
    for l in list(lambda_dict[vel].values()):
        _temp = dict(l)
        lambda_array.append(list(_temp.values()))
    lambda_array = np.asarray(lambda_array)
    lambda_array = np.reshape(lambda_array, (51,100))
    lambda_array = lambda_array[:,curr_level:]
    ttc_list = []
    q_distr_prob = 1
    for time_indx in range(51):
        #sampled_level = np.random.randint(max(0,curr_level-2),min(curr_level+3,101))
        ttc_opt = min(round( (100 - curr_level) / 10 , 1 ) + .1,10)
        ttc_sub_opt = max(ttc_opt - 0.1 , 0 )
        shape_range = lambda_array.shape[1]
        if shape_range == 0:
            sampled_level = 0
        else:
            sampled_level = np.random.randint(0,shape_range)
        if lambda_array.shape[1] != 0:
            lambda_param = lambda_array[time_indx,sampled_level]
        else:
            lambda_param = -30000
        ttc_sample,prob = calc_ttc_from_lambda(lambda_param,ttc_opt,ttc_sub_opt)
        q_distr_prob = q_distr_prob * prob
        ttc_list.append(ttc_sample)
        curr_level = int(100 - round(ttc_sample,1)*10)
    ttc_list = running_mean(ttc_list[1:],3).tolist() + ttc_list[-3:]    
    return ttc_list,q_distr_prob
    
    
def get_p_distr_prob(trans_prob_model,ttc_arr,vel):  
    trans_prob_model = trans_prob_model[vel]
    p_distribution_prob = 1
    for ttc_pair in list(zip(ttc_arr[:-1],ttc_arr[1:])):
        row_indx = min(int(ttc_pair[0]*10),10)
        col_indx = min(int(ttc_pair[1]*10),10)
        trans_prob = trans_prob_model.predict([(row_indx,col_indx)])
        p_distribution_prob = p_distribution_prob * trans_prob
    return p_distribution_prob

def rationality_is_sampler():
    vel_kph = np.random.randint(0,120)
    vel = None
    if 0 < vel_kph < 30:
        vel = 'low_speed'
    elif 30 <= vel_kph < 80:
        vel = 'med_speed'
    elif vel_kph >= 80:
        vel = 'high_speed'
    js = open('/media/atrisha/Data/datasets/SPMD/processing_lists/lambda_dist.json')
    lambda_dict = json.load(js)
    crash_count = 0
    range_range_rate_sample = get_range_and_range_rate_distr(vel_kph)
    range = abs(range_range_rate_sample[0])
    range_rate = range_range_rate_sample[1] 
    ttc_init = min(range/abs(range_rate),10)
    range_init,range_rate_init = range,range_rate 
    time_thresh = 50
    time_ticker = 0
    ttc_init = min(range/abs(range_rate),10)
    ttc_transition_model = get_markov_chain_trainsition_matrix()
    ttc_sample_next_five_sec,q_distr_prob = sample_ttc_from_lambda(lambda_dict,ttc_init,vel)
    ttc_t_minus_1 = ttc_init
    num_trials = 0
    range_init,range_rate_init = range,range_rate
    trial_count = 0
    result_array,weight_array = [],[]
    while True:
        range_range_rate_sample = get_range_and_range_rate_distr(vel_kph)
        range = abs(range_range_rate_sample[0]) 
        range_rate = range_range_rate_sample[1]
        if range_rate >= 0:
            continue
        time_ticker = 0
        if range_rate != 0:
            ttc_init = range/-range_rate
        else:
            ttc_init = 10
        if ttc_init < 0:
            ttc_init = 10
        else:
            ttc_init = min(ttc_init,10)
        ttc_sample_next_five_sec,q_distr_prob = sample_ttc_from_lambda(lambda_dict,ttc_init,vel)
        p_distribution_prob = get_p_distr_prob(ttc_transition_model,ttc_sample_next_five_sec,vel)
        ttc_t_minus_1 = ttc_init
        range_init,range_rate_init = range,range_rate
        range_rate_arr = []
        while time_ticker < time_thresh :
            time_ticker = time_ticker + 1
            range_prime = range + (range_rate * .1)
            if range_prime < 0.001:
                crash_count = crash_count + 1
                result_array.append(1)
                print('added',p_distribution_prob,q_distr_prob)
                weight_array.append((p_distribution_prob , q_distr_prob))
                with open('/media/atrisha/Data/datasets/SPMD/processing_lists/is_results.out', 'w') as wfile:
                    str_line = str(num_trials) + ',' +str(crash_count) + '\n'
                    wfile.write(str_line)
            
                break
            ttc_t = ttc_sample_next_five_sec[time_ticker]
            range_rate = -1 * (range_prime / ttc_t) 
            range_rate_arr.append(range_rate*3.6)
            range = range_prime
            ttc_t_minus_1 = ttc_t
        #print(range_rate_arr)
        if range_prime >= 0.001:
            result_array.append(0)
        num_trials = num_trials + 1
        print(num_trials,round(range_prime,2),crash_count)
        if crash_count > 100:
            break
    intvl = st.t.interval(0.95,len(result_array)-1,loc = np.mean(result_array), scale = st.sem(result_array))
    if crash_count != 0:
        print(weight_array)
        print('crash prob is '+ str(np.mean(result_array)) + ' with conf interval of',intvl)
        print('by IS',np.mean([x[0]/x[1] for x in weight_array]))
        
        
    
            
            
        
        
monte_carlo_sampler_calc_prob()
    