'''
Created on Nov 26, 2018

@author: atrisha
'''

import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt
from data_analysis import *
import itertools
import json
from simulation import simulate_run, simulate_run_simple
from simulation import flush_results
from simulated_annealing_opt import SimulatedAnnealing
from utils import flush_list

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")


import traci
import csv
import scipy.optimize

sumoBinary = "/usr/bin/sumo"
sumoCmd = [sumoBinary, "-c", "sumo_configs/lc.sumocfg",
           "--step-length","0.1","--collision.mingap-factor","0",
           "--collision.action","none"]

load_params = ["-c", "sumo_configs/lc.sumocfg",
           "--step-length","0.1","--collision.mingap-factor","0",
           "--collision.action","none"]


    
def plot_parameters(params1,params2):
    
    ttc_inv_params = params1[0]
    range_inv_params = params1[1]
    ttc_inv_params_new = params2[0]
    range_inv_params_new = params2[1]
    
    for idx in np.arange(3):
        f, (ax, ax2) = plt.subplots(1,2)
        X = np.arange(0,4,.001)
        Y = _exp(X, ttc_inv_params[idx][0], ttc_inv_params[idx][1])
        Y2 = _exp(X, ttc_inv_params_new[idx][0], ttc_inv_params_new[idx][1])
        ax.plot(X,Y)
        ax2.plot(X,Y2)
        plt.show()
    f, (ax, ax2) = plt.subplots(1,2)
    X = np.arange(0,.1,.001)
    Y = _exp(X, range_inv_params[0], range_inv_params[1])
    print(Y)
    Y2 = _exp(X, range_inv_params_new[0], range_inv_params_new[1])
    ax.plot(X,Y)
    ax2.plot(X,Y2)
    plt.show()
        
        


def test_mc_run():
    traci.start(sumoCmd)
    
    count=0
    line_num = 0
    data_file = []
    no_crash_count = 0
    crash_count = 0
    results = []
    
    out_f = open(root_path+'temp_out','w')
    with open(root_path+'interac_seq_data_smooth.csv','r') as csvfile:
        csv_reader = csv.reader(csvfile,delimiter=',')
        for row in csv_reader:
            if int(row[0]) == count:
                data_file.append(row)
            elif int(row[0]) > count:
                count = count + 1
                print('processing: '+str(line_num/1755166 * 100)+'%','crash:'+str(crash_count)+',no crash:'+str(no_crash_count),file=out_f)
                if len(data_file) >= 10:
                    print('starting run')
                    traci.load(load_params)
                    '''
                    print('set speed for 0 at',max(0,float(data_file[0][2])))
                    print('set speed for 1 at',max(0,float(data_file[0][3])))
                    print('allowed is',traci.lane.getMaxSpeed('1to2_0'))
                    '''
                    traci.vehicle.add(vehID='veh_0', routeID='route0',  depart=0,pos=0, speed=max(0,float(data_file[0][2])))
                    traci.vehicle.add(vehID='veh_1', routeID='route0',  depart=0,pos=float(data_file[0][4]), speed=max(0,float(data_file[0][3])))
                    traci.vehicle.moveTo('veh_1','1to2_0',float(data_file[0][4]))
                    traci.vehicle.setSpeedMode('veh_1',12)
                    traci.vehicle.setSpeedMode('veh_0',12)
                    for step in np.arange(len(data_file)):
                        vehicle_ids = traci.vehicle.getIDList()
                        traci.simulationStep()
                        traci.vehicle.setSpeed('veh_1',speed=max(0,float(data_file[step][3])))
                        print(len(vehicle_ids),step+1,traci.vehicle.getSpeed('veh_0'),traci.vehicle.getSpeed('veh_1'),float(traci.vehicle.getLanePosition('veh_1')) - float(traci.vehicle.getLanePosition('veh_0')))
                        range_x = float(traci.vehicle.getLanePosition('veh_1')) - float(traci.vehicle.getLanePosition('veh_0'))
                        if range_x <= 0.001:
                            crash_count = crash_count + 1
                            results.append((crash_count,no_crash_count))
                            break
                        
                    no_crash_count = no_crash_count + 1
                    
                data_file = []
                data_file.append(row)
            line_num = line_num + 1
    traci.close()
    print(no_crash_count)
    print(crash_count)
    print(results)
    with open(root_path+'sumo_mc_results.out', 'a') as wfile:
        str_line = str(no_crash_count) + ',' +str(crash_count) + '\n'
        wfile.write(str_line)
        for r in results:
            wfile.write(r)
            
            
def cmc_run(max_iters=100,N_per_iter=1000):
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    def _gauss(x,mu,sigma,A):
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
        return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]
    
    range_inv_pareto_params = 2.25843986
    def _pareto(X,param):
        return [(param*math.pow(.1,param))/(math.pow(x, param+1)) if x>=.1 else 0 for x in X]
    
    ttc_inv_5_15_pareto_params = 0.00586138
    ttc_inv_15_25_pareto_params = 0.00622818
    ttc_inv_25_35_pareto_params = 0.00707637
    def _exp(X,lambda_param,a):
        return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    
    wfile = open(root_path+'sumo_cmc_log.out', 'w')
    crash_count = 0
    no_crash_count = 0
    results = []
    min_range = np.inf
    traci.start(sumoCmd)
    X,Y = [],[]
    for batch_id in np.arange(max_iters):
        vel_choice_list = np.arange(0,60,0.1)
        vel_choice_weights = _bimodal(vel_choice_list,*vel_bimodal_params)
        sum_weights = sum(vel_choice_weights)
        vel_choice_weights = [x/sum_weights for x in vel_choice_weights]
        sample_vel_lc_mps = np.random.choice(vel_choice_list,p=vel_choice_weights,size=1000)
        
        sample_range_inverse = np.random.pareto(range_inv_pareto_params,size=1000)
        sample_ttc_inv = []
        for vel in sample_vel_lc_mps:
            if 0 <= vel <15:
                ttc_inv_pareto_params = ttc_inv_5_15_pareto_params
            elif 15 <= vel <25:
                ttc_inv_pareto_params = ttc_inv_15_25_pareto_params
            else:
                ttc_inv_pareto_params = ttc_inv_25_35_pareto_params
        
            ttc_inv_choice_list = np.arange(0,4,0.001)
            ttc_inv_choice_weights = _pareto(ttc_inv_choice_list,ttc_inv_pareto_params)
            sum_weights = sum(ttc_inv_choice_weights)
            ttc_inv_choice_weights = [x/sum_weights for x in ttc_inv_choice_weights]
            sample_ttc_inv.append(np.random.choice(ttc_inv_choice_list,p=ttc_inv_choice_weights))
        range_rate = [-x[0]/x[1] for x in zip(sample_ttc_inv,sample_range_inverse)]
        sample_vel_s_mps = [x[0] - x[1] for x in zip(sample_vel_lc_mps,range_rate)]
        count_crash_iter = 0
        for run_id in np.arange(N_per_iter):
            traci.load(load_params)
            range_x = 1/sample_range_inverse[run_id]
            speed_lc = min(55,sample_vel_s_mps[run_id])
            speed_s = min(55,sample_vel_lc_mps[run_id])
            traci.vehicle.add(vehID='veh_0', routeID='route0',  depart=0,pos=0, speed=speed_s)
            traci.vehicle.add(vehID='veh_1', routeID='route0',  depart=0,pos=range_x, speed=speed_lc)
            traci.vehicle.moveTo('veh_0','1to2_0',0)
            traci.vehicle.moveTo('veh_1','1to2_0',range_x)
            traci.vehicle.setSpeedMode('veh_1',12)
            traci.vehicle.setSpeedMode('veh_0',12)
            for step in np.arange(50):
                traci.simulationStep()
                vehicle_ids = traci.vehicle.getIDList()
                if len(vehicle_ids) > 2:
                    exit()
                if len(vehicle_ids) == 2:
                    range_x = max(traci.vehicle.getLanePosition(vehicle_ids[0]) , traci.vehicle.getLanePosition(vehicle_ids[1])) \
                            - min(traci.vehicle.getLanePosition(vehicle_ids[0]) , traci.vehicle.getLanePosition(vehicle_ids[1]))
                    if range_x < min_range:
                        min_range = range_x
                    if range_x <= 0.1:
                        count_crash_iter = count_crash_iter + 1
                        crash_count = crash_count + 1
                        break
            no_crash_count = no_crash_count + 1
            results.append(crash_count/no_crash_count)
            str_line = 'batch:'+str(batch_id) + ',run:'+str(run_id)+ ',no-crash:'+str(no_crash_count) + ',crash:' +str(crash_count)+',min range_x:'+str(min_range) + '\n'
            wfile.write(str_line)
        p_crash_iter = count_crash_iter / 1000
        X.append(batch_id)
        Y.append(p_crash_iter)
    traci.close()
    plt.plot(X,Y)
    plt.show()
    print(list(zip(X,Y)))
    res_file_name = root_path+'cmc_res_final.list'
    flush_list(res_file_name, list(zip(X,Y)))
    


def lambda_op_run(max_iters,opt_l=None):
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    def _gauss(x,mu,sigma,A):
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
        return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]
    
    
    def _pareto(X,param):
        return [(param*math.pow(.1,param))/(math.pow(x, param+1)) if x>=.1 else 0 for x in X]
    
    def _exp(X,lambda_param,a):
        return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]
    
    sample_vel_s_mps,sample_vel_s_dict = sample_vel_s(100000)
    sample_ttc_inv,sample_range_inverse = [],[]
    count = 0
    
    range_inv_pareto_params = 2.25843986
    ttc_inv_5_15_pareto_params = 0.00586138
    ttc_inv_15_25_pareto_params = 0.00622818
    ttc_inv_25_35_pareto_params = 0.00707637
    ttc_inv_choice_list = np.arange(0,4,0.001)
    range_inv_choice_list = np.arange(.007,1,.001)
    
    ttc_inv_5_15_choice_weights = _pareto(ttc_inv_choice_list,ttc_inv_5_15_pareto_params)
    ttc_inv_15_25_choice_weights = _pareto(ttc_inv_choice_list,ttc_inv_15_25_pareto_params)
    ttc_inv_25_35_choice_weights = _pareto(ttc_inv_choice_list,ttc_inv_25_35_pareto_params)
    range_inv_choice_weights = _pareto(range_inv_choice_list,range_inv_pareto_params)
    
    sum_weights_5_15 = sum(ttc_inv_5_15_choice_weights)
    sum_weights_15_25 = sum(ttc_inv_15_25_choice_weights)
    sum_weights_25_35 = sum(ttc_inv_25_35_choice_weights)
    sum_weights_range_inv = sum(range_inv_choice_weights)
    
    ttc_inv_5_15_choice_weights = [x/sum_weights_5_15 for x in ttc_inv_5_15_choice_weights]
    ttc_inv_15_25_choice_weights = [x/sum_weights_15_25 for x in ttc_inv_15_25_choice_weights]
    ttc_inv_25_35_choice_weights = [x/sum_weights_25_35 for x in ttc_inv_25_35_choice_weights]
    range_inv_choice_weights = [x/sum_weights_range_inv for x in range_inv_choice_weights]
    
    sample_range_inverse = np.random.choice(range_inv_choice_list,p=range_inv_choice_weights,size=100000)
    
    for vel in sample_vel_s_mps:
        count = count + 1
        print('progress of 100000',count)
        if 0 <= vel <15:
            sample_ttc_inv.append(np.random.choice(ttc_inv_choice_list,p=ttc_inv_5_15_choice_weights))
        elif 15 <= vel <25:
            sample_ttc_inv.append(np.random.choice(ttc_inv_choice_list,p=ttc_inv_5_15_choice_weights))
        else:
            sample_ttc_inv.append(np.random.choice(ttc_inv_choice_list,p=ttc_inv_25_35_choice_weights))
    
        
    range_rate = [x[0]/x[1] for x in zip(sample_ttc_inv,sample_range_inverse)]
    sample_vel_lc_mps = [max((x[0] - x[1]),1) if x[0] >=1 else max(x[0]-.1,.1) for x in zip(sample_vel_s_mps,range_rate)]
    all_configs = list(zip(sample_vel_s_mps,sample_vel_lc_mps,sample_range_inverse))
    
    #algo = 'simulated_annealing'
    wfile = open(root_path+'sumo_lambda_results.out', 'w')
    if os.path.exists(root_path+'lambda_opt_res_final.json'):
        os.remove(root_path+'lambda_opt_res_final.json')
    res_file_name = root_path+'lambda_opt_res_final.json'
    count = 0
    opt_lambda = None
    p_crash = 0
    traci.start(sumoCmd)
    global_crash_prob_dict = dict()
    global_index = 0
    X,Y = [],[]
    U_prime = dict()
    for state_config in all_configs:
        vel_s,vel_lc,range_x = state_config[0],state_config[1],1/state_config[2]
        print('progress of 100000',count,'(',vel_s,vel_lc,range_x,')')
        count = count + 1
        u_p = util_progress(vel_lc)
        util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
        u_ttc = util_ttc(range_x/(vel_s-vel_lc),util_ttc_param)
        util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
        u_d = util_dist(range_x,util_dist_param)
        opt_act_util = get_optimal_action(vel_s)
        print('progress of 100000',count,'(',vel_s,vel_lc,range_x,opt_act_util,')')
        u_prime_vector = [u_p - 1,u_ttc - 1,u_d - 1]
        U_prime[(vel_s,vel_lc,range_x)] = u_prime_vector
    if opt_l is None:
        status_dict = {0:{'id':'c_1','lambdas_tried':[[-10,-10,-10,-1.8]],'constraints':[(-100,0),(-100,0),(-100,0)],'crash_probability':[]},\
                       1:{'id':'la_0','lambdas_tried':[[-10,-10,10,-1.8]],'constraints':[(-100,0),(-100,0),(0,100)],'crash_probability':[]},\
                       2:{'id':'lc_0','lambdas_tried':[[-10,10,-10,-1.8]],'constraints':[(-100,0),(0,100),(-100,0)],'crash_probability':[]},\
                       3:{'id':'a_0','lambdas_tried':[[-10,10,10,-1.8]],'constraints':[(-100,0),(0,100),(0,100)],'crash_probability':[]},\
                       4:{'id':'c_0','lambdas_tried':[[10,-10,-10,-1.8]],'constraints':[(0,100),(-100,0),(-100,0)],'crash_probability':[]},\
                       5:{'id':'lc_1','lambdas_tried':[[100,-10,10,-1.8]],'constraints':[(0,100),(-100,0),(0,100)],'crash_probability':[]},\
                       6:{'id':'lc_2','lambdas_tried':[[10,10,-10,-1.8]],'constraints':[(0,100),(0,100),(-100,0)],'crash_probability':[]},\
                       7:{'id':'a_2','lambdas_tried':[[10,10,10,-1.8]],'constraints':[(0,100),(0,100),(0,100)],'crash_probability':[]}
            }
        for k,v in status_dict.items():
            styl_id = v['id']
            if len(v['lambdas_tried']) > 0:
                l = v['lambdas_tried'][0]
                p_crash_iter = simulate_run(l,U_prime,global_crash_prob_dict,wfile,opt_lambda,None)
                v['crash_probability'].append(p_crash_iter)
                if p_crash_iter > p_crash:
                    p_crash = p_crash_iter
                    opt_lambda = l
                flush_results(global_crash_prob_dict,res_file_name)
                global_crash_prob_dict = dict() 
        SA = SimulatedAnnealing(status_dict,U_prime,global_crash_prob_dict,wfile,opt_lambda,p_crash,res_file_name) 
        SA.optimize()                  
        flush_results(global_crash_prob_dict,res_file_name)   
    else:
        f_q_u_list = []
        for iter in np.arange(max_iters):
            p_crash_iter,q_u_list = simulate_run(opt_l,U_prime,global_crash_prob_dict,wfile,opt_lambda,iter,sample_vel_s_dict)
            X.append(iter)
            Y.append(p_crash_iter)
            f_q_u_list.append((iter,q_u_list))
    traci.close() 
    if opt_l is None:
        print(status_dict)
        print('DONE')  
    else:
        res_file_name = root_path+'lambda_opt_res_final_wu.list'
        plt.plot(X,Y)
        plt.show()
        print(list(zip(X,Y)))
        print(f_q_u_list)
        flush_list(res_file_name, f_q_u_list)
        res_file_name = root_path+'lambda_opt_res_final_prob.list'
        flush_list(res_file_name, list(zip(X,Y)))
                        
def sample_vel_s(num_samples):  
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    def _gauss(x,mu,sigma,A):
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
        return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]    
    
    vel_choice_list = np.arange(0.1,60,0.1)
    vel_choice_list = [round(x,1) for x in vel_choice_list]
    vel_choice_weights = _bimodal(vel_choice_list,*vel_bimodal_params)
    sum_weights = sum(vel_choice_weights)
    vel_choice_weights = [x/sum_weights for x in vel_choice_weights]
    sample_indexes = np.random.choice(np.arange(len(vel_choice_list)),p=vel_choice_weights,size=num_samples)
    samples = [vel_choice_list[i] for i in sample_indexes]
    probs = [vel_choice_weights[i] for i in sample_indexes]
    samples_dict = dict()
    for ind in np.arange(len(samples)):
        if samples[ind] in samples_dict:
            samples_dict[samples[ind]] = samples_dict[samples[ind]] + probs[ind] 
        else:
            samples_dict[samples[ind]] = probs[ind]
    return samples,samples_dict

def eval_vel_s(vel_s):
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    def _gauss(x,mu,sigma,A):
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
        return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]
    return _bimodal([vel_s],*vel_bimodal_params)[0]

    

def _pareto(X,param):
        return [(param*math.pow(.1,param))/(math.pow(x, param+1)) if x>=.1 else 0 for x in X]
    
def _exp(X,lambda_param,a):
    return [a*(1/lambda_param) * np.exp(-1*x/lambda_param) for x in X]



def sample_exp(params,num_samples,range_step_tuple):    
    choice_list = np.arange(range_step_tuple[0],range_step_tuple[1],range_step_tuple[2])
    choice_weights = _exp(choice_list,params[0],params[1])
    sum_weights = sum(choice_weights)
    ttc_inv_choice_weights = [x/sum_weights for x in choice_weights]
    return np.random.choice(choice_list,p=ttc_inv_choice_weights,size=num_samples) if num_samples > 0 else np.random.choice(choice_list,p=ttc_inv_choice_weights)
               
def cross_entr_sampling(max_iters=100,N_per_iter=1000,sim_with_opt_params = True):
    def _eval_exp(x,lambda_param,a,domain):
        try:
            x = round(x,3)
            domain = [round(v,3) for v in np.arange(domain[0],domain[1],domain[2])]
            choice_weights = [a*(1/lambda_param) * np.exp(-1*v/lambda_param) for v in domain]
            sum_weights = sum(choice_weights)
            choice_probs = [x/sum_weights for x in choice_weights]
            val_index = domain.index(x)
            return choice_probs[val_index]
        except ValueError:
            br = 5
    if not sim_with_opt_params:
        if os.path.exists(root_path+'cross_entr_res.json'):
            os.remove(root_path+'cross_entr_res.json')
    wfile = open(root_path+'sumo_ce_results.out', 'w')
    file_name = root_path+'cross_entr_res.json'
    sample_ttc_inv,sample_range_inv,sample_vel_lc_mps = [],[],[]
    count = 0
    w_ttc_inv = [[0.1238042,  0.23736926],[0.09011138, 0.31582509],[0.06177499, 0.54435407]]
    w_range_inv = [0.04528787, 0.00955973]
    
    traci.start(sumoCmd)
    v_t_minus1_ttc_inv,v_t_minus1_range_inv = list(w_ttc_inv),list(w_range_inv)
    if not sim_with_opt_params:
        v_t_ttc_inv,v_t_range_inv = [],[]
    else:
        v_t_ttc_inv = [[0.026876620533787147, 0.23736926], [5.827655346382957e-11, 0.31582509], [0.11359287806005547, 0.54435407]]
        v_t_range_inv = [1.1125586833624332, 0.00955973]
        #v_t_ttc_inv = [[4.2223424157314353e-10, 0.23736926], [2.514711568293229e-27, 0.31582509], [2.6997665752354978e-15, 0.54435407]]
        #v_t_range_inv = [2.3696684887638217, 0.00955973]
    res_dict = dict()
    res_avg = []
    X,Y = [],[]
    q_u_list = []
    all_param_vals = [v_t_minus1_ttc_inv]
    for iter in np.arange(max_iters):
        sample_vel_s_mps_all,sample_vel_s_dict = sample_vel_s(100000)
        if not sim_with_opt_params:
            sample_range_inv = sample_exp(v_t_minus1_range_inv, 100000,(.007,1,.001))
        else:
            sample_range_inv = sample_exp(v_t_range_inv, 100000,(.007,1,.001))
        all_configs = list(zip(sample_vel_s_mps_all,sample_range_inv))
        low_speed_configs,med_speed_configs,high_speed_configs = [],[],[]
        num_samples_per_range = [0,0,0]
        probs = []
        s_ct = 0
        for _val in all_configs:
            s_ct = s_ct + 1
            print('progress of 100000',s_ct)
            if 0 <= _val[0] <15:
                num_samples_per_range[0] += 1
                low_speed_configs.append((_val[0],_val[1]))
            elif 15 <= _val[0] <25:
                num_samples_per_range[1] += 1
                med_speed_configs.append((_val[0],_val[1]))
            else:
                num_samples_per_range[2] += 1
                high_speed_configs.append((_val[0],_val[1]))
        if not sim_with_opt_params:
            sample_low_speed_ttc_inv = sample_exp(v_t_minus1_ttc_inv[0], num_samples_per_range[0],(0,4,.001))
            sample_med_speed_ttc_inv = sample_exp(v_t_minus1_ttc_inv[1], num_samples_per_range[0],(0,4,.001))
            sample_high_speed_ttc_inv = sample_exp(v_t_minus1_ttc_inv[2], num_samples_per_range[0],(0,4,.001))
        else:
            sample_low_speed_ttc_inv = sample_exp(v_t_ttc_inv[0], num_samples_per_range[0],(0,4,.001))
            sample_med_speed_ttc_inv = sample_exp(v_t_ttc_inv[1], num_samples_per_range[0],(0,4,.001))
            sample_high_speed_ttc_inv = sample_exp(v_t_ttc_inv[2], num_samples_per_range[0],(0,4,.001))
        low_speed_configs = [(x[0][0],x[1],x[0][1]) for x in list(zip(low_speed_configs, sample_low_speed_ttc_inv))]
        med_speed_configs = [(x[0][0],x[1],x[0][1]) for x in list(zip(med_speed_configs, sample_med_speed_ttc_inv))]
        high_speed_configs = [(x[0][0],x[1],x[0][1]) for x in list(zip(high_speed_configs, sample_high_speed_ttc_inv))]
        all_configs_by_vel = [low_speed_configs,med_speed_configs,high_speed_configs]
        total_crash_ct,total_ct = 0,0
        iter_crash_count = 0
        q_u_list_iter = []
        
        if not sim_with_opt_params:
            v_t_ttc_inv,v_t_range_inv = list(v_t_minus1_ttc_inv),list(v_t_minus1_range_inv)
        for idx,all_configs_set in enumerate(all_configs_by_vel):
            crash_count,no_crash_count = 0,0
            n_all_configs = len(all_configs_set)
            all_configs_indexes = np.random.choice(np.arange(len(all_configs_set)),size=len(all_configs_set)//100)
            all_configs_iter = [all_configs_set[x] for x in all_configs_indexes]
            n_all_configs = len(all_configs_iter)
            _val_n_range_inv,_val_d_range_inv,_val_n_ttc_inv,_val_d_ttc_inv = 0,0,0,0
            state_ctr = 0
            for state_config in all_configs_iter:
                speed_s = min(55,state_config[0])
                range_x = 1/state_config[2]
                range_rate = range_x / state_config[1]
                speed_lc =  max((speed_s - range_rate),1) if speed_s >=1 else max(speed_s-.1,.1)
                speed_lc = min(55,speed_lc)
                state_ctr = state_ctr + 1
                has_crashed = simulate_run_simple(speed_s,speed_lc,range_x,wfile,((iter,idx),str(state_ctr)+'/'+str(n_all_configs)),crash_count,no_crash_count)
                if len(all_param_vals) > 1:
                    curr_diff = []
                    for i in np.arange(3):
                        curr_diff.append(abs(all_param_vals[-1][i][0] - all_param_vals[-2][i][0]))
                    wfile.write(str(curr_diff)+'\n')
                else:
                    wfile.write('\n')
                    
                    
                if has_crashed:
                    crash_count = crash_count + 1
                    iter_crash_count = iter_crash_count + 1
                    ttc_inv =  state_config[1]
                    if sim_with_opt_params:
                        p_u = sample_vel_s_dict[speed_s] * _eval_exp(1/range_x, w_range_inv[0], w_range_inv[1], (.007,1,.001)) * _eval_exp(ttc_inv,w_ttc_inv[idx][0],w_ttc_inv[idx][1],(0,4,.001))
                        q_u = sample_vel_s_dict[speed_s] * _eval_exp(1/range_x,v_t_range_inv[0],v_t_range_inv[1],(.007,1,.001)) * _eval_exp(ttc_inv,v_t_ttc_inv[idx][0],v_t_ttc_inv[idx][1],(0,4,.001))
                        q_u_list_iter.append((q_u,p_u))
                    if not sim_with_opt_params:
                        _l_n = _exp([ttc_inv],w_ttc_inv[idx][0],w_ttc_inv[idx][1])[0] * \
                                _exp([range_x],w_range_inv[0],w_range_inv[1])[0]
                        _l_d = _exp([ttc_inv],v_t_minus1_ttc_inv[idx][0],v_t_minus1_ttc_inv[idx][1])[0] * \
                                _exp([range_x],v_t_minus1_range_inv[0],v_t_minus1_range_inv[1])[0]
                        _l = _l_n / _l_d
                        ''' Based on Rubenstein's formulation.'''
                        _val_n_range_inv = _val_n_range_inv + _l*range_x
                        _val_d_range_inv = _val_d_range_inv + _l
                        _val_n_ttc_inv = _val_n_ttc_inv + _l*ttc_inv
                        _val_d_ttc_inv = _val_d_ttc_inv + _l
                        
                else:
                    no_crash_count = no_crash_count + 1
            if not sim_with_opt_params:
                if _val_d_range_inv != 0:
                    updated_param = _val_n_range_inv/_val_d_range_inv
                    updated_param = updated_param if (not math.isnan(updated_param) and updated_param!=0) else v_t_minus1_range_inv[0]
                    v_t_range_inv = list([updated_param,w_range_inv[1]])
                if _val_d_ttc_inv != 0:
                    updated_param = _val_n_ttc_inv/_val_d_ttc_inv
                    updated_param = updated_param if (not math.isnan(updated_param)  and updated_param!=0) else v_t_minus1_ttc_inv[idx][0]
                    v_t_ttc_inv[idx] = list([updated_param, w_ttc_inv[idx][1]])
            total_crash_ct = crash_count + total_crash_ct  
            total_ct = total_ct + n_all_configs
        res_avg.append(total_crash_ct/total_ct)
        if sim_with_opt_params:
            q_u_list.append((iter,q_u_list_iter))
        p_crash_iter = iter_crash_count/total_ct
        X.append(iter)
        Y.append(p_crash_iter)
        if not sim_with_opt_params:
            
            converged = True
            if iter > 10:
                for i in np.arange(3):
                    converged = converged and abs(v_t_ttc_inv[i][0] - v_t_minus1_ttc_inv[i][0]) < 0.001
                converged = converged and abs(v_t_range_inv[0] - v_t_minus1_range_inv[0]) < 0.001
                if converged:
                    print('diffs')
                    for i in np.arange(3):
                        print(abs(v_t_ttc_inv[i][0] - v_t_minus1_ttc_inv[i][0]))
                    print(abs(v_t_range_inv[0] - v_t_minus1_range_inv[0]))
                    print('exiting after iteration', iter)
                    break
            
            all_param_vals.append(v_t_ttc_inv)
            v_t_minus1_range_inv = list(v_t_range_inv)
            v_t_minus1_ttc_inv = list(v_t_ttc_inv)
            res_dict[str(iter)] = [(list(v_t_ttc_inv),list(v_t_range_inv),total_crash_ct/total_ct)]
            
    traci.close() 
    if not sim_with_opt_params:
        print(v_t_ttc_inv)
        print('-------')
        print(v_t_range_inv)
        print('-------')
        print(res_avg)
        plt.plot(np.arange(len(res_avg)),res_avg)
        plt.show()  
        flush_results(res_dict,file_name)
    else:
        res_file_name = root_path+'ce_opt_res_final_wu.list'
        plt.plot(X,Y)
        plt.show()
        print(list(zip(X,Y)))
        print(q_u_list)
        flush_list(res_file_name, q_u_list)
        res_file_name = root_path+'ce_opt_res_final_prob.list'
        flush_list(res_file_name, list(zip(X,Y)))
        
        
    

def run_with_opt_vals():
    opt_l = [-6, -71, 6,-4.1]
    max_iters = 10
    N_per_iter = 1000
    lambda_op_run(max_iters,opt_l)
    #cross_entr_sampling(max_iters=max_iters)
    #cmc_run()
    
def run_opt_scheme():
    #lambda_op_run()
    cross_entr_sampling(max_iters=100,sim_with_opt_params=False)
        
        
''' all runs below '''
    
run_with_opt_vals()

'''
w_ttc_inv = [[0.1238042,  0.23736926],[0.09011138, 0.31582509],[0.06177499, 0.54435407]]
w_range_inv = [0.04528787, 0.00955973]

v_t_ttc_inv = [[0.07338749587482443, 0.23736926], [0.03132487936214528, 0.31582509], [0.18098219010007013, 0.54435407]]
v_t_range_inv = [2.378342605717401, 0.00955973]


plot_parameters((w_ttc_inv,w_range_inv),(v_t_ttc_inv,v_t_range_inv))
'''


