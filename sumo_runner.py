'''
Created on Nov 26, 2018

@author: atrisha
'''

import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")


import traci
import csv

sumoBinary = "/usr/bin/sumo"
sumoCmd = [sumoBinary, "-c", "/media/atrisha/Data/Behavioral_models/sumo_lc/lc.sumocfg",
           "--step-length","0.1","--collision.mingap-factor","0",
           "--collision.action","none"]

load_params = ["-c", "/media/atrisha/Data/Behavioral_models/sumo_lc/lc.sumocfg",
           "--step-length","0.1","--collision.mingap-factor","0",
           "--collision.action","none"]

def test_mc_run():
    traci.start(sumoCmd)
    
    count=0
    line_num = 0
    data_file = []
    no_crash_count = 0
    crash_count = 0
    results = []
    
    out_f = open('/media/atrisha/Data/datasets/SPMD/processing_lists/temp_out','w')
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/interac_seq_data_smooth.csv','r') as csvfile:
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
                        range = float(traci.vehicle.getLanePosition('veh_1')) - float(traci.vehicle.getLanePosition('veh_0'))
                        if range <= 0.001:
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
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/sumo_mc_results.out', 'a') as wfile:
        str_line = str(no_crash_count) + ',' +str(crash_count) + '\n'
        wfile.write(str_line)
        for r in results:
            wfile.write(r)
            
            
def cmc_run():
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
    
    
    wfile = open('/media/atrisha/Data/datasets/SPMD/processing_lists/sumo_cmc_results.out', 'w')
    crash_count = 0
    no_crash_count = 0
    results = []
    min_range = np.inf
    traci.start(sumoCmd)
    for batch_id in np.arange(10000):
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
        
        
        for run_id in np.arange(1000):
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
                    range = max(traci.vehicle.getLanePosition(vehicle_ids[0]) , traci.vehicle.getLanePosition(vehicle_ids[1])) \
                            - min(traci.vehicle.getLanePosition(vehicle_ids[0]) , traci.vehicle.getLanePosition(vehicle_ids[1]))
                    if range < min_range:
                        min_range = range
                    if range <= 0.1:
                        crash_count = crash_count + 1
                        results.append((crash_count,no_crash_count))
                        break
            no_crash_count = no_crash_count + 1
            str_line = 'batch:'+str(batch_id) + ',run:'+str(run_id)+ ',no-crash:'+str(no_crash_count) + ',crash:' +str(crash_count)+',min range:'+str(min_range) + '\n'
            wfile.write(str_line)
            if run_id == 999:
                wfile.close()
                wfile = open('/media/atrisha/Data/datasets/SPMD/processing_lists/sumo_cmc_results.out', 'w')
        
    traci.close()
    print(results)
    with open('/media/atrisha/Data/datasets/SPMD/processing_lists/sumo_cmc_results.out', 'a') as wfile:
        for r in results:
            wfile.write(r)
    
    
cmc_run()          
