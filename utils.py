'''
Created on Jul 4, 2018

@author: Atrisha
'''

import sqlite3
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from prompt_toolkit.key_binding.bindings.named_commands import complete
from math import inf
from geopy.geocoders import Nominatim
import time
import re

root_path = 'processing_lists/'
data_path = '/media/atrisha/Data/datasets/SPMD/' 
                
                
            
def eval_vel_s(vel_s):
    vel_bimodal_params = (16.47087736,7.401838,-18.54877962,16.4562847,-7.41718461,18.56954167)
    def _gauss(x,mu,sigma,A):
        return A*math.exp(-(x-mu)**2/2/sigma**2)

    def _bimodal(X,mu1,sigma1,A1,mu2,sigma2,A2):
        return [_gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2) for x in X]
    return _bimodal([vel_s],*vel_bimodal_params)[0]

def eval_exp(x,lambda_param,a):
        return a*(1/lambda_param) * np.exp(-1*x/lambda_param)


def util_progress(x,scale=.2,thresh=8):
    return np.tanh(5*(x-5)) if x < 100 else np.tanh(.5*(105-x))

def util_reaction(time_sec):
    return np.cos(np.radians(60*time_sec))

def util_dist(dist_m,thresh):
    x = dist_m
    #return 1.5/(1+math.exp(-1 * (x-0))) - 0.5 -1 
    return np.tanh(.8*(x-(thresh-7)))

def util_ttc(ttc_sec,thresh):
    if ttc_sec > 100:
        return 1
    x = ttc_sec
    #return 1.5/(.72+math.exp(-2 * (x-thresh))) - 1.03 -1
    return np.tanh(1.5*(x-thresh))

def plot_utils():
    import matplotlib.pyplot as plt
    ttc_vals = ttc_inv_l = [round(x,2) for x in np.arange(0,4,.001)]
    dist_vals = [round(x,2) for x in np.arange(0,10,.1)]
    p_vals = [round(x,1) for x in np.arange(0.1,120,0.1)]
    
    
    #util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
    #util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
    print(util_ttc(0,2),util_ttc(4,2))
    y_ttc_vals = [util_ttc(ttc_sec, 2) for ttc_sec in ttc_vals]
    y_dist_vals = [util_dist(dist_m, 10) for dist_m in dist_vals]
    y_p_vals = [util_progress(x) for x in p_vals]
    
    plt.plot(ttc_vals,y_ttc_vals)
    plt.show()
    print(util_dist(0,10),util_dist(10,10))
    plt.plot(dist_vals,y_dist_vals)
    plt.show()
    
    print(util_progress(0),util_progress(120))
    plt.plot(p_vals,y_p_vals)
    plt.show()
    
#plot_utils()    


def get_br_prob_dict(lp):
    import os
    import pickle
    if os.path.exists(root_path+'br_prob.dict'):
        new_dict = dict()
        print('acessing dict start')
        pickle_in = open(root_path+'br_prob.dict',"rb")
        br_prob_dict = pickle.load(pickle_in)
        print('acessing dict end')
        return br_prob_dict
    else:
        speed_l = [round(x,1) for x in np.arange(0.1,60,0.1)]
        ''' we would adjust the final probability by multiplying by 100, since we 
        are approximating the ttc_inv and range_inv to 2 decimal places.'''
        ttc_inv_l = [round(x,2) for x in np.arange(0,4,.001)]
        range_inv_l = [round(x,2) for x in np.arange(.007,1,.001)]
        #confs = list(itertools.product(speed_l,ttc_inv_l,range_inv_l))
        br_prob_dict = dict()
        tot = len(speed_l)*len(range_inv_l)*len(ttc_inv_l)
        ct = 0
        sum1,sum2,sum3 = 0,None,None
        for sp in speed_l:
            iter_u_p = 0
            for ti in ttc_inv_l:
                for ri in range_inv_l:
                    ct += 1
                    
                    c = (sp,ti,ri)
                    vel_s = c[0]
                    range_rate = c[1]/c[2]
                    range_x = 1/c[2]
                    vel_lc = max((vel_s - range_rate),1) if vel_s >=1 else max(vel_s-.1,.1)
                    
                    u_p = util_progress(vel_lc)
                    util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
                    u_ttc = util_ttc(range_x/(vel_s-vel_lc),util_ttc_param)
                    util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
                    u_d = util_dist(range_x,util_dist_param)
                    ps = []
            
                    l=lp[0]
                    y_max = (l*np.exp(l*-2)) / (np.exp(2*l) -1)
                    y = (l*np.exp(l*u_p)) / (np.exp(2*l) -1)
                    ps.append(y/y_max)
                    l=lp[1]
                    y_max = (l*np.exp(l*-2)) / (np.exp(2*l) -1)
                    y = (l*np.exp(l*-u_ttc)) / (np.exp(2*l) -1)
                    ps.append(y/y_max)
                    l=lp[2]
                    y_max = (l*np.exp(l*0)) / (np.exp(2*l) -1)
                    y = (l*np.exp(l*u_d)) / (np.exp(2*l) -1)
                    ps.append(y_max/y)
                    br_prob_dict[c] = ps
                    p_f = sum(ps)/3
                    print(p_f,ct/tot*100)
                    sum1 = sum1 + p_f
                    #y = [((l*np.exp(l*v)) / (np.exp(2*l) -1) )/y_max for v in x]
        ct = 0
        tot = len(br_prob_dict)
        for k, v in br_prob_dict.items():
            ct += 1
            br_prob_dict[k] = v/sum1
            print(ct/tot)
        pickle_out = open(root_path+'br_prob.dict',"wb")
        pickle.dump(br_prob_dict, pickle_out)
        pickle_out.close()
        return br_prob_dict      

#get_br_prob_dict((-6,-71,6))

          
def flush_list(res_file_name,l):
    with open(res_file_name, 'w') as f:
        for item in l:
            if isinstance(item, tuple):
                f.write(''.join('[(%s, %s)]' % item))
            else:    
                f.write('%s' % item)
            
def _is_highway(addr_str):
    x = re.search('[M|I]\s[0-9]+',addr_str)
    if x is None:
        return False
    else:
        return True


def _is_street(addr_string):
        street_suffixes = ['Road','Street','Drive','Township','Avenue','Court','Lane']
        for s in street_suffixes:
            if addr_string.endswith(s):
                return True
        if _is_highway(addr_string):
            return True
        return False
    

query_cache = dict()
def query_nomanatim(pt):
    global query_cache
    if pt[0] not in query_cache:
        time.sleep(1)
        geolocator = Nominatim(user_agent="a9sarkar@gsd.uwaterloo.ca")
        location = geolocator.reverse(str(pt[1][0])+','+str(pt[1][1]))
        split_address = [addr.strip() for addr in re.split(',|;',location.address)]
        #print(split_address)
        street_idx = [idx for idx in np.arange(len(split_address)) if _is_street(split_address[idx])] 
        print(pt[0],[split_address[i] for i in street_idx])
        query_cache[pt[0]] = split_address[street_idx[0]]
        return split_address[street_idx[0]]
    else:
        return query_cache[pt[0]]

def detect_intersections(traj,inters,l,h):
    #query_nomanatim(traj[5531])
    print(' ',l,h,inters)
    m = l + ((h-l) // 2)
    if (h==l+1):
        l_ad = query_nomanatim(traj[l])
        h_ad = query_nomanatim(traj[h])
        if l_ad is not h_ad:
            inters.append((traj[h][0],h_ad))
        return inters
    if query_nomanatim(traj[m]) == query_nomanatim(traj[l]):
        return detect_intersections(traj,inters,m,h)
    elif query_nomanatim(traj[m]) == query_nomanatim(traj[h]):
        return detect_intersections(traj,inters,l,m)
    else: 
        detect_intersections(traj,inters,l,m)
        detect_intersections(traj,inters,m,h)
        return inters
        
        
