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


def util_progress(x,scale=.2,thresh=5):
    return np.tanh(scale*(x-thresh))

def util_reaction(time_sec):
    return np.cos(np.radians(60*time_sec))

def util_dist(dist_m,thresh):
    x = dist_m
    return 1.5/(1+math.exp(-1 * (x-thresh))) - 0.5
    
def util_ttc(ttc_sec,thresh):
    if ttc_sec > 100:
        return 1
    x = ttc_sec
    return 1.5/(1+math.exp(-1 * (x-thresh))) - 0.5
    
                
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
        
        
