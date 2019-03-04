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


root_path = 'processing_lists/'
data_path = '.' 
                
                
            
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
            f.write("%s\n" % item)
