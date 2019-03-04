'''
Created on Feb 15, 2019

@author: atrisha
'''

import csv
import numpy as np
from sumo_runner import get_optimal_action, util_progress, util_dist, util_ttc
import matplotlib.pyplot as plt
from scipy.optimize import *
import statsmodels.api as sm
import scipy.stats as stats
from data_analysis import final_results_plot
from utils import root_path


def get_quantile(x,quants):
    for idx,q in enumerate(list(zip(quants[:-1],quants[1:]))):
        if q[0] < x < q[1] :
            return idx
    return len(quants)-1
        
def _p_2(U,l1,l2,a):
        return [(a*(l1*np.exp(l1*u)) / (np.exp(2*l1) - 1)) + \
                 ((1-a)*(l2*np.exp(l2*u)) / (np.exp(2*l2) - 1)) for u in U] 
    
                  
def calc_mle_est(s,n,l):
    return True

def generate_sample_data(N,ttc_choices):
    l_1_options = np.arange(-100,100,10)
    l_2_options = np.arange(-100,100,10)
    a_options = np.arange(0.01,1,.1)
    import itertools
    all_options = list(itertools.product(l_1_options,l_2_options,a_options))
    res_array = np.ndarray(shape=(len(all_options),N))
    tot_len = len(all_options)
    for idx,option in enumerate(all_options):
        print(idx,'/',tot_len)
        weights = _p_2([util_ttc(x, thresh=2)+1 for x in ttc_choices],*option)
        _sum = sum(weights)
        weights = [x/_sum for x in weights]
        gen_samples = np.random.choice(ttc_choices,size = N, p=weights)
        res_array[idx,:] = gen_samples
    return res_array

def custom_opt(data,obs,bins):
    from numpy import linalg as LA
    import math
    hist = np.histogram(obs,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = [y/100 for y in hist[0]] 
    res = np.zeros(shape=(len(data)+1,len(X)))
    res[0,:] = Y
    for i in np.arange(len(data)):
        hist = np.histogram(data[i,:],bins = bins,density=True)
        X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
        Y = [y/100 for y in hist[0]]
        res[i+1,:] = Y
        if np.any(np.isnan(Y)):
            g=8
            print('NA value',i)
    sq_err = dict()
    for i in np.arange(1,res.shape[0]):
        sq_err[i] = LA.norm(res[i,:] - res[0,:])
        if math.isnan(sq_err[i]):
            g = res[i,:] - res[0,:]
            f = 6
    return res
    
         

def split_cube_plot_pie(pt):
    edge_1 = pt[0]
    edge_2 = pt[1]
    edge_3 = 1-pt[0]
    edge_4 = 1 - pt[1]
    edge_5 = 1 - pt[2]
    edge_6 = pt[2]
    cube = [None]*8
    cube[0] = edge_3 * edge_4 * edge_6
    cube[1] = edge_3 * edge_2 * edge_6
    cube[2] = edge_1 * edge_2 * edge_5
    cube[3] = edge_1 * edge_4 * edge_5
    cube[4] = edge_3 * edge_4 * edge_5
    cube[5] = edge_3 * edge_2 * edge_5
    cube[6] = edge_1 * edge_2 * edge_6
    cube[7] = edge_1 * edge_4 * edge_6
    cube = [round(x,3) for x in cube]
    print(cube,':',sum(cube))
    labels = ['B1','B2','B3','B4','B5','B6','B7','B8']
    data_list = [x[1] for x in enumerate(list(zip(cube,labels))) if x[1][0] != 0]
    fig1, ax1 = plt.subplots()
    sizes = [x[0] for x in data_list]
    labels = [x[1] for x in data_list]
    cmap = plt.get_cmap("Set3")
    p_colors = cmap(np.arange(len(sizes)))
    ax1.pie(sizes,labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90,colors=p_colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
        
    

def mle_est():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
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
            if float(row[6]) < 0:
                range_inv = 1/float(row[5])
                ttc = min(float(row[5]) / -float(row[6]),10)
                if row[0]+'-'+row[1]+'-'+row[2] in wsu_dict.keys():
                    vel_mps = int(round(float(wsu_dict[row[0]+'-'+row[1]+'-'+row[2]][0][16]) / 3.6))
                    if 5 <= vel_mps < 15:
                        vel_5_15.append((vel_mps,range_inv,ttc))
                    elif 15 <= vel_mps < 25:
                        vel_15_25.append((vel_mps,range_inv,ttc))
                    elif 25 <= vel_mps < 35:
                        vel_25_35.append((vel_mps,range_inv,ttc))
    state_samples = vel_5_15
    N = len(state_samples)
    train_indices = np.random.choice(np.arange(N),size = int(np.floor(.7*N)),replace=False).tolist()
    train_indices.sort()
    train_samples = [state_samples[i] for i in train_indices]
    test_indices = []
    for i in np.arange(N):
        print('processing',i,N)
        if i not in train_indices:
            test_indices.append(i)
    #test_indices = [i for i in np.arange(N) if i not in train_indices]
    test_samples = [state_samples[i] for i in test_indices]
    N_train = len(train_samples)
    sigma_p,sigma_d,sigma_t = 0,0,0
    train_sample_u_ttcs,train_sample_u_range = [],[]
    vel_util_lc_train = []
    for state_t in train_samples:
        vel_s = state_t[0]
        ttc = state_t[2]
        range_inv = state_t[1]
        range_x = 1/range_inv
        range_rate = range_x/ttc
        vel_lc = max(0.01,vel_s - range_rate)
        opt_act_util = get_optimal_action(vel_s)
        util_progress_param = 5 if 0 <= vel_s <15 else 20 if 15 <= vel_s <25 else 30
        u_p = util_progress(vel_lc,thresh=util_progress_param)
        vel_util_lc_train.append((vel_lc,u_p))
        util_ttc_param = 2 if 0 <= vel_s <15 else 4 if 15 <= vel_s <25 else 3.5  
        u_ttc = util_ttc(ttc)
        train_sample_u_ttcs.append(u_ttc + 1)
        util_dist_param = 10 if 0 <= vel_s <15 else 50 if 15 <= vel_s <25 else 100 
        u_d = util_dist(range_x,thresh=util_dist_param)
        train_sample_u_range.append(u_d + 1)
        u_prime_vector = [u_p+1,u_ttc+1,u_d+1] 
        sigma_p = sigma_p + u_prime_vector[0]
        sigma_t = sigma_t + u_prime_vector[1]
        sigma_d = sigma_d + u_prime_vector[2]
    ttc_vals_in_test = [x[2] for x in test_samples]
    range_vals_in_test = [1/x[1] for x in test_samples]
    vel_lc_in_test = [x[0] - (x[2]/x[1]) for x in test_samples]
    train_range_vals = [1/x[1] for x in train_samples]
    ttc_vals_in_train = [x[2] for x in train_samples]
    n_ttc_test = len(ttc_vals_in_test)
    def _p(U,l):
        return [(l*np.exp(l*u)) / (np.exp(2*l) - 1) for u in U] 
    
    
    bins = np.arange(0,2.01,.01)
    hist = np.histogram(train_sample_u_ttcs,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    _sum = sum(hist[0])
    Y = hist[0]
    #plt.plot(X,Y,'.')
    
    
    
    popt, pcov = curve_fit(_p_2, X[:-5],Y[:-5],bounds=[(0,-100,0),(100,0,1)],p0=[1,-2,.7])
    print('ttc',popt)
    pop_fit = popt
    at = popt[2]
    popt = (1,2,.7)
    #popt = (1,5,.6)
    
    
    ttc_choices = np.arange(min(ttc_vals_in_test),max(ttc_vals_in_test),.1)
    #gen_data = generate_sample_data(n_ttc_test,ttc_choices)
    #custom_opt(gen_data,train_sample_u_ttcs,bins)
    
    #plot_mix_exp(lt_1, lt_2, at,plt,bins)
    #plt.title('ttc fit')
    #plt.ylim((0,4))
    #plt.show()
    #plt.plot(X, _p(X, *popt))
    #axs[2].plot(X, _p_2(X, *popt),color='red')
    #axs[2].plot(X, _p_2(X, *pop_fit),color='black')
    #axs[2].plot(X, _p_2(X, lt_1,lt_2,at),color='red')
    bins = np.arange(0,10,.1)
    hist = np.histogram(ttc_vals_in_test,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    plt.plot(X,Y,'x')
    #plt.plot(X,Y,'x')
    '''
    hist = np.histogram(ttc_vals_in_train,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    axs[0].plot(X,Y,'o','b')
    '''
    
    #ttc_choices = np.arange(min(ttc_vals_in_test),max(ttc_vals_in_test),.1)
    ttc_choices = np.arange(0,10,.0001)
    #ttc_choices = np.arange(.1,10,.5)
    
    weights = _p_2([util_ttc(x,thresh=.1)+1 for x in ttc_choices],*pop_fit)
    _sum = sum(weights)
    weights = [x/_sum for x in weights]
    gen_samples = np.random.choice(ttc_choices,size = n_ttc_test, p=weights)
    hist = np.histogram(gen_samples,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    plt.plot(X,Y,'.')
    #axs[0].plot(ttc_choices,_p_2([util_ttc(x, thresh=2)+1 for x in ttc_choices],*pop_fit),color='red')
    #axs[0].plot(ttc_choices,_p_2([util_ttc(x, thresh=2)+1 for x in ttc_choices],*popt),color='blue')
    #plt.plot(X,Y,'.')
    #plt.xlim(0,6)
    plt.xlabel('ttc (s)')
    plt.ylabel('likelihood')
    #plt.show()
    '''
    x,y = [],[]
    for u in ttc_choices:
        x.append(u)
        y.append(util_ttc(u,thresh=.1))
    axs[1].plot(x,y)
    '''
    plt.show()
    
    
    
    ''' range plotting start'''
    
    
    bins = np.arange(0,2.1,.1)
    hist = np.histogram(train_sample_u_range,bins = bins,density=True)
    print('min and max range u',min(train_sample_u_range),max(train_sample_u_range))
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    
    #plt.plot(X,Y,'.')
    
    popt, pcov = curve_fit(_p_2, X,Y,bounds=[(0,-100,0),(100,0,1)],p0=[5,-5,.5])
    ld_1,ld_2,ad = popt[0],popt[1],popt[2]
    print('popt range',ld_1,ld_2,ad)
    #plot_mix_exp(ld_1, ld_2, ad,plt,bins)
    #plt.title('range fit')
    #plt.ylim((0,10))
    #plt.show()
    
    
    bins = np.arange(0,50)
    hist = np.histogram(range_vals_in_test,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    plt.plot(X,Y,'x')
    
    
    
    
    n_range_test = len(range_vals_in_test)
    range_choices = np.arange(min(range_vals_in_test),max(range_vals_in_test)+1)
    weights = _p_2([util_dist(x, thresh=1)+1 for x in range_choices],ld_1,ld_2,ad)
    _sum = sum(weights)
    weights = [x/_sum for x in weights]
    gen_samples = np.random.choice(range_choices,size = n_range_test*2, p=weights)
    hist = np.histogram(gen_samples,bins = bins,density=True)
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    
    plt.plot(X,Y,'.')
    
    plt.xlabel('range (meters)')
    plt.ylabel('likelihood')
    
    plt.show()
    
    # range plotting end 
    
    #vel plotting starts
    
    bins = np.arange(0,2.1,.1)
    vel_lc_train = [x[0] for x in vel_util_lc_train]
    vel_lc_util_train = [x[1] for x in vel_util_lc_train]
    hist = np.histogram(vel_lc_util_train,bins = bins,density=True)
    print('min and max vel',min(vel_lc_train),max(vel_lc_train))
    X = [(x[0] + x[1])/2 for x in zip(bins[:-1],bins[1:])]
    Y = hist[0]
    
    #plt.plot(X,Y,'.')
    
    popt, pcov = curve_fit(_p_2, X,Y,bounds=[(0,-100,0),(100,0,1)],p0=[5,-5,.5])
    lv_1,lv_2,av = popt[0],popt[1],popt[2]
    print('popt vel',lv_1,lv_2,av)
    
    
    
    # vel plotting ends
    # plot pie chart begin
    
    split_cube_plot_pie((ad,at,av))
    
    # plot pie chart ends
    
    
    
    
    
    from scipy.stats.stats import pearsonr
    ttc_choices = np.arange(min(ttc_vals_in_test),max(ttc_vals_in_test),.1)
    #pop = np.arange(min(util_vals_in_test),max(util_vals_in_test)+.1,.1)
    weights = _p_2([util_ttc(x, thresh=2)+1 for x in ttc_choices],*pop_fit)
    _sum = sum(weights)
    weights = [x/_sum for x in weights]
    choices = np.random.choice(ttc_choices,size = n_ttc_test, p=weights)
    choices.sort()
    res_x = np.percentile(choices,q=np.arange(.1,100.1,.1))
    res_y = np.percentile(ttc_vals_in_test,q=np.arange(.1,100.1,.1))
    plt.plot(res_x,res_y,'bo')
    cor_coeff = pearsonr(res_x, res_y)
    print('rho ttc',cor_coeff)
    plt.plot(res_x,1.71*res_x-4,'r')
    plt.xlabel('theoretical quantiles')
    plt.ylabel('data quantile')
    
    plt.show()
    
    range_choices = np.arange(min(range_vals_in_test),max(range_vals_in_test),.1)
    #pop = np.arange(min(util_vals_in_test),max(util_vals_in_test)+.1,.1)
    weights = _p_2([util_dist(x, thresh=5)+1 for x in range_choices],ld_1,ld_2,ad)
    _sum = sum(weights)
    weights = [x/_sum for x in weights]
    choices = np.random.choice(range_choices,size = n_range_test, p=weights)
    
    choices.sort()
    res_x = np.percentile(choices,q=np.arange(.1,100.1,.1))
    res_y = np.percentile(range_vals_in_test,q=np.arange(.1,100.1,.1))
    plt.plot(res_x,res_y,'bo')
    cor_coeff = pearsonr(res_x, res_y)
    print('rho range',cor_coeff)
    plt.plot(res_x,[.75*x for x in res_x],'r')
    plt.xlabel('theoretical quantiles')
    plt.ylabel('data quantile')
    
    plt.show()
    

def plot_mix_exp(util_r,l1,l2,a):
    #a,l1,l2 = .5,-.5,10
    def _p_2(U,l1,l2,a):
        return [(a*(l1*np.exp(l1*u)) / (np.exp(2*l1) - 1)) + \
                 ((1-a)*(l2*np.exp(l2*u)) / (np.exp(2*l2) - 1)) for u in U] 
    x = util_r
    y = _p_2(x,l1,l2,a)
    plt.xlim((0,2))
    plt.plot(x,y)
        
    
    
    




''' all runs below '''

mle_est()

