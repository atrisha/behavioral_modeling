'''
Created on Jan 8, 2019

@author: atrisha
'''
import numpy as np
import math
from simulation import simulate_run
from simulation import flush_results
import itertools
import numpy.random as nr

class SimulatedAnnealing(object):
    '''
    classdocs
    '''


    def __init__(self, status_dict,U_prime,global_crash_prob_dict,wfile,opt_lambda,p_crash,res_file_name):
        '''
        Constructor
        '''
        self.status_dict = status_dict
        self.U_prime = U_prime
        self.global_crash_prob_dict = global_crash_prob_dict
        self.wfile = wfile
        self.opt_lambda = opt_lambda
        self.p_crash = p_crash
        self.res_file_name = res_file_name
        
    def prob_accept(self,E,E_prime,T):
        if E_prime > E:
            return 1
        else:
            return math.exp(-(E - E_prime)/T)
    
    def pick_lambdas(self,s_curr,T):
        lambdas_tried = self.status_dict[s_curr]['lambdas_tried']
        crash_probs = self.status_dict[s_curr]['crash_probability']
        max_prob = max(crash_probs)
        max_lambda = lambdas_tried[crash_probs.index(max_prob)]
        
        
        list_1 = list(np.arange(self.status_dict[s_curr]['constraints'][0][0],\
                            self.status_dict[s_curr]['constraints'][0][1],10))
        list_2 = list(np.arange(self.status_dict[s_curr]['constraints'][1][0],\
                            self.status_dict[s_curr]['constraints'][1][1],10))
        list_3 = list(np.arange(self.status_dict[s_curr]['constraints'][2][0],\
                            self.status_dict[s_curr]['constraints'][2][1],10))
        list_4 = [round(x,1) for x in np.arange(-4,.5,.5)]
        all_lambdas = list(itertools.product(list_1,list_2,list_3,list_4))
        all_lambdas = [[x[0],x[1],x[2],x[3]] for x in all_lambdas]
        #max_lambda_index = all_lambdas.index(max_lambda)
        selected_index = np.random.choice(np.arange(len(all_lambdas)))
        selected_value = all_lambdas[selected_index]
        while selected_value == max_lambda:
            selected_index = np.random.choice(np.arange(len(all_lambdas)))
            selected_value = all_lambdas[selected_index]
        
        E_prime = 0
        if selected_value in lambdas_tried:
            indx = lambdas_tried.index(selected_value)
            E_prime = crash_probs[indx]
        
        accept_prob = self.prob_accept(max_prob,E_prime,T)
        accept = True 
        if np.random.random_sample() < accept_prob:
            accept = True
        if not accept:
            selected_value = max_lambda
        
        lambda_0,lambda_1,lambda_2,lambda_3 = selected_value[0],selected_value[1],selected_value[2],selected_value[3]
        chosen_lambda = [nr.randint(lambda_0-10,lambda_0+10),\
                         nr.randint(lambda_1-10,lambda_1+10),\
                         nr.randint(lambda_2-10,lambda_2+10),\
                         lambda_3-1+round(nr.random_sample(),1)]
        while chosen_lambda in lambdas_tried:
            chosen_lambda = [nr.randint(lambda_0-10,lambda_0+10),\
                         nr.randint(lambda_1-10,lambda_1+10),\
                         nr.randint(lambda_2-10,lambda_2+10),\
                         lambda_3-1+round(nr.random_sample(),1)]
        
        return chosen_lambda
            
        
    
    
    def optimize(self):
        num_styles = len(self.status_dict.keys())
        p1 = 0.7
        p5 = 0.001
        T_l1 = -1/math.log(p1)
        t5 = -1/math.log(p5)
        n=20
        ctr = 0
        frac = (t5/T_l1)**(1/(n-1))
        max_outer = 20
        max_inner = 5
        for outer_iter in np.arange(max_outer):
            crash_probs = np.zeros(shape=(num_styles,))
            for k,v in self.status_dict.items():
                indx = int(k)
                crash_probs[k,] = min(v['crash_probability'])
            
            max_index = crash_probs.argmax()
            E = crash_probs.max()
            crash_probs[max_index,] = 0
            
            _sum = np.sum(crash_probs)
            crash_probs_normalized = np.divide(crash_probs,_sum)
            accept = False
            if outer_iter < 10:
                chosen_indx = np.random.choice(np.arange(num_styles))
                accept = True
            else:
                chosen_indx = np.random.choice(np.arange(num_styles),p=crash_probs_normalized)
            E_prime = crash_probs[chosen_indx,]
            
            if np.random.random_sample() < self.prob_accept(E,E_prime,T_l1):
                accept = True
            s_curr = chosen_indx if accept else max_index
            T_l2 = -1/math.log(p1)
            for inner_iter in np.arange(max_inner):
                ctr = ctr + 1
                print(ctr,'/25')
                next_lambda = self.pick_lambdas(s_curr,T_l2)
                if str(next_lambda) not in self.global_crash_prob_dict:
                    self.global_crash_prob_dict[str(next_lambda)] = []
                p_crash_iter = simulate_run(next_lambda,self.U_prime,self.global_crash_prob_dict, \
                                            self.wfile,self.opt_lambda,None)
                if p_crash_iter > self.p_crash:
                    self.p_crash = p_crash_iter
                    self.opt_lambda = next_lambda
                flush_results(self.global_crash_prob_dict,self.res_file_name)
                self.global_crash_prob_dict = dict() 
                self.status_dict[s_curr]['lambdas_tried'].append(next_lambda)
                self.status_dict[s_curr]['crash_probability'].append(p_crash_iter)
                T_l2 = frac * T_l2
            T_l1 = frac * T_l1 
        
        
            