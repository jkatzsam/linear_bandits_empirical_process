import logging
import os
#must set these before loading numpy to limit number of threads
os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '1' # export MKL_NUM_THREADS=6
import numpy as np
import matplotlib.pyplot as plt
import logging, sys, itertools, time, pickle
import networkx as nx
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import distributions

from helper_library import *
from oracles import *
from library import *
from uniform import *
from problem_creator import *
from fixed_budget import *
from transductive_bandits_alg import *
from utilsBandit import * #contains akshay algorithm

import multiprocessing as mp
from collections import defaultdict


def run_experiment(thetastar, mo, seed, idx, run_akshay_alg = False, uniform_less = True, main_algs = True, torch=False, params=None):
    print(f"uniform_less {uniform_less}")
    np.random.seed(seed)
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    
    d = len(thetastar)
    val, zstar = mo.max(thetastar)
    
    def found_zstar(z):
        return (np.sum(z != zstar) == 0)

    output = {}
    if run_akshay_alg:
        alg = Combi(mo.Z, thetastar, delta=.05)
        output_akshay = alg.run()   
        
        output['correct_akshay']=  found_zstar(output_akshay[0])
        output['num_samples_akshay'] =  output_akshay[1]
        output['Z'] = mo.Z
    
    if main_algs:
        mo.max_calls = 0
        print('running combi no torch idx {}'.format(idx))
        output_combi_no_torch  = combi_alg(d, thetastar, mo, rounds=50, 
                                           batch=10, iters=1000, #needs to be 1000
                                           visualize=False, torch=False)
        combi_no_torch_oracle_calls = mo.max_calls
        print('combi calls', combi_no_torch_oracle_calls)
        if torch:
            print('running combi torch idx {}'.format(idx))
            mo.max_calls = 0
            output_combi_torch  = combi_alg(d, thetastar, mo, rounds=100, 
                                      batch=5, iters=50, 
                                      visualize=False, torch=False)
            combi_torch_oracle_calls = mo.max_calls

        mo.max_calls = 0
        print('running chen idx {}'.format(idx))
        output_chen = ChenAlg(d, thetastar, mo, rounds=100000000, delta=.05, visualize=False)    
        chen_oracle_calls = mo.max_calls
        print('chen calls', chen_oracle_calls)

        main_alg_output = {'correct_chen': found_zstar(output_chen[0]),
                  'correct_combi_no_torch':  found_zstar(output_combi_no_torch[0]),
                  'num_samples_chen': output_chen[-1], 
                  'num_samples_combi_no_torch': output_combi_no_torch[-1],
                  'combi_no_torch_oracle_calls': combi_no_torch_oracle_calls,
                  'chen_oracle_calls':chen_oracle_calls,
                  'thetastar': thetastar,
                  'zstar': zstar,
                  'oracle': mo,
                  'output_chen': output_chen,
                  'output_combi_no_torch': output_combi_no_torch,
                  'dim' : len(thetastar)
                 }
        
        output = {**output, **main_alg_output}
    
    mo.max_calls = 0
    print('running uniform idx {}'.format(idx))
    if not uniform_less:
        output_uniform = uniform(d, thetastar, mo, rounds = 100000000, delta=.05)
        
        if not output:
            output = {'thetastar': thetastar,
                      'zstar': zstar,
                      'oracle': mo,
                      'dim' : len(thetastar)
                     }

        output['correct_uniform']=  found_zstar(output_uniform[0])
        output['num_samples_uniform'] =  output_uniform[1]
    
    if torch:
        output['corrrect_torch'] = output_combi_torch[0]
        output['num_samples_combi_torch'] = output_combi_torch[-1]
        output['output_combi_torch'] = output_combi_torch
        output['combi_torch_oracle_calls'] = combi_torch_oracle_calls
    print('finished {}'.format(idx))
    return output
 
        
###experiments    
def paths_experiment_divided_net_vary_gap(num_cores, num_trials = 5):
    print("running paths experiment divided net style vary gap....")

    gaps = [.2,.15,.1,.05]

    params = []
    experiment_num = 0
    for gap in gaps: 
        print(f"gap {gap}")
        for trial in range(num_trials):
            
            G,source, sink, thetastar = generate_divided_net_sparse(26, 2, diff = gap, num_paths = 2)
            mo = ShortestPathDAGOracle(G,source,sink)

            params.append((thetastar, mo, np.random.randint(1e6), experiment_num, False, False))
            
            experiment_num += 1
            print(f'thetastar {thetastar}')

            
    pool = mp.Pool(num_cores)
    output = pool.starmap(run_experiment, params)

    filename = './results/{}_{}.pkl'.format(time.time(), 'shortest_path_divided_net_vary_gap')
    with open(filename, 'wb') as f:
        pickle.dump(output, f) 
        
        
def matching_experiment_vary_gap(num_cores, num_trials = 5):
    print("running matching experiment varying number of good zs....")
    
    gaps = [.15,.1,.05,.025]

    params = []
    experiment_num = 0
    for gap in gaps:
        print(f"gap {gaps}")
        for trial in range(num_trials):
        
            G, thetastar=generate_bipartite_graph_two_groups_sparse(14, diff = gap, other_val = 0)
            mo = MatchingOracle(G)

            params.append((thetastar, mo, np.random.randint(1e6), experiment_num, False, False))
            
            experiment_num += 1
            print(f'thetastar {thetastar}')
            
    pool = mp.Pool(num_cores)
    output = pool.starmap(run_experiment, params)

    filename = './results/{}_{}.pkl'.format(time.time(), 'matching_vary_gap')
    with open(filename, 'wb') as f:
        pickle.dump(output, f)
        
        
def biclique_experiment(num_cores, num_trials):
    print(f"running biclique experiment....")
    
    gaps = [.1,.08,.06,.04,.02]

    Z = create_biclique(matrix_size=8,num_side=2)

    print(f"Z {Z}")
    dim = len(Z[0])
    
    params = []
    experiment_num = 0
    for gap in gaps:                    
        for trial in range(num_trials):
                    
            #make oracle and thetastar
            mo = NaiveOracle(Z)
            rand_weights = np.random.randn(dim)
            val, zstar = mo.max(rand_weights)
            
            thetastar = np.zeros((dim,))
            np.putmask(thetastar,zstar.astype(int),1)
            
            #pick zstar2
            np.putmask(rand_weights,zstar.astype(int),-100000)
            val, zstar2 = mo.max(rand_weights)
            
            #pass in parameters
            params.append((thetastar, mo, np.random.randint(1e6), experiment_num, True, False))
            
            experiment_num += 1
            
    pool = mp.Pool(num_cores)
    output = pool.starmap(run_experiment, params)
    
    filename = './results/{}_{}.pkl'.format(time.time(), 'biclique')
  
    with open(filename, 'wb') as f:
        pickle.dump(output, f)  

def transductive_bandits(num_cores, num_trials, n):
    outputs = []
    for n in [1000, 10000, 100000, 1000000]:
        outputs.append(parallelize_transductive_bandits(num_cores, num_trials, n))
    filename = './transductive.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(outputs, f)
    
    
def fixed_budget(k, d, num_cores, num_trials = 5):
    print("running multivariate bandit testing experiment....", num_cores)
    params = [(k, d, np.random.randint(1e6), trial) for  trial in range(num_trials)]
    
    pool = mp.Pool(num_cores)
    print('pool allocated')
    outputs = pool.starmap(run_experiment_fixed_budget, params)

    filename = './fixed_budget.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(outputs, f)

if __name__=='__main__':
    
#    paths_experiment_divided_net_vary_gap(num_cores = 15, num_trials = 3)

#     matching_experiment_vary_gap(num_cores = 15, num_trials = 3)

    biclique_experiment(num_cores = 15, num_trials = 3)
#     transductive_bandits(num_cores=15, num_trials = 5, n=100)
#    fixed_budget(k=6, d=3, num_cores=15, num_trials=15)
