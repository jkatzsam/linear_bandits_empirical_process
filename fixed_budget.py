import sys
import networkx as nx
import os
os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '1' # export MKL_NUM_THREADS=6
import scipy as sc
import scipy.stats as stats
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import distributions
from sqrtm import sqrtm
from itertools import product, combinations
from transductive_bandits_alg import gammastar_tb_torch, gamma_tb_torch

from library import *
from transductive_bandits_alg import _rounding


def lin_bandit_succ_elim(T, X, Z, thetastar, iters = 100, visualize=False, epsilon=.5):
    logger.debug("solving for initial gamma...")
    l1, initial_gamma = gamma_tb_torch(X, Z, iters=iters, visualize=visualize, objective=True)
    num_epochs = int(np.ceil(max(np.log2(initial_gamma),2)))
    epoch_length = int(np.floor(T/num_epochs))
    accepted = epoch_length > X.shape[0]/epsilon
    print('accepted', accepted, T, num_epochs, T/num_epochs, X.shape[0]/.5)
    logger.info("lin_bandit successive elimination:,T {} num_epochs {} epoch_length {}".format(T,num_epochs,epoch_length))
    Zk = Z
    all_covariance = 0
    all_cross = 0
    outers = np.array([np.outer(X[i,:], X[i,:]) for i in range(X.shape[0])])
    for epoch in range(num_epochs):
        logger.info("successive elimination: epoch {}/{} number of remaining Zs {}".format(epoch,num_epochs,Zk.shape[0]))
        lk, gammak = gamma_tb_torch(X, Zk, iters=iters, visualize=visualize, objective=True)
        allocation = _rounding(lk, epoch_length) 
        #pull new arms
        new_covariance = 0
        new_cross = 0
        for i,num in enumerate(allocation):
            new_covariance += num*outers[i]
            new_cross += (num*np.inner(X[i,:],thetastar)+np.sum(np.random.randn(num)))*X[i,:]
        
        #compute thetak
        all_covariance = all_covariance+new_covariance
        all_cross = all_cross+new_cross
        thetak = np.linalg.pinv(all_covariance)@all_cross.T
        logging.debug('thetak {}'.format(thetak))

        logger.info("allocation {}".format(allocation))
        #sort the remainings into descending order
        idx = np.argsort(-Zk@thetak)
        Zk= Zk[idx]
        
        #eliminate Zs
        if Zk.shape[0] > 2:
            threshold = max(int(np.ceil(gammak/2)),1)
            logger.debug('threshold {}'.format(threshold))
            logger.debug("before elimination: Zk@thetak {}".format(Zk@thetak))
            Zk = eliminate_Zs(threshold, X, Zk, iters = iters, visualize = visualize)
            logger.debug("after elimination: Zk@thetak {}".format(Zk@thetak))
            
    best_z = Zk[np.argmax(Zk@thetak),:]
    logging.info('best_z {}'.format(best_z@thetastar))
    return best_z


def eliminate_Zs(threshold, X,Z,iters = 1000, visualize = False):
    lower_limit = 1
    higher_limit = Z.shape[0]
    
    _, gamma_lower = gamma_tb_torch(X, Z[:1,:], iters=iters, visualize=visualize, objective=True)
    if gamma_lower > threshold:
        return Z

    mdpt = (lower_limit +higher_limit)//2
    attempts = 0
    while higher_limit - lower_limit > 1:
        logger.debug('on round {}'.format(attempts))
        logger.debug('lower {}, mdpt {}, higher{}'.format(lower_limit, mdpt, higher_limit))
        
        _, gamma_mdpt = gamma_tb_torch(X, Z[:mdpt,:], iters=iters, visualize=visualize, objective=True)
        
        if gamma_mdpt > threshold:
            higher_limit = mdpt
        else:
            lower_limit = mdpt
        attempts += 1
        
        mdpt = (lower_limit +higher_limit)//2
    return Z[:max(mdpt,1),:]
        

def uniform_fixed_budget(T,X,Z,thetastar):
    
    n = X.shape[0]
    l = np.ones(n)/n 
    allocation = _rounding(l,T) #compute allocation and round
    pulls = []
    rewards = []
    for i,num in enumerate(allocation):
        for j in range(num):
            pulls.append(list(X[i]))
    pulls_m = np.vstack(pulls)
    rewards = pulls_m@thetastar + np.random.randn(allocation.sum())
    thetak = np.linalg.pinv(pulls_m.T@pulls_m)@pulls_m.T@rewards    
    best_z = Z[np.argmax(Z@thetak),:]    
    return best_z



def make_design(k, d):
    eye = np.eye(k)
    X = []
    alpha1 = 1
    alpha2 = 1
    for seq in product(range(k), repeat=d):
        x = [1]
        for i in seq:
            x.extend(eye[i])
        for idx1,idx2 in combinations(range(d), 2):
            i,j = seq[idx1], seq[idx2]
            a = np.zeros((k,k))
            #print(idx1, idx2, i,j, a.shape)
            a[i,j] = 1*alpha2
            x.extend(a.reshape(-1))
        X.append(x)
    X = np.array(X)
    np.random.seed(50000)
    thetastar = np.zeros(X.shape[1])
    idx1 = 1
    idx2 = 1
    idx3 = 1
    thetastar[d*k+idx1*idx2] = .8
    thetastar[d*k+2*k**2+idx2*idx3] = .05
    np.random.seed()
    return X, thetastar

    
def run_experiment_fixed_budget(k, d, seed, idx):
    print('running experiment')
    np.random.seed(seed)
    X, thetastar = make_design(k,d)
    r = np.linalg.matrix_rank(X)
    V = np.linalg.svd(X)[2][0:r,:]
    X = X@V.T
    thetastar = V@thetastar
    zstar = X[np.argmax(X@thetastar),:]
    Ts = [7500, 15000, 30000, 60000]
    output = {}
    output['uniform'] = {}
    output['succ_elim'] = {}
    for T in Ts:
        print("idx {} T {} executing succ elim alg....".format(idx, T))
        z_alg = lin_bandit_succ_elim(T, X, X, thetastar, iters = 500, visualize=False)
        print('checking equality succ elim idx', np.all(z_alg == zstar), 'T', T)
        print("idx {} T {} executing uniform.... ".format(idx,T))
        z_uniform = uniform_fixed_budget(T,X,X,thetastar)
        
        output['uniform'][T] = np.all(z_uniform == zstar)
        output['succ_elim'][T] = np.all(z_alg == zstar)
        
    print('finished {}'.format(idx))
    return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    