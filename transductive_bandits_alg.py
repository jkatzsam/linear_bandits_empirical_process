import sys
import networkx as nx
import scipy as sc
import numpy as np
import random
import itertools
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import distributions
from sqrtm import sqrtm
from library import *
from RAGE import RAGE
import time 
import pickle

    
    



def optimal_allocation(X, Y, outers=None):
    design = np.ones(X.shape[0])
    design /= design.sum()  
    max_iter = 2000
    if outers is None:
        outers = np.array([np.outer(X[i,:], X[i,:]) for i in range(X.shape[0])])
    for count in range(1, max_iter):
        A_inv = np.linalg.pinv(np.sum(design[:, np.newaxis,np.newaxis]*outers, axis=0))    
        U,D,V = np.linalg.svd(A_inv)
        Ainvhalf = U@np.diag(np.sqrt(D))@V.T
        newY = (Y@Ainvhalf)**2
        rho = newY@np.ones((newY.shape[1], 1))           
        idx = np.argmax(rho)
        y = Y[idx, :, None]
        g = ((X@A_inv@y)*(X@A_inv@y)).flatten()
        g_idx = np.argmax(g)        
        gamma = 2/(count+2)
        design_update = -gamma*design
        design_update[g_idx] += gamma
        relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
        design += design_update
        if relative < 0.01:
            break      
    idx_fix = np.where(design < 1e-5)[0]
    design[idx_fix] = 0
    return design, np.max(rho)

def build_Y(Z, theta):
    z0 = Z[np.argmax(Z@theta),:]
    Y = z0-Z
    return Y

def lin_tb_alg(X, Z, thetastar, rounds=20, iters=500, visualize=True, delta=.05, epsilon=.5, idx=0):
    d = len(X[0])
    B = None
    Zk = Z
    K = Z.shape[0]
    total_samples = 0
    thetak = np.random.randn(d)
    outers = np.array([np.outer(X[i,:], X[i,:]) for i in range(X.shape[0])])
    k = 1
    while Zk.shape[0] > 1:
        all_covariance = 0
        all_cross = 0
        Y = build_Y(Zk, thetak)
        logging.debug('Y computed')
        
        # Gamma allocation
        lk_gamma, tk_gamma = gamma_tb_torch(X, Y, iters=iters, objective=True, visualize=visualize)        
        # rho allocation
        lk_rho, tk_rho = optimal_allocation(X, Y, outers=outers)
        # mix allocations
        lk = (lk_gamma +lk_rho)/2
        # compute number of samples
        tk = 2*tk_rho*np.log(2*k**2/delta)+tk_gamma
        if B is None:
            B = tk

        # Compute the support
        support = np.sum((lk > 0).astype(int))
        # q(epsilon)
        n_min = support/epsilon
        # number of samples is maximum
        # Note that this is not quite the same k as in the paper.
        nk = max(np.ceil(tk*(2**(k+2)/B)**2*(1+epsilon)), n_min)
        total_samples += nk
        allocation = _rounding(lk,nk) #compute allocation and round

        logging.critical('gamma round:{} tk:{:.2f} nk:{} rho:{} gamma:{} B:{}'.format(k, tk, nk, tk_rho, tk_gamma, B))
        logging.critical('design lk {}'.format(lk))
        logging.critical('allocation: {}'.format(allocation))
        
        # Take Samples
        new_covariance = 0
        new_cross = 0
        for i,num in enumerate(allocation):
            new_covariance += num*outers[i]
            if num > 0:
                new_cross += np.sum(num*(X[i,:]@thetastar)+np.sum(np.random.randn(num)))*X[i,:]
        all_covariance = all_covariance+new_covariance
        all_cross = all_cross+new_cross
        thetak = np.linalg.pinv(all_covariance)@all_cross.T
        
        # Remove suboptimal arms
        z0 = Zk[np.argmax(Zk@thetak),:]
        Zk = Zk[np.where((z0-Zk)@thetak < B/2**(k+1))]

        logging.critical('size of Zk {}'.format(Zk.shape[0]))
        logging.info('linear: best this round zk {}'.format(z0))
        logging.debug('linalg: thetak', thetak)
        if Zk.shape[0] == 1:
            z0 = Zk[0]
            break
        k+=1
    print('found', z0)
    return z0, lk, thetak, total_samples

def gamma_tb_torch(X_data, Z, iters=1000, visualize=False, objective=False):
    n = X_data.shape[0]
    d = X_data.shape[1]
    x = nn.Parameter(torch.zeros(X_data.shape[0])) 
    optim = Adam([x], lr=1e-2)
    l = torch.softmax(x, dim=-1)
    # X is shape n by d
    X = torch.tensor(X_data).float()
    # this computes bmm on a (n, d, 1) tensor with a (n, 1, d) tensor. The result is (n, d, d)
    outers = torch.bmm(X.unsqueeze(2), X.unsqueeze(1))  
    saved_grads = []
    for t in range(iters):
        A = torch.sum(l.view(-1, 1, 1) * outers, dim=0).requires_grad_()  
        #draw eta
        eta = np.random.randn(d)
        #compute z that attains max
        A_sqrt_inv = sqrtm(torch.inverse(A).requires_grad_()).requires_grad_()
        _,max_z = gamma_est(Z, A_sqrt_inv.clone().detach().numpy(),l.detach().numpy(), eta)
        loss = (A_sqrt_inv @ torch.tensor(max_z).float()) @ torch.tensor(eta).float()
        if visualize:
            lold = l.detach().numpy().copy()
        loss.backward()
        optim.step()
        optim.zero_grad()
        l = torch.softmax(x, dim=-1)
        if visualize:
            saved_grads.append(np.linalg.norm(lold-l.detach().numpy()))
        if t>=500 and t%500==0:
            logger.debug('gamma_combi: l {} t {}'.format(l, t))
            if visualize:
                plt.plot(saved_grads)
                plt.show()
    ldetach = l.detach_().numpy()
    A = torch.sum(l.view(-1, 1, 1) * outers, dim=0)
    A_sqrt_inv = sqrtm(torch.inverse(A)).detach().numpy()
    if objective:
        total = 0
        for i in range(1000):
            eta = np.random.randn(d)
            val,_ = gamma_est(Z, A_sqrt_inv, ldetach, eta)
            total += val
        return ldetach, (total/1000)**2 #it is gaussian width squared
    else:
        return ldetach 
    
    
def gamma_est(Z, A_sqrt_inv, l, eta):
    scores  = Z@A_sqrt_inv@eta
    idx = np.argmax(scores)
    z_1 = Z[idx,:]
    return scores[idx], z_1


def _rounding(design, num_samples):        
    '''
    Routine to convert design to allocation over num_samples following rounding procedures in Pukelsheim.
    '''
    num_support = (design > 0).sum()
    support_idx = np.where(design>0)[0]
    support = design[support_idx]
    n_round = np.ceil((num_samples - .5*num_support)*support)
    while n_round.sum()-num_samples != 0:
        if n_round.sum() < num_samples:
            idx = np.argmin(n_round/support)
            n_round[idx] += 1
        else:
            idx = np.argmax((n_round-1)/support)
            n_round[idx] -= 1

    allocation = np.zeros(len(design))
    allocation[support_idx] = n_round
    return allocation.astype(int)



def run_experiment_transductive(seed, idx, n, params=None):
    np.random.seed(seed)
    Z = [[1,0], [np.cos(3*np.pi/4), np.sin(3*np.pi/4)]]
    #Z.extend([ [np.cos(.01+r), np.sin(.01+r)] for r in np.random.rand(n)*.03+.01] )
    Z.extend([ [np.cos(np.pi/4+r), np.sin(np.pi/4+r)] for r in np.random.rand(n)*.05+np.pi/4] )
    Z = np.array(Z)
    thetastar = Z[0,:]
    zstar = Z[np.argmax(Z@thetastar),:]
    X = np.eye(2)
    def found_zstar(z):
        return (np.sum(z != zstar) == 0)
    print('running linalg torch idx {}'.format(idx))
    output_linalg = lin_tb_alg(X, Z, thetastar, epsilon=.5, 
                              rounds=20, iters=500, 
                              visualize=False, delta=.05, idx=idx)
    
    logging.critical("********************")
    print('running RAGE idx {}'.format(idx))
    
    rage = RAGE(X, thetastar, epsilon=.5, delta=.05, Z=Z)
    rage.algorithm(seed, var=True)
    print(rage.output_arm)
    output = {'correct_rage': rage.success,
              'correct_linalg':  found_zstar(output_linalg[0]),
              'num_samples_rage': rage.N, 
              'num_samples_linalg': output_linalg[-1],
              'thetastar': thetastar,
              'zstar': zstar,
              'output_rage': rage,
              'output_linalg': output_linalg,
              'dim' : len(thetastar)
             }
    print('finished {}'.format(idx))
    return output

def parallelize_transductive_bandits(num_cores, num_trials = 5, n=100):
    
    pool = mp.Pool(num_cores)
    print('pool allocated')
    params = [(np.random.randint(1e6), trial, n) for  trial in range(num_trials)]
    output = pool.starmap(run_experiment_transductive, params)
    return output

def gammastar_tb_torch(X_data, Z, z0, thetak, k, B, iters=1000, visualize=False, objective=False):
    n = X_data.shape[0]
    d = X_data.shape[1]
    shift = 2**(-k)*B
    saved_grads = []
    x = nn.Parameter(torch.zeros(X_data.shape[0]))
    optim = Adam([x], lr=1e-2)
    l = torch.softmax(x, dim=-1)
    X = torch.tensor(X_data).float()
    outers = torch.bmm(X.unsqueeze(2), X.unsqueeze(1))  
    logging.critical('outers made in gammastar')
    for t in range(iters):
        A = torch.sum(l.view(-1, 1, 1) * outers, dim=0).requires_grad_()   
        A_sqrt_inv = sqrtm(torch.inverse(A).requires_grad_()).requires_grad_()
        eta = np.random.randn(d)
        _,z = gfrac(Z, z0, A_sqrt_inv.clone().detach().numpy(), shift, thetak, eta)        
        loss = (((A_sqrt_inv @ torch.tensor(z0-z).float()) @ torch.tensor(eta).float())/
                (shift + (z0-z)@thetak))
        lold = l.detach().numpy().copy()
        optim.zero_grad()
        loss.backward()
        optim.step()
        l = torch.softmax(x, dim=-1)
        logger.debug('gamma_combi: iter {}, l {}, x {}'.format(t, l.detach().numpy(), x.detach().numpy()))
        saved_grads.append(np.linalg.norm(lold-l.detach().numpy()))
        if t>=500 and t%500==0:
            logger.info('gamma_combi: l {} t {}'.format(l, t))
            if visualize:
                plt.plot(saved_grads)
                plt.show()
    ldetach = l.detach_().numpy()
    A = torch.sum(l.view(-1, 1, 1) * outers, dim=0)
    A_sqrt_inv = sqrtm(torch.inverse(A)).detach().numpy()
    if objective:
        total = 0
        for i in range(500):
            eta = np.random.randn(d)
            val,_ = gfrac(Z, z0, A_sqrt_inv, shift, thetak, eta)
            total += val
        return ldetach, (total/500)**2 #it is gaussian width squared
            
    else:
        return ldetach 
    
def gfrac(Z, z0, A_sqrt_inv, shift, thetak, eta):
    def _gfrac_helper(z):
        return np.sum(A_sqrt_inv@(z0-z)@eta)/(shift + np.inner((z0-z),thetak) )
    values = [_gfrac_helper(Z[i,:]) for i in range(Z.shape[0])]
    idx = np.argmax(values)
    z_1 = Z[idx,:]
    return values[idx], z_1