import logging
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import networkx as nx
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch import distributions

from helper_library import *
from oracles import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)



    
def maxZ(z0, linv, shift, thetak, eta, oracle,  tol=1e-8):
    '''
    Computes
    max_z (z0-z) A(l)^{-1/2} eta/(shift + thetak(z0-z)
    
    Input:
        z0: vector of dimension d
        l: vector of dimension d
        shift: a non-negative number
        thetak: vector of dimension d
        eta: vector of dimension d
        oracle: oracle object that contains a ``max`` function passable d dimensional weight vector
    
    Output:
        z: argmax vector
        val: value of above expression
    '''
    d = len(z0)
    s = 'calibrating low:{} high:{} val@high:{} val@low:{}'


    def glinmax(r, arg=False):
        v = -np.sqrt(linv)*eta + r*thetak
        # Call the max_oracle here.
        val, z = oracle.max(v)
        val = val - r*(shift+np.inner(thetak, z0)) + np.sum(z0*np.sqrt(linv)*eta)
        if arg:
            return val, z
        return val 

    high = 2
    low = 1
    if glinmax(0) <= 0:
        low = -1
        logging.debug('set low=-1 '+s.format(low, high, glinmax(high), glinmax(low)))
    elif glinmax(low)< 0:
        while glinmax(low)< 0:
            low = low/2
            
    # calibrate high and low
    while glinmax(high) > 0:
        logger.debug('doubled high '+s.format(low, high, glinmax(high), glinmax(low)))
        high = 2*high
    while glinmax(low) < 0:
        logger.debug('doubled low '+s.format(low, high, glinmax(high), glinmax(low)))
        low = 2*low
    logging.debug('finished '+s.format(low, high, glinmax(high), glinmax(low)))
    
    # Enter Binary Search
    while high-low > tol:
        val, z = glinmax(.5*(high+low), arg = True)
        v = np.sqrt(linv)*eta
        z_obj_val = ((z0-z)@v)/(shift+ thetak@(z0-z))
        
        if np.abs(val -0) <= 1e-8:
            low = .5*(high+low)
            break
        if val < 0:
            high = .5*(low+high)
        else:
            low = max(.5*(low+high), z_obj_val) #take the max with z_obj_val because you know at least one z attains this value          
    val, z = glinmax(low, arg=True)
    return low, z


def _B_combi(d, max_oracle, iters=1000, visualize=False):
    '''
    Return B
    '''
    l = np.ones(d)/d
    saved_grads = []
    for t in range(iters):
        step = .01*np.sqrt(np.log(d)/(iters))
        eta = np.random.randn(d)
        linv = np.array([1/li if li>1e-5 else 0 for li in l])
        val, z = max_oracle.max(np.sqrt(linv)*eta)
        grad = -1/2*z*np.sqrt(linv**3)*eta# TODO - what if l is sparse?
        lold = l
        l = l*np.exp(-step*grad-max(-step*grad))
        l = l/sum(l)
        saved_grads.append(np.linalg.norm(lold-l))
        if visualize and t>100 and t%100==0:
            plt.plot(saved_grads)
            plt.show()
    total = 0
    linv = np.array([1/li if li>1e-5 else 0 for li in l])
    for i in range(500):
        eta = np.random.randn(d)
        val, z = max_oracle.max(np.sqrt(linv)*eta)
        total += val
    return l, (total/500)**2 #B is square of gaussian width 


def gamma_combi(z0, thetak, k, B, max_oracle, iters=1000, batch=1, visualize=False, objective=False):
    '''
    Return an allocation for gamma in round k using OSMD
    Input:
        d: underlying problem dimension d
        max_oracle: max_oracle, needs a max(v) function that returns max(v@Z)
        epsilon: rounding error
        rounds: TODO: replace with stopping criterion
        visualize: If True, plots of norm_grad while computing allocation
    
    Output:
        z0: best arm
        l: allocation
        thetak: empirically computed theta estimate
    '''
    d = len(z0)
    shift = 2**(-k)*B
    l = np.ones(d)/d
    saved_grads = []
    obj_vals = []
    num_resets = 0
    ls = [l.tolist()]
    for t in range(iters):
        step = .01*np.sqrt(1/iters) #TODO: take to be 1/sqrt(iters)
        total_grad = 0
        
        linv = np.array([1/li if li>1e-8 else 1e8 for li in l])
        for b in range(batch):
            eta = np.random.randn(d)
            val, z = maxZ(z0, linv, shift, thetak, eta, max_oracle, tol=1e-8)
            grad = -1/2*(z0-z)*np.sqrt(linv)**3*eta/(shift + (z0-z)@thetak)
            total_grad += grad
        total_grad = total_grad/batch
        lold = l
        l = l*np.exp(-step*total_grad-max(-step*total_grad))
        
        
        #project
        l = l/sum(np.absolute(l))  
        
        #if something weird happens in optimization, reset
        if np.abs(np.sum(l)- 1) > 1e-8:
            num_resets += 1
            logging.info('number of resets {}'.format(num_resets))            
            l = np.ones(d)/d
            
        #check if it has nan and then restart
        if np.sum(np.isnan(l)):
            num_resets += 1
            logging.info('number of resets {}'.format(num_resets))            
            l = np.ones(d)/d    
            
        #add allocation
        ls.append(l.tolist())

        logging.debug('gamma_combi: iter {}, l {}'.format(t, l))
        saved_grads.append(np.linalg.norm(lold-l))
        if t>=100 and t%100==0:
            logging.info('gamma_combi: l {} t {}'.format(l, t))
            if visualize:
                plt.plot(saved_grads)
                plt.show()
                
                num_samples = 30
                total = 0
                linv = np.array([1/li if li>1e-8 else 1e8 for li in l])
                for i in range(num_samples):
                    eta = np.random.randn(d)
                    val, _ = maxZ(z0, linv, shift, thetak, eta, max_oracle, tol=1e-5)
                    total += val
                    
                obj_vals.append(total/num_samples)
                
                plt.plot(obj_vals)
                plt.show()
    
    #final l is a mean of the ls
    l = np.mean(np.array(ls),axis=0)
    print("l {}".format(l))
    if objective:
        total = 0
        linv = np.array([1/li if li>1e-8 else 1e8 for li in l])
        val_ests = []
        while len(val_ests) < 2000 and (len(val_ests) < 300 or np.var(np.array(val_ests)/len(val_ests)) <= 1):
            eta = np.random.randn(d)
            val, _ = maxZ(z0, linv, shift, thetak, eta, max_oracle, tol=1e-5)
            val_ests.append(val)
        
        return l, (np.mean(val_ests))**2
            
    else:
        return l 

    
def combi_alg(d, thetastar, max_oracle, epsilon=.5, rounds=20, batch=1, iters=500, visualize=True, delta=.05, torch=False, use_gap_upper_bound = True):
    '''
    Runs a computationally efficient algorithm for Combinatorial bandits over a family Z.
    
    Input:
        d: underlying problem dimension d
        max_oracle: max_oracle, needs a max(v) function that returns max(v@Z)
        epsilon: rounding error
        rounds: TODO: replace with stopping criterion
        visualize: If True, plots of norm_grad while computing allocation
    
    Output:
        z0: best arm
        l: allocation
        thetak: empirically computed theta estimate
    '''
    X = np.eye(d)
    thetak = np.zeros(d)
    if not use_gap_upper_bound:
        _,B = _B_combi(d, max_oracle, visualize=False)
    else:
        B = 2*d
    print(B)
    
    logging.info('combi_alg: obtained B {}'.format(B))
    all_zs = []
    rewards = np.zeros((d,))
    pulls = np.zeros((d,))
    for k in range(1,rounds):
        _, z0 = max_oracle.max(thetak)
        logging.info('combi_alg: best this round zk {}'.format(z0))
        if torch:
            lk, tk = gamma_combi_torch(z0, thetak, k, B, max_oracle, iters=iters, objective=True, visualize=visualize)
        else:
            lk, tk = gamma_combi(z0, thetak, k, B, max_oracle, 
                                 batch=batch, iters=iters, objective=True, visualize=visualize)
        nk = tk*np.log(((np.pi**2)/6)*2*(k**3)/delta)
        nk = max(4*np.ceil(nk), d)
        logging.info('combi_alg: lk {} tk {} nk {}'.format(lk, tk, nk))
        allocation = np.ceil(lk*nk).astype(int) #compute allocation and round
        print('combi_alg: round {} took {} samples '.format(k, nk))
        print(f"allocation {allocation}")
                
        for i in range(len(allocation)):
            for num_pulls in range(allocation[i]):
                pulls[i] += 1
                rewards[i] += thetastar[i] + np.random.randn()
        thetak = rewards/pulls
        
        gap = toptwogap(thetak, max_oracle)
        gap_exit_cond = B/(2**(k+1))
        logger.info('round {}: gap exit condition {} gap {}'.format(k, gap_exit_cond,gap))
        if gap > gap_exit_cond:
            logger.info('Returning z {}'.format(z0))
            print('number of pulls {}'.format(np.sum(pulls)))
            logger.info('Exiting on round {}'.format(k))
            break

        logging.debug('combi_alg: thetak', thetak)
    return z0, lk, thetak, all_zs, np.sum(pulls)


def toptwogap(theta, max_oracle):
    val, top = max_oracle.max(theta)
    potential = []
    for i in range(len(top)):
        if top[i]:
            thetaprime = theta.copy()
            thetaprime[i] = -10000
            potential.append(max_oracle.max(thetaprime))
    second = sorted(potential, reverse=True, key=lambda x:x[0])[0][1]
    return np.inner(top - second, theta)


def ChenAlg(d, thetastar, max_oracle, rounds=1000, delta=.05, visualize=False):
    rewards = thetastar+np.random.randn()
    pulls = np.ones(d)
    logger.info('clucb: running')
    all_zs = []
    for t in range(d, rounds):
        wt = rewards/pulls
        _, mt = max_oracle.max(wt)
        radt = 1.7 * np.sqrt((np.log(d*np.log2(2*t)) + 0.72*np.log(5.2/delta))/pulls)
        wtilde = np.array([wt[i] - radt[i] if mt[i]==1 else wt[i]+ radt[i] for i in range(d)])
        
        _, mtilde = max_oracle.max(wtilde)
        if np.all(mtilde ==  mt):
            logging.info('Ended in round {}'.format(t))
            return mt, np.sum(pulls)
        pt = []
        for i in range(d):
            if (mt[i]==1 and mtilde[i]==0) or (mt[i]==0 and mtilde[i]==1): 
                pt.append(radt[i])
            else:
                pt.append(-float('inf'))
        pt = np.argmax(pt)
        
        rewards[pt] += thetastar[pt] + np.random.randn()
        pulls[pt] += 1
        if t%1000 == 0 and visualize:
            
            logging.info('t {} mt {}'.format(t, mt))
            logging.info('mtilde {}'.format(mtilde))
            logging.info('radt {}'.format(radt))
            logging.info('wt {}'.format(wt))
            plt.plot(pulls)
            plt.show()
            
    return mt, all_zs, np.sum(pulls)
        
    
if __name__ == "__main__":
    # TEST CASE 1, Naive Oracle Max
    print('test case 1')
    logger.setLevel(logging.DEBUG)
    d = 5
    Z = np.random.randn(6,d)#np.triu(np.ones((d, d + 1)), 1).T
    l = np.random.rand(d); l = l/sum(l)
    eta = np.array([np.sign(np.random.randn()) for i in range(d)])
    shift = 0
    thetak = np.array([-0.22455496, -0.92992234, 0.00534272, -0.98120296, -0.4143715])
    istar = np.argmax(Z@thetak) 
    z0 = Z[istar,:]
    Zp = Z[[i for i in range(Z.shape[0]) if i!= istar],:]
    naive_oracle = NaiveOracle(Zp)
    print('actual argmax', naive_oracle._gfracmax(z0, l, shift, thetak, eta))
    result = maxZ(z0, l, shift, thetak, eta, naive_oracle)
    print('output', result)
