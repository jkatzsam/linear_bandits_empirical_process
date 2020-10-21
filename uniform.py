import logging
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import networkx as nx
import random

from library import *


def uniform(d, thetastar, max_oracle, rounds = 100000000, delta=.05):

    
    #compute gaussian width for stopping condition
    l = np.ones(d)/d
    linv = np.array([1/li for li in l])
    total = 0
    for i in range(1000):
        eta = np.random.randn(d)
        val, z = max_oracle.max(np.sqrt(linv)*eta)
        total += val
    gamma_unif = (total/1000)**2
    logger.info('gamma_unif {}'.format(gamma_unif))
    
    high_prob_term, _ = max_oracle.max(linv)
    
    #start taking samples
    rewards = thetastar+np.random.randn()
    pulls = np.ones(d)
    logger.info('uniform: running')
    for t in range(d, rounds):
        wt = rewards/pulls
        
        pt = t % d
        
        rewards[pt] += thetastar[pt] + np.random.randn()
        pulls[pt] += 1

        #check if difference between top empirical and second top empirical is greater than this. then terminate
        if t % 1000 == 0:
            gap = toptwogap(wt, max_oracle)
            gap_exit_conditionk = np.sqrt(4* np.log(1/delta)*gamma_unif/np.sum(pulls)) #+ np.sqrt(2* np.log(1/delta)* high_prob_term/np.sum(pulls)) #NOTE: OMMITTED high prob term
            if gap >= gap_exit_conditionk:
                _, z0 = max_oracle.max(wt)
                logger.info('Returning z {}'.format(z0))
                logger.info('number of pulls {}'.format(len(rewards)))
                logger.info('Exiting on round {}'.format(t))
                break
    
    _, z0 = max_oracle.max(wt)
     
    print("uniform: pulls {}".format(pulls))
    
    return z0, t