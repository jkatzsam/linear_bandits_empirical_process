import numpy as np
import numpy as np
import logging
import itertools
import pickle
import os
import sys
import functools


def rhostar(X, Y, thetastar, iters=1000):
        Y = (z0-Z)/((z0-Z)@thetastar)
    
        design = np.ones(X.shape[0])
        design /= design.sum()  
        for count in range(1, iters):
            A_inv = np.linalg.pinv(X.T@np.diag(design)@X)    
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
            if count % 100 == 0:
                logging.debug('design status %s, %s, %s' % (count, relative, np.max(rho)))
            if relative < 0.01:
                 break
        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        return design, np.max(rho)      






#Note that the following code is from Fiez 2019 paper
def orthogonal_design_problem_instance(D, prob_nonzero = 1, prob_arm = 1, num_sparse=None):

    alpha1 = 1
    alpha2 = 0.5
    variants = list(range(D))
    individual_index_dict = {}

    count = 0
    for key in variants:
        individual_index_dict[key] = count
        count += 1

    pairwise_index_dict = {}
    count = 0
    pairs = []
    for pair in itertools.combinations(range(D), 2):
        pairs.append(pair)
        key1 = pair[0]
        key2 = pair[1]
        pairwise_index_dict[(key1, key2)] = count
        count += 1

    individual_offset = 1
    pairwise_offset = 1 + len(individual_index_dict)
    num_features = 1 + len(individual_index_dict) + len(pairwise_index_dict)
    num_arms = 2**D

    combinations = list(itertools.product([-1, 1], repeat=D))

    X = -np.ones((num_arms, num_features))

    for idx in range(num_arms):
        bias_feature_index = [0]
        individual_feature_index = [individual_offset + individual_index_dict[i] for i, val in enumerate(combinations[idx]) if val == 1]
        pairwise_feature_index = [pairwise_offset + pairwise_index_dict[pair] for pair in pairs if combinations[idx][pair[0]] == combinations[idx][pair[1]]]
        feature_index = bias_feature_index + individual_feature_index + pairwise_feature_index
        X[idx, feature_index] = 1

    while True:
        theta_star = np.random.randint(-3, 3+1, (num_features, 1))/10
        #theta_star = np.random.permutation(np.linspace(-.5,.5, num_features))
        #theta_star = theta_star*np.random.binomial(1,prob_nonzero,size=theta_star.shape)
        theta_star = theta_star[:,0]
        theta_star[individual_offset:pairwise_offset] = alpha1*theta_star[individual_offset:pairwise_offset]
        theta_star[pairwise_offset] = alpha2*theta_star[pairwise_offset]

        if num_sparse:
            sparse_index = np.zeros(D)
            sparse_index[np.random.choice(len(sparse_index), num_sparse, replace=False)] = 1
            bias_feature_index = [0]
            individual_feature_index = [individual_offset + individual_index_dict[i] for i, val in enumerate(sparse_index) if val == 1]
            pairwise_feature_index = [pairwise_offset + pairwise_index_dict[pair] for pair in pairs if sparse_index[pair[0]] == 1 and sparse_index[pair[1]] == 1]
            feature_index = bias_feature_index + individual_feature_index + pairwise_feature_index
            theta_star[~np.array(feature_index)] = 0

        rewards = (X@theta_star).reshape(-1)
        top_rewards = sorted(rewards, reverse=True)[:2]
        
        if top_rewards[0] - top_rewards[1] < 10e-6:
            continue
        else:
            break
    
    #only keep some of the arms
    #print(top_rewards)
    #X = X[np.random.binomial(1, prob_arm,X.shape[0]).astype(bool),:]      
    #print(X)
    return X, theta_star


def top_k_allocation(theta, k):
    '''
    Computes
    optimal allocation for topK

    Input:
        theta: (true theta)
        k: number of arms to return
    
    Output:
        l: probabilistic allocation over arms
    '''
    if len(theta) <= k:
        print("error: k smaller than dimension")
        return
    
    #find largest theta_i outside of top k and smallest theta_i inside of topk
    neg_theta = -theta #flips weights because sort is in ascending order
    idx_sorted = neg_theta.argsort()
    theta_sorted = theta[idx_sorted]
    theta_high = theta_sorted[k-1] #account for indexing
    theta_low = theta_sorted[k] #account for indexing

    #calculate inverse squares
    l = np.zeros_like(theta).astype(float)
    for i in range(len(theta)):
        if theta[i] >= theta_high:
            l[i] = 1/np.power(theta[i]-theta_low,2)
        else:
            l[i] = 1/np.power(theta[i]-theta_high,2)
            
    #normalize
    complexity = np.sum(l)
    l = l/np.sum(l)
    
    print(l)
    return l,complexity

def find_top_k(v,k):
    '''
    helper function for finding top k
    '''

    neg_v = -v.copy() #flips weights because sort is in ascending order
    temp = neg_v.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(neg_v))
    z = (ranks < k).astype('double')
    
    return z

    
if __name__ == "__main__":

    #test case 1: top_k_allocation
    epsilon = .5
    k = 10
    d = 30
    theta = np.array([epsilon]*k + [0]*(d-k))
    theta = np.random.permutation(theta) #permute arms
    
    true_alloc = np.array([1/np.power(epsilon,2)]*d)
    true_alloc = true_alloc/np.sum(true_alloc)
    
    returned_alloc = top_k_allocation(theta, k)
    
    print("true_alloc {}".format(true_alloc))
    print("returned alloc {}".format(returned_alloc))
    print("success {}".format(returned_alloc == true_alloc))