import numpy as np
import itertools
import logging
import time


class RAGE(object):
    def __init__(self, X, theta_star, epsilon=.5, delta=.05, Z=None):
        
        self.X = X
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(self.Z@theta_star)
        self.delta = delta
        self.epsilon = epsilon
        self.outers = np.array([np.outer(X[i,:], X[i,:]) for i in range(X.shape[0])])
        
        
    def algorithm(self, seed, var=True, binary=False):
        
        self.var=var
        self.seed = seed
        np.random.seed(self.seed)

        self.active_arms = list(range(len(self.Z)))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        self.phase_index = 1
        self.theta_hat = np.random.randn(self.d)
        while len(self.active_arms) > 1:    
            self.delta_t = self.delta/(self.phase_index**2)      
            self.build_Y()
            design, rho = self.optimal_allocation()
            support = np.sum((design > 0).astype(int))
            n_min = support/self.epsilon
            
            num_samples = max(np.ceil(2*(2**(self.phase_index+2))**2*rho*(1+self.epsilon)*np.log(2*self.K_Z**2/self.delta_t)), int(n_min))
            logging.critical('round {} total {} rho {} K_Z {} logKZ {} log {}'.format(self.phase_index, 
                                                                                      num_samples, rho, self.K_Z, 
                                                                                      np.log(2*self.K_Z**2/self.delta_t), np.log(2/self.delta_t)), )
            allocation = self.rounding(design, num_samples)
            pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            logging.critical('design support {}'.format(support))
            logging.critical('allocation: {}'.format(allocation))

            if not binary:
                rewards = pulls@self.theta_star + np.random.randn(allocation.sum())
            else:
                rewards = np.random.binomial(1, pulls@self.theta_star, (allocation.sum()))
            
            self.A_inv = np.linalg.pinv(pulls.T@pulls)
            self.theta_hat = np.linalg.pinv(pulls.T@pulls)@np.sum(pulls*rewards[:,np.newaxis], axis=0)
            self.drop_arms()
            self.phase_index += 1
            self.arm_counts += allocation
            self.N += num_samples
            
            logging.info('\n\n')
            logging.info('finished phase %s' % str(self.phase_index-1))
            logging.info('design %s' % str(design))
            logging.debug('allocation %s' % str(allocation))
            logging.debug('arm counts %s' % str(self.arm_counts))
            logging.info('round sample count %s' % str(num_samples))
            logging.info('total sample count %s' % str(self.N))
            logging.info('active arms %s' % str(self.active_arms)) 
            logging.info('rho %s' % str(rho))      
            logging.info('\n\n')

        del self.Yhat
        #del self.idxs
        del self.X
        del self.Z
        self.success = (self.opt_arm in self.active_arms)
        self.output_arm = self.active_arms[0]
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))
        
            
    
    def build_Y(self):
        curr_Z = self.Z[self.active_arms,:]
        z0 = curr_Z[np.argmax(curr_Z@self.theta_hat),:]
        self.Yhat = z0-curr_Z

        
    
    def optimal_allocation(self):
        
        design = np.ones(self.K)
        design /= design.sum()  
        
        max_iter = 2000
        
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(np.sum(design[:, np.newaxis,np.newaxis]*self.outers, axis=0))
            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V.T
            
            newY = (self.Yhat@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))
                        
            idx = np.argmax(rho)
            y = self.Yhat[idx, :, None]
            g = ((self.X@A_inv@y)*(self.X@A_inv@y)).flatten()
            g_idx = np.argmax(g)
                        
            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma
                
            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
                        
            design += design_update
            
            if count % 100 == 0:
                logging.debug('design status %s, %s, %s, %s' % (self.seed, count, relative, np.max(rho)))
                            
            if relative < 0.01:
                 break
                        
        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        
        return design, np.max(rho)
    
                
    def rounding(self, design, num_samples):
        
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
      
        
    def drop_arms(self):         
        threshold = 2**(-self.phase_index-2)
        curr_Z = self.Z[self.active_arms,:]
        z0 = curr_Z[np.argmax(curr_Z@self.theta_hat),:]
        keep = np.where((z0-curr_Z)@self.theta_hat < threshold)
        self.active_arms = [self.active_arms[i] for i in keep[0]]
        logging.info("num active arms {}".format(len(self.active_arms)))
        logging.info("gaps{} threshold{} idxs{}".format((z0-curr_Z)@self.theta_hat, 
                                                            threshold, np.where((z0-curr_Z)@self.theta_hat < threshold)))
    
    
def rhostar(X, Y, iters=1000):
        
        design = np.ones(self.K)
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