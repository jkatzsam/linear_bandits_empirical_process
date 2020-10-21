import numpy as np
import random
import logging
import time
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
import IPython
import matplotlib.pyplot as plt

class Combi(object):
    def __init__(self, Z, thetastar, delta=.1):
        '''
        policies: set of subsets of [n]
        '''
        self.Z = Z
        self.d = Z.shape[1]
        self.delta = delta
        self.thetastar = thetastar
        self.build_D()
        plt.imshow(self.D)
        print('in init')
        print('built B, D')
                
    def build_D(self):
        self.D = np.zeros((self.Z.shape[0],self.Z.shape[0]), dtype=int)
        self.B = np.zeros((self.Z.shape[0], self.Z.shape[1]))
        maxk = float('-Inf')
        for i in range(self.Z.shape[0]):
            for j in range(i+1, self.Z.shape[0]):
                k = len(Combi._symmetric_difference(self.Z[(i,j),:]))
                self.D[i,j] = k
                self.D[j,i] = k
                self.B[i, k] += 1
                self.B[j, k] += 1
                
    @staticmethod
    def _symmetric_difference(Z):
        z = np.sum(Z, axis=0)
        d = Z.shape[1]
        symmetric_difference = np.array(((z < Z.shape[0]) & (z>0))).nonzero()
        return symmetric_difference[0]
        
    def run(self):
        Zk = self.Z
        t = 1
        thetasum = np.zeros(self.d)
        num_samples = np.zeros(self.d)
        

        while Zk.shape[0]>1: 
            # Sample in symmetric difference 
            Tk = Combi._symmetric_difference(Zk)
            for i in Tk:
                thetasum[i] += np.random.randn()+self.thetastar[i]
                num_samples[i] += 1
            thetahat = thetasum/num_samples
            
            remove = []
            for i in range(Zk.shape[0]):
                for j in range(Zk.shape[0]):
                    if i != j:
                        sym_diff = self.D[i,j]
                        diam = np.log(max(self.B[i, sym_diff], self.B[j, sym_diff]))
                        ci = np.sqrt(8*sym_diff/t*(np.log(np.pi**2*t**2*self.d/(3*self.delta))+diam))
                        if (Zk[i,:] - Zk[j,:])@thetahat >  ci:
                            remove.append(j)
            keep = [i for i in range(Zk.shape[0]) if i not in remove]
            Zk = Zk[keep,:]
            self.D = self.D[np.ix_(keep,keep)]
            self.B =self.B[keep,:]
            
#             if t%50 == 0:
#                 IPython.display.clear_output(wait=True)
#                 fig, ax = plt.subplots(1,3, figsize=(10,3))
                
#                 ax[0].plot(num_samples)
#                 ax[1].imshow(Zk, aspect='auto')
#                 ax[2].imshow(self.D, aspect='auto')
#                 plt.tight_layout()
#                 plt.show()
#                 logging.debug('size of Zk:{}, t:{}'.format(len(Zk), t))
#                 logging.debug('thetahat {}'.format(thetahat))
                
            t += 1
        return Zk[0,:],t
