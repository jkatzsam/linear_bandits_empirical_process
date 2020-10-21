import os
#must set these before loading numpy to limit number of threads
os.environ["OMP_NUM_THREADS"] = '1' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '1' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '1' # export MKL_NUM_THREADS=6
import logging
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import networkx as nx
import random
import itertools

from library import *
# from problem_creator import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)




class NaiveOracle(object):
    def __init__(self, Z):
        self.Z = Z
        self.max_calls = 0
    
    def max(self, v):
        self.max_calls += 1
        return np.max(self.Z@v), self.Z[np.argmax(self.Z@v),:]

    def _gfracmax(self, z0, l, shift, thetak, eta):
        '''
        For debugging purposes. Computes
            max_z (z0-z) A(l)^{-1/2} eta/(shift + thetak(z0-z))
        by explicitly computing this over all Z's.
        '''
        def _gfrac(z):
            return np.sum((z0-z)*np.sqrt(1/l)*eta)/(shift + np.inner((z0-z),thetak) ) 
        idx = np.argmax([_gfrac(self.Z[i,:]) for i in range(self.Z.shape[0])])
        fracvalue = _gfrac(self.Z[idx,:])
        diffvalue = (self.Z[idx,:]@(-np.sqrt(1/l)*eta + fracvalue*thetak) 
                     - fracvalue*(shift+np.inner(thetak, z0)) + np.sum(z0*np.sqrt(1./l)*eta))
        return fracvalue, diffvalue


class GraphOracle():
    def __init__(self, G):
        self.G = G
        self.edgelist = list(nx.to_edgelist(G))
        self.edge_to_idx = {}
        self.max_calls  = 0
        for i, edge in enumerate(self.edgelist):
            self.edge_to_idx[edge[0], edge[1]] = i


    def _weightG(self, v):
        '''
        Set the edge weights of G according to v
        '''
        for i, edge in enumerate(self.edgelist):
            a,b,_ = edge
            self.G[a][b]['weight'] = v[i]

    def _path_to_z(self, path):
        '''
        Represent a path as a series of vertices as a vector in the edge basis.
        '''
        z = [0]*len(self.edgelist)
        for i in range(len(path)-1):
            if (path[i], path[i+1]) in self.edge_to_idx.keys():
                z[self.edge_to_idx[(path[i], path[i+1])]] = 1
            else:
                z[self.edge_to_idx[(path[i+1], path[i])]] = 1
        return z

    def _edges_to_z(self, edges):
        '''
        Represent a list of edges as a series of vertices as a vector in the edge basis.
        '''
        z = np.zeros(len(self.edgelist))
        for edge in edges:
            if (edge[0], edge[1]) in self.edge_to_idx.keys():
                z[self.edge_to_idx[(edge[0], edge[1])]] = 1
            else:
                z[self.edge_to_idx[(edge[1], edge[0])]] = 1
        return z
    
    
class ShortestPathDAGOracle(GraphOracle):
    def __init__(self, G, source, target):
        super().__init__(G)
        self.Z = None
        self.source = source
        self.target = target
    
    def _z_to_path(self, z):
        path = [self.source]
        for i in z:
            if i==0:
                pass
            elif i==1:
                a,b,_ = self.edgelist[i]
                if a==path[-1]:
                    path.append(b)
                elif b == path[-1]:
                    path.append(a)
        return path

    def max(self, v):
        '''
        Given the set of weights -v, returns the shortest path between source and target
        '''
        self.max_calls += 1
        self._weightG(-v) #CRITICAL: flips the weights so you can use the shortest-path
        path = nx.bellman_ford_path(self.G,source = self.source, target = self.target,weight='weight')
#         paths = nx.johnson(self.G,weight='weight')
#         path = paths[self.source][self.target]
        z = np.array(self._path_to_z(path))
        logging.debug('max shortest path: {}'.format(z))

        return np.inner(v, z), z
    
    def _makeZ(self):
        self.Z = np.array([self._path_to_z(path) for path in nx.all_simple_paths(self.G,self.source,self.target)])
    
    def _gfracmax(self, z0, l, shift, thetak, eta):
        '''
        For debugging purposes. Computes
            max_z (z0-z) A(l)^{-1/2} eta/(shift + thetak(z0-z)
        by explicitly enumerating all paths as Z's.
        '''
        all_paths = nx.all_simple_paths(self.G, self.source, self.target)
        Z = []
        paths = []
        for i,path in enumerate(all_paths):
            paths.append(path)
            z = self._path_to_z(path)
            Z.append(z)
        Z = np.array(Z)
        self.Z = Z
        
        def _gfrac(z):
            return np.sum((z0-z)*np.sqrt(1/l)*eta)/(shift + np.inner((z0-z),thetak) ) 
        idx = np.argmax([_gfrac(Z[i,:]) for i in range(Z.shape[0])])
        fracvalue = _gfrac(Z[idx,:])
        diffvalue = (Z[idx,:]@(-np.sqrt(1/l)*eta + fracvalue*thetak) 
                     - fracvalue*(shift+np.inner(thetak, z0)) + np.sum(z0*np.sqrt(1./l)*eta))
        return fracvalue, diffvalue 
        
        
class MatchingOracle(GraphOracle):
    def __init__(self, G):
        self.Z = None
        super().__init__(G)
        
    def max(self, v):
        '''
        Given the set of weights -v, returns the shortest path between source and target
        '''
        self.max_calls += 1
        self._weightG(-v) #CRITICAL: flips the weights so you can use the minimum weight matching
        
        matching = nx.bipartite.minimum_weight_full_matching(self.G,weight='weight')
        edges = [(k,v) for k,v in matching.items()]
        z = self._edges_to_z(edges)
        return np.inner(v, z), z
    
    def _makeZ(self):
        if self.Z is None:
            left, right = nx.bipartite.sets(self.G)
            perms = itertools.permutations(right)
            matchings = []
            Z = []
            for p in perms:
                edges = [(i,p[i]) for i in range(len(left))]
                matchings.append(edges)
                z = self._edges_to_z(edges)
                Z.append(z)
            Z = np.array(Z)
            self.Z = Z
    
    def _gfracmax(self, z0, l, shift, thetak, eta):
        '''
        For debugging purposes. Computes
            max_z (z0-z) A(l)^{-1/2} eta/(shift + thetak(z0-z)
        by explicitly enumerating all matchings as Z's.
        Warning: run for graphs with n<5
        '''
        self._makeZ()
        def _gfrac(z):
            return np.sum((z0-z)*np.sqrt(1/l)*eta)/(shift + np.inner((z0-z),thetak) ) 
        idx = np.argmax([_gfrac(self.Z[i,:]) for i in range(self.Z.shape[0])])
        fracvalue = _gfrac(self.Z[idx,:])
        diffvalue = (self.Z[idx,:]@(-np.sqrt(1/l)*eta + fracvalue*thetak) 
                     - fracvalue*(shift+np.inner(thetak, z0)) + np.sum(z0*np.sqrt(1./l)*eta))
        return fracvalue, diffvalue 
    
    def _gaps(self, thetastar):
        self._makeZ()
        zstar = self.Z[np.argmax(self.Z@thetastar),:]
        return (zstar-self.Z)@thetastar
    

class MinSpanTreeOracle(GraphOracle):
    def __init__(self, G):
        self.Z = None
        super().__init__(G)
        
    def max(self, v):
        '''
        Given the set of weights -v, returns the mst with largest weight total
        '''
        self.max_calls += 1
        self._weightG(-v) #CRITICAL: flips the weights so you can use the minimum weight matching
        mst = nx.minimum_spanning_tree(self.G, weight = 'weight')
        edges = [(k,v) for k,v in mst.edges]
        z = self._edges_to_z(edges)
        return np.inner(v, z), z
    
    
class topKOracle():
    def __init__(self,d,k):
        self.Z = None
        self.dim = d #number of arms
        self.k = k #top k
        self.max_calls = 0
        
    def max(self, v):
        '''
        given the set of weights -v, return the set of size k with largest values
        '''
#         neg_v = -v.copy() #flips weights because sort is in ascending order
#         temp = neg_v.argsort()
#         ranks = np.empty_like(temp)
#         ranks[temp] = np.arange(len(neg_v))
#         z = (ranks < self.k).astype('double')
        self.max_calls += 1
        z = find_top_k(v,self.k)
    
        return z@v, z.astype('int')
    
    def _makeZ(self):
        if self.Z is None:
            self.Z = np.array([ls for ls in itertools.product([0, 1], repeat=self.dim) if np.sum(ls) == self.k])

            
class topKSetOracle():
    def __init__(self,set_size, set_num, k):
        '''
        set_size: size of each set
        set_num: number of sets
        k: number top sets to choose
        '''
        self.set_size = set_size
        self.set_num = set_num
        self.d = set_size *set_num
        self.k = k
        self.Z = None
        self.max_calls = 0
        
    def max(self,v):
        '''
        given the set of weights -v, return k sets whose sum is the largest
        
        note: set assume each set has same size and that they are contiguous
            i.e., {0,...,self.set_size-1}, {self.set_size, ..., 2*self.set_size -1}, ...
        '''
        self.max_calls += 1
        v_reshaped = np.reshape(v,(self.set_num, self.set_size))
        set_scores = v_reshaped.sum(axis=1)
        
        set_choices = find_top_k(set_scores, self.k) #the sets that are in the top k
        z = np.repeat(set_choices, self.set_size) #repeat entries for each set choice to get to best z
        
        return z@v, z.astype('int')

    def _makeZ(self):
        if self.Z is None:
            Z = np.vstack([ls for ls in itertools.product([0, 1], repeat=self.set_num) if np.sum(ls) == self.k])
            self.Z = np.repeat(Z,self.set_size, axis=1)
    
    

    
        
    
if __name__ == "__main__":
    
    ############################################
    # TEST CASE 2, Naive Oracle Allocation
    logger.setLevel(logging.DEBUG)
    d = 5
    Z = np.random.rand(1000, d)
    thetastar = np.array([-0.22455496, -0.92992234, 0.00534272, -0.98120296, -0.4143715])
    i_star = np.argmax(Z@thetastar)
    Z_star = Z[i_star, :]
    Zp = Z[[i for i in range(Z.shape[0]) if i!= i_star],:]
    naive_oracle = NaiveOracle(Zp)
    l = gamma_combi(Z_star, thetastar, k=0, B=0, max_oracle=naive_oracle, iters=100)
    plt.plot(l)
    plt.show()
    print(l)
    #############################################
    # TEST CASE 3 - Shortest Path Max
#     G = random_dag(10, 25, seed=25)
#     spo = ShortestPathDAGOracle(G, 2, 6)
#     d = len(spo.edgelist)
#     thetak = np.random.randn(d)
#     _,z0 = spo.max(thetak)
#     print('foundz0', z0)
#     l = np.random.rand(d); l = l/sum(l)
#     eta = np.array([np.sign(np.random.randn()) for i in range(d)])
#     shift = 1

#     result1 = spo._gfracmax(z0, l=l, shift=shift, thetak=thetak, eta=eta)
#     print('obtained through _gfracmax', result2)


#     result2 = maxZ(z0, l=l, shift=shift, thetak=thetak, eta=eta, oracle=spo)
#     print('obtained through maxZ with SPDO', result1)

#     logger.setLevel(logging.DEBUG)
#     Zp = spo.Z
#     naive_oracle = NaiveOracle(Zp)
#     result3 = maxZ(z0, l=l, shift=shift, thetak=thetak, eta=eta, oracle=naive_oracle)
#     print('obtained through Naive', result3)
    
    
    
    #############################################
    # TEST CASE 4 - Maximal Matching Bipartite Graph
    logger.setLevel(logging.DEBUG)
    G = nx.complete_bipartite_graph(3,3)
    mo = MatchingOracle(G)
    d = len(mo.edgelist)
    print('d', d)
    thetak = np.random.randn(d)
    _,z0 = mo.max(thetak)
    print('foundz0', z0)
    l = np.random.rand(d); l = l/sum(l)
    eta = np.array([np.sign(np.random.randn()) for i in range(d)])
    shift = 1

    result1 = mo._gfracmax(z0, l=l, shift=shift, thetak=thetak, eta=eta)
    print('obtained through _gfracmax', result1)


    result2 = maxZ(z0, l=l, shift=shift, thetak=thetak, eta=eta, oracle=mo)
    print('obtained through maxZ', result2)

    logger.setLevel(logging.DEBUG)
    Zp = mo.Z
    naive_oracle = NaiveOracle(Zp)
    result3 = maxZ(z0, l=l, shift=shift, thetak=thetak, eta=eta, oracle=naive_oracle)
    print('obtained through naive', result3)

    #############################################
    # TEST CASE 5 - TopK oracle
    
    d = 10
    k = 3
    mo = topKOracle(d,k)
    
    mo._makeZ()
    Z = mo.Z
    no = NaiveOracle(Z)
    
    for i in range(50000):
        v = np.random.randn(d)
        val1, z1 = mo.max(v)
        val2, z2 = no.max(v)
        
#         if val1 != val2:
        if np.abs(np.sum(val1-val2)) >= 1e-4:
#         if np.sum(z1 != z2) >= 1:
            print("i {}".format(i))
            print("oracles disagree!! argh")
            print("v {} ".format( v))
            print("mo_val {} no_val {}".format(val1, val2))
            print("mo_z {} no_z {}".format(z1,z2))
    


    #############################################
    # TEST CASE 6 - TopKSet oracle

    set_num = 10
    set_size = 3
    k = 3
    mo = topKSetOracle(set_size,set_num,k)
    
    mo._makeZ()
    Z = mo.Z
    no = NaiveOracle(Z)
    
    for i in range(50000):
        v = np.random.randn(set_num*set_size)
        val1, z1 = mo.max(v)
        val2, z2 = no.max(v)
        
#         if val1 != val2:
#         if np.abs(np.sum(val1-val2)) >= 1e-4:
        if np.sum(z1 != z2) >= 1:
            print("i {}".format(i))
            print("oracles disagree!! argh")
            print("v {} ".format( v))
            print("mo_val {} no_val {}".format(val1, val2))
            print("mo_z {} no_z {}".format(z1,z2))
            
    #############################################
    # TEST CASE 7 - shortest path
    
    layer_num = 2
    layer_width = 3
    G, source, sink = createFeedforwardGraph(layer_num, layer_width)
    mo = ShortestPathDAGOracle(G, source, sink)
    d = len(mo.edgelist)

    mo._makeZ()
    no=NaiveOracle(mo.Z)
    
    for i in range(50000):
        v = np.random.randn(d)
        val1, z1 = mo.max(v)
        val2, z2 = no.max(v)
        
#         if val1 != val2:
#         if np.abs(np.sum(val1-val2)) >= 1e-4:
        if np.sum(z1 != z2) >= 1:
            print("i {}".format(i))
            print("oracles disagree!! argh")
            print("v {} ".format( v))
            print("mo_val {} no_val {}".format(val1, val2))
            print("mo_z {} no_z {}".format(z1,z2))
            
            
            
    #############################################
    # TEST CASE 8 - mst