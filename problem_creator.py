import os
#must set these before loading numpy to limit number of threads
os.environ["OMP_NUM_THREADS"] = '8' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '8' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '8' # export MKL_NUM_THREADS=6
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import networkx as nx
import random
import itertools

from oracles import *



def createDividedFeedforwardGraph(layer_num, layer_width):
    '''
    Creates feedforward network
    
    input:
        layer_num: number of layers
        layer_width: width of the layers
        edge_prob: probability that edge b/w intermediate layers is kept
        
    output:
        G: graph 
        source and sink
    '''

    G = nx.DiGraph()

    #create nodes and group into layers
    layers = {}
    layers[0] = {}
    layers[1] = {}
    
    source = 0
    sink = 1
    G.add_node(source)
    G.add_node(sink)
    node_counter = 2
    
    for graph_part in [0,1]:

        for i in range(layer_num):
            layers[graph_part][i] = []
            for j in range(layer_width):

                layers[graph_part][i].append(node_counter)
                G.add_node(node_counter)
                node_counter += 1

        #add_edges
        for i in layers[graph_part][0]:
            G.add_edge(source, i)

        for i in layers[graph_part][layer_num-1]:
            G.add_edge(i, sink)

        for cur_layer in range(layer_num-1):
            for node_prev in layers[graph_part][cur_layer]:
                for node_next in layers[graph_part][cur_layer+1]:
                    G.add_edge(node_prev,node_next)

    return G, source, sink, layers


def generate_divided_net_sparse(layer_num, layer_width, diff = .1, num_paths = 5):
    
    G, source, sink, layers = createDividedFeedforwardGraph(layer_num, layer_width)
    mo = ShortestPathDAGOracle(G,source,sink)
    d = len(mo.edgelist)
    thetastar = np.zeros((d,))
    
    edge1 =  mo.edge_to_idx[(0,layers[0][0][0])]
    weights = np.ones(d)
    weights[edge1] = 1000000000
    val, new_good_z = mo.max(weights)

    np.putmask(thetastar,new_good_z.astype(int),1)
    
    edge2 = mo.edge_to_idx[(0,layers[1][0][0])]
    weights = np.ones(d)
    weights[edge2] = 1000000000
    val, new_good_z = mo.max(weights)

    np.putmask(thetastar,new_good_z.astype(int),1-diff)
    
    np.putmask(thetastar,thetastar == 0, -1)
        
    return G, source, sink, thetastar


def generate_bipartite_graph_two_groups_sparse(num_nodes, diff = .1,other_val = 0):
    '''
    Create bipartite graph where each node has three edges to the other side
    '''

    G = nx.complete_bipartite_graph(num_nodes,num_nodes)

    #two groups in bipartite graph
    U = list(nx.bipartite.sets(G)[0])
    V = list(nx.bipartite.sets(G)[1])


    mo = MatchingOracle(G)
    d = len(mo.edgelist)
    thetastar = np.zeros((d,))

    for i in range(num_nodes):
        thetastar[mo.edge_to_idx[(U[i],V[i])]] = 1

    for i in range(num_nodes):
        thetastar[mo.edge_to_idx[(U[i-1],V[i])]] = 1-diff
        
    np.putmask(thetastar,thetastar == 0, other_val)
        
    return G, thetastar

    
def create_biclique(matrix_size,num_side):
    '''
    create Zs for biclique
    '''
    
    dim = matrix_size**2

    edge_ids = []
    edge_to_idx = {}
    idx = 0
    for i in range(matrix_size):
        for j in range(matrix_size):
            edge_ids.append((i,j))
            edge_to_idx[(i,j)] = idx

            idx += 1

    all_groups = list(itertools.combinations(range(matrix_size),num_side))
    Zs = []
    for group_1 in all_groups:
        for group_2 in all_groups:
            z = np.zeros((dim,))
            for i in group_1:
                for j in group_2:
                    z[edge_to_idx[(i,j)]] = 1

            Zs.append(list(z))
            
    return np.array(Zs)


if __name__ == "__main__":

    #test manhattan grid
    G,source,sink = create_manhattan_grid(grid_size=5)
    nx.draw_networkx(G)
    
    #biclique test
    num_side = 2
    Zs = create_biclique(matrix_size=5,num_side=2)
    
    for z in Zs:
        if np.sum(z) != num_side**2:
            print("there is issue")
