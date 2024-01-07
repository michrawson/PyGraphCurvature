import numpy as np
import networkx as nx
from itertools import combinations
from numpy.random import random_sample

def make_chung_lu_clustering_graph(n,alpha,beta,gamma):

    nodes = list(range(n))

    G = nx.Graph()

    G.add_nodes_from(nodes)

    edges = list(combinations(nodes, 2))

    x_array = np.arange(1,n+1)

    expected_degree_seq = np.exp(alpha) * (x_array**(-beta))

    if np.amax(expected_degree_seq)**2. >= np.sum(expected_degree_seq):
        return nx.Graph(),[],[],0,0
        
    for i in range(expected_degree_seq.shape[0]):
        node_i = nodes[i]
        
        for j in range(i+1,expected_degree_seq.shape[0]):
            node_j = nodes[j]
    
            r = random_sample()
            p_ij = expected_degree_seq[i]*expected_degree_seq[j]/sum(expected_degree_seq)
            if r < p_ij:
                if not G.has_edge(node_i, node_j) and node_i != node_j:
                    G.add_edge(node_i, node_j)
                    G.edges[node_i, node_j]['color'] = "black"
    
    adj = dict(G.adj).copy()

    for i in range(expected_degree_seq.shape[0]):
        node_i = nodes[i]
        
        node_i_nbhd = dict(adj[node_i]).keys()
        
        for j in range(i+1,expected_degree_seq.shape[0]):
            node_j = nodes[j]
    
            node_j_nbhd = dict(adj[node_j]).keys()

            nbhd_ij = list(set(node_i_nbhd).intersection(set(node_j_nbhd)))
            
            for node_k in nbhd_ij:
                r = random_sample()
                if r < gamma: 
                    if not G.has_edge(node_i, node_j):
                        G.add_edge(node_i, node_j)
                        G.edges[node_i, node_j]['color'] = "red"
                        
    return G

