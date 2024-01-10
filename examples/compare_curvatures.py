import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from example_graphs import make_chung_lu_clustering_graph
from zoneinfo import ZoneInfo
from time import time
import sys
sys.path.insert(0, '..')
from graph_curvature import lly_curvature, forman_curvature, forman_aug_curvature  # noqa: E402


tz = ZoneInfo("America/Los_Angeles")


def compare_curvatures(G, G_fcurv, G_fcurv_alt, LLY_curvatures):

    G_adj = np.zeros((len(G.nodes), len(G.nodes)))

    G_nodes = list(G.nodes)

    G_nodes_ind_map = {}
    for node_ind in range(len(G_nodes)):
        node = G_nodes[node_ind]
        G_nodes_ind_map[node] = node_ind

    for node1_ind in range(len(G_nodes)):
        node1 = G_nodes[node1_ind]
        for node2 in G.adj[node1]:
            node2_ind = G_nodes_ind_map[node2]
            G_adj[node1_ind, node2_ind] = 1

    plt.matshow(G_adj)
    plt.title('adjacency matrix')
    plt.colorbar()
    plt.show()

    plt.hist(G_fcurv[np.triu_indices(G_fcurv.shape[0])].flatten(),
             bins=np.arange(np.nanmin(G_fcurv.flatten()),
                            1 + np.nanmax(G_fcurv.flatten())),
             rwidth=0.85,
             color='skyblue')  # ,edgecolor='black')
    # plt.title('histogram')
    plt.xlabel('Forman curvature')
    plt.ylabel('count')
    plt.show()

    plt.hist(G_fcurv_alt[np.triu_indices(G_fcurv_alt.shape[0])].flatten(),
             bins=np.arange(np.nanmin(G_fcurv_alt.flatten()),
                            1 + np.nanmax(G_fcurv_alt.flatten())),
             rwidth=0.85,
             color='skyblue')  # ,edgecolor='black')
    # plt.title('histogram')
    plt.xlabel('Forman aug. curvature')
    plt.ylabel('count')
    plt.show()

    plt.hist(LLY_curvatures[np.triu_indices(LLY_curvatures.shape[0])].flatten(),
             bins=40, rwidth=0.85, color='skyblue')  # ,edgecolor='black')
    # plt.title('histogram')
    plt.xlabel('LLY curvature')
    plt.ylabel('count')
    plt.show()


def curv_vs_time(graph_data, n_range, alpha):

    forman_times = []
    forman_alt_times = []
    lly_times = []

    for index in range(len(n_range)):
        n = n_range[index]

        if graph_data == 'chung_lu':

            G = make_chung_lu_clustering_graph(n, alpha=1., beta=.1, gamma=.0)

        elif graph_data == 'erdos':

            G = nx.erdos_renyi_graph(n, p=.003)

        else:
            raise NotImplementedError()

        start = time()

        _ = forman_curvature(G)

        end = time()
        forman_times.append(end - start)

        start = time()

        _ = forman_aug_curvature(G)

        end = time()
        forman_alt_times.append(end - start)

        start = time()

        _ = lly_curvature(G, alpha)

        end = time()
        lly_times.append(end - start)

    plt.loglog(n_range, forman_times,
               n_range, forman_alt_times,
               n_range, lly_times)

    plt.legend(['Forman', 'Forman Aug.', 'LLY'])

    plt.ylabel('Seconds')
    plt.xlabel('Number of nodes')

    plt.show()

    plt.semilogy(n_range, forman_times,
                 n_range, forman_alt_times,
                 n_range, lly_times)

    plt.legend(['Forman', 'Forman Aug.', 'LLY'])

    plt.ylabel('Seconds')
    plt.xlabel('Number of nodes')

    plt.show()

    plt.plot(n_range, forman_times,
             n_range, forman_alt_times,
             n_range, lly_times)

    plt.legend(['Forman', 'Forman Aug.', 'LLY'])

    plt.ylabel('Seconds')
    plt.xlabel('Number of nodes')

    plt.show()
