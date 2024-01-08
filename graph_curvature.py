import numpy as np
import networkx as nx
import ot


def graph_main_component(G):

    G_main_component = nx.Graph(G)

    G_cc = sorted(nx.connected_components(G), key=len, reverse=True)

    for i in range(1, len(G_cc)):
        component_set = G_cc[i]
        for node in list(component_set):
            G_main_component.remove_node(node)

    return G_main_component


def lly_curvature(G, alpha):

    if G.number_of_nodes() == 0:
        return np.zeros((0, 0))

    A = nx.adjacency_matrix(G)
    A = A.toarray()

    n = np.shape(A)[0]
    M_plus = np.zeros([n, n])
    M_minus = np.zeros([n, n])
    alpha_plus = 1
    alpha_minus = alpha  # 0.9

    for i in range(n):
        for j in range(n):
            if j == i:
                M_plus[i, j] = alpha_plus
                M_minus[i, j] = alpha_minus
            else:
                if A[i, j] == 1:
                    M_plus[i, j] = (1 - alpha_plus) / sum(A[i, :])
                    M_minus[i, j] = (1 - alpha_minus) / sum(A[i, :])

    optimal_transport_distances_plus = np.zeros((n, n))
    OR_curvatures_plus = np.matrix(np.ones((n, n)) * np.inf)
    optimal_transport_distances_minus = np.zeros((n, n))
    OR_curvatures_minus = np.matrix(np.ones((n, n)) * np.inf)

    shortest_distances = nx.floyd_warshall_numpy(G)

    for i in range(n):
        for j in range(n):
            if A[i, j] == 1:
                optimal_transport_distances_plus[i, j] = ot.emd2(M_plus[i, :], M_plus[j, :], shortest_distances)
                OR_curvatures_plus[i, j] = 1 - optimal_transport_distances_plus[i, j] / shortest_distances[i, j]
                optimal_transport_distances_minus[i, j] = ot.emd2(M_minus[i, :], M_minus[j, :], shortest_distances)
                OR_curvatures_minus[i, j] = 1 - optimal_transport_distances_minus[i, j] / shortest_distances[i, j]

    LLY_curvatures = (OR_curvatures_plus - OR_curvatures_minus) / (alpha_minus - alpha_plus)

    return np.array(LLY_curvatures)


def forman_curvature(G):

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

    G_deg_np = np.array(np.sum(G_adj, axis=0)).flatten()

    G_fcurv = np.zeros((len(G_nodes), len(G_nodes)))
    G_fcurv[:] = np.nan

    for node1_ind in range(len(G_nodes)):
        node1 = G_nodes[node1_ind]
        for node2 in G.adj[node1]:
            node2_ind = G_nodes_ind_map[node2]
            G_fcurv[node1_ind, node2_ind] = 4 - G_deg_np[node1_ind] - G_deg_np[node2_ind]

    return G_fcurv


def forman_aug_curvature(G):

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

    G_neighbors = {}
    for node in G.nodes:
        G_neighbors[node] = np.array(G.adj[node])

    G_fcurv_alt = np.zeros((len(G_nodes), len(G_nodes)))
    G_fcurv_alt[:] = np.nan

    G_deg_np = np.array(np.sum(G_adj, axis=0)).flatten()

    for edge in G.edges:

        node1 = edge[0]
        node2 = edge[1]

        node1_ind = G_nodes_ind_map[node1]
        node2_ind = G_nodes_ind_map[node2]

        G_fcurv_alt[node1_ind, node2_ind] = forman_aug_curvature_edge(node1, node2,
                                                                      G_neighbors, G_nodes_ind_map, G_deg_np)

        G_fcurv_alt[node2_ind, node1_ind] = G_fcurv_alt[node1_ind, node2_ind]

    return G_fcurv_alt


def forman_aug_curvature_edge(node1, node2, G_neighbors, G_nodes_ind_map, G_deg_np):

    node1_ind = G_nodes_ind_map[node1]
    node2_ind = G_nodes_ind_map[node2]

    node1_nbhd_np = G_neighbors[node1]
    node2_nbhd_np = G_neighbors[node2]

    nbhd_intersection_size = np.intersect1d(node1_nbhd_np, node2_nbhd_np).shape[0]

    return 4 - G_deg_np[node1_ind] - G_deg_np[node2_ind] + 3 * nbhd_intersection_size
