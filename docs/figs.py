import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, '..')
from graph_curvature import forman_curvature  # noqa: E402

G = nx.Graph(np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 0],
                       [1, 1, 0, 0]]))
pos = nx.spring_layout(G)
nx.draw(G, pos=pos, node_color='#ADD8E6')
plt.show()

fcurvature_edges, fcurvature_nodes = forman_curvature(G)
print(fcurvature_edges)
print(fcurvature_nodes)

G.edges[(0, 1)]['curvature'] = -2
G.edges[(0, 2)]['curvature'] = -1
G.edges[(0, 3)]['curvature'] = -1
G.edges[(1, 2)]['curvature'] = -1
G.edges[(1, 3)]['curvature'] = -1
edge_labels = {}
for e in G.edges:
    edge_labels[e] = G.edges[e]['curvature']
labels = {0: -1.33,
          1: -1.33,
          2: -1,
          3: -1}
nodes = nx.draw_networkx_nodes(G, pos=pos, node_color='#ADD8E6')
labels = nx.draw_networkx_labels(G, pos=pos, labels=labels)
edges = nx.draw_networkx_edges(G, pos=pos)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
