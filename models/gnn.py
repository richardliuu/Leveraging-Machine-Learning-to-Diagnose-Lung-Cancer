import networkx as nx 
import matplotlib.pyplot as plt 

G = nx.DiGraph()

G.add_nodes_from([
    (0, {'color': 'blue', 'size': 250}),
    (1, {'color': 'yellow', 'size': 400}),
    (2, {'color': 'orange', 'size': 150}),
    (3, {'color': 'red', 'size': 600}),
])

G.add_edges_from([
    (0, 1),
    (1, 2),
    (1, 0),
    (1, 3),
    (2, 3),
    (3, 0),
])

node_colours = nx.get_node_attributes(G, "color").values()
colours = list(node_colours)
node_sizes = nx.get_node_attributes(G, "size").values()
sizes = list(node_sizes)

nx.draw(G, with_labels=True, node_color=colours, node_size=sizes)

plt.show()