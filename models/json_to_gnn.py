import json
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

with open('data/gnn_data.json', "r") as f:
    json_data = json.load(f)

node_features = []
true_labels = []

node_features = [list(entry['features'].values()) + [entry['predicted_label']] for entry in json_data]
node_features = np.array(node_features)

# Reducing complexity of the graph 

k = 10

knn = NearestNeighbors(n_neighbors = k+1, metric='cosine')
knn.fit(node_features)
neighbors = knn.kneighbors(node_features, return_distance=False)

# Building the edges (relationships)

sim_matrix = cosine_similarity(np.array(node_features))
edge_index = []

# Explanation on the threshold required 

for x in range(len(node_features)):
    for y in neighbors[1][1:]:
        edge_index.append([x, y])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor([entry['true_label'] for entry in json_data], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print(data)