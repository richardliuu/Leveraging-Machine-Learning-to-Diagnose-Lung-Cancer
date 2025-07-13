import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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

class GNNExplainer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(1234567)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x
        
model = GNNExplainer(in_channels=38, hidden_channels=64, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    out = model(data)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss

def test(loss):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    acc = (pred == data.y).sum().item() / data.num_nodes
    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return acc

for epoch in range(25):
    loss = train()
    test(loss=loss)


model.eval()  # Set model to evaluation mode

# Forward pass (usually gives output for all nodes)
out = model(data)

# Let's say you want output for node with index `node_idx`
node_idx = 42  # or any index you want to inspect

# Output vector (logits) for that node
single_output = out[node_idx]

# Convert to predicted class
predicted_class = single_output.argmax(dim=0)

print("Logits:", single_output)
print("Predicted class:", predicted_class.item())
