import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
print(f"Dataset: {dataset}")

print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes {dataset.num_classes}")

data = dataset[0]
print(data)

class GCN(torch.nn.module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manuel_seed(1234567)

        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)

            return x 
        
model = GCN(hidden_channels=16)
print(model)
