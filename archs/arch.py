import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, num_node_features, embedding_size, num_relations=8, num_layers=3):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_fc = nn.Linear(num_node_features, embedding_size)
        self.convs = nn.ModuleList([
            RGCNConv(embedding_size, embedding_size, num_relations)
            for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_type = edge_attr[:, 1].long()
        edge_distance = edge_attr[:, 0].float()
        h = F.relu(self.input_fc(x))
        
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_type))
        
        return h, edge_distance
