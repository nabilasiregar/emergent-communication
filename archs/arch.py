import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, NNConv
import pdb

class GNN(nn.Module):
    def __init__(self, num_node_features, embedding_size, num_relations=8, num_layers=5):
        super().__init__()
        self.num_relations = num_relations
        self.input_fc = nn.Linear(num_node_features, embedding_size)
        self.rgcn_convs = nn.ModuleList([
            RGCNConv(embedding_size, embedding_size, num_relations)
            for _ in range(num_layers)
        ])

        self.edge_mlp = nn.Sequential(
            nn.Linear(1 + num_relations, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size * embedding_size)
        )

        self.nnconv_layers = nn.ModuleList(
            NNConv(embedding_size,embedding_size, self.edge_mlp, aggr='mean')
            for _ in range(num_layers)
        )

    def forward(self, data, x_override=None):
        x = x_override if x_override is not None else data.x
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if x_override is not None:
            assert torch.equal(x, data.x) is False 
        edge_type = edge_attr[:, 1].long()
        edge_distance = edge_attr[:, 0].unsqueeze(-1)
        dir_onehot  = F.one_hot(edge_type, self.num_relations).float()
        edge_attr_c = torch.cat([edge_distance, dir_onehot], dim=1) 
        h = F.relu(self.input_fc(x))

        for rgcn, nnconv in zip(self.rgcn_convs, self.nnconv_layers):
            h_disc = F.relu(rgcn(h, edge_index, edge_type))
            h_cont = F.relu(nnconv(h, edge_index, edge_attr_c))
            h      = h + h_disc + h_cont
        
        return h
