import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch import RGCN
from helpers import strip_node_types
from torch.distributions import Categorical
from typing import Optional
import pdb
    
class BeeSender(nn.Module):
    def __init__(self, n_features: int, embedding_dim: int, hidden_dim: int, n_relations: int):
        super().__init__()
        self.encoder = RGCN(
            num_node_features=n_features,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_relations=n_relations
        )
        self.fc = nn.Linear(2 * embedding_dim, hidden_dim)

    def forward(self, x, aux_input):
        data, nest, food = aux_input["data"], aux_input["nest_tensor"], aux_input["food_tensor"]
        # ablation: zeroed-out distance
        # data.edge_attr[:, 0] = 0
        # ablation: bin distance
        distances = data.edge_attr[:, 0]
        min_dist, max_dist = distances.min(), distances.max()
        bin_size = (max_dist - min_dist) / 3
        binned_distances = torch.floor((distances - min_dist) / bin_size)
        # to handle edge case where distance == max_dist
        binned_distances = torch.clamp(binned_distances, 0, 2)
        node = self.encoder(data)
        h    = torch.tanh(self.fc(torch.cat([node[nest], node[food]], dim=-1)))
        return h 

class BeeReceiver(nn.Module):
    def __init__(self, n_features: int, embedding_dim: int, hidden_dim: int, n_relations: int,
                keep_dims: tuple = (), gnn_num_bases: Optional[int] = None):
        super().__init__()
        self.encoder   = RGCN(num_node_features=n_features,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_relations=n_relations,
            num_bases=gnn_num_bases
            )
        self.merge     = nn.Linear(hidden_dim, embedding_dim)
        self.keep_dims = keep_dims
    
    def forward(self, message, _receiver_input, aux_input):
        data = aux_input["data"]
        nest = aux_input["nest_tensor"]
        message_vec = torch.relu(self.merge(message))

        # ablation: zeroed-out distance
        # data.edge_attr[:, 0] = 0

        # ablation: bin distance
        distances = data.edge_attr[:, 0]
        min_dist, max_dist = distances.min(), distances.max()
        bin_size = (max_dist - min_dist) / 3
        binned_distances = torch.floor((distances - min_dist) / bin_size)
        # to handle edge case where distance == max_dist
        binned_distances = torch.clamp(binned_distances, 0, 2)
        
        data.edge_attr[:, 0] = binned_distances

        x_clean = strip_node_types(data.x, self.keep_dims)
        node = self.encoder(data, x_override=x_clean)

        relative = node - node[nest][data.batch]
        scores = (relative * message_vec[data.batch]).sum(-1)

        B = int(data.batch.max().item()) + 1
        N = scores.size(0) // B
        logits = scores.view(B, N)
        return F.log_softmax(logits, -1)

class HumanSender(nn.Module):
    def __init__(self, node_feat_dim: int, embed_dim: int, hidden_size: int, num_rel: int,
                 gnn_num_bases: Optional[int] = None):
        super().__init__()
        self.encoder = RGCN(num_node_features=node_feat_dim,
            hidden_channels=hidden_size,
            out_channels=embed_dim,
            num_relations=num_rel,
            num_bases=gnn_num_bases)
        self.fc  = nn.Linear(2 * embed_dim, hidden_size)

    def forward(self, x, aux_input):
        data       = aux_input["data"]
        nest_id   = aux_input["nest_tensor"]
        food_id   = aux_input["food_tensor"]

        # ablation: zeroed-out distance
        # data.edge_attr[:, 0] = 0
        # ablation: bin distance
        distances = data.edge_attr[:, 0]
        min_dist, max_dist = distances.min(), distances.max()
        bin_size = (max_dist - min_dist) / 3
        binned_distances = torch.floor((distances - min_dist) / bin_size)
        # to handle edge case where distance == max_dist
        binned_distances = torch.clamp(binned_distances, 0, 2)

        node_emb   = self.encoder(data)
        nest_emb   = node_emb[nest_id]
        food_emb   = node_emb[food_id]

        h0 = torch.tanh(self.fc(torch.cat([nest_emb, food_emb], dim=-1)))
        return h0

class HumanReceiver(nn.Module):
    def __init__(self, node_feat_dim: int, embed_dim: int, hidden_size: int, num_rel: int,
                 keep_dims: tuple = (), gnn_num_bases: Optional[int] = None):
        super().__init__()
        self.keep_dims = keep_dims
        self.encoder = RGCN(num_node_features=node_feat_dim,
            hidden_channels=hidden_size,
            out_channels=embed_dim,
            num_relations=num_rel,
            num_bases=gnn_num_bases)
        self.message_fc = nn.Linear(hidden_size, embed_dim)

    @staticmethod
    def _pad_scores(scores: torch.Tensor, batch: torch.Tensor):
        B = int(batch.max().item()) + 1
        N = scores.numel() // B
        return scores.view(B, N)

    def forward(self, x, _receiver_input, aux_input):
        data       = aux_input["data"]
        # ablation: zeroed-out distance
        # data.edge_attr[:, 0] = 0
        # ablation: bin distance
        distances = data.edge_attr[:, 0]
        min_dist, max_dist = distances.min(), distances.max()
        bin_size = (max_dist - min_dist) / 3
        binned_distances = torch.floor((distances - min_dist) / bin_size)
        # to handle edge case where distance == max_dist
        binned_distances = torch.clamp(binned_distances, 0, 2)
        # receiver knows nest position
        nest_id   = aux_input["nest_tensor"]
        # but not other node types
        x_clean    = strip_node_types(data.x, self.keep_dims)
        node_emb   = self.encoder(data, x_override=x_clean)
        nest_emb   = node_emb[nest_id]
        # everything is calculated relative to nest node
        rel_node   = node_emb - nest_emb[data.batch]

        message_emb    = self.message_fc(x)

        scores     = (rel_node * message_emb[data.batch]).sum(-1)
        logits     = self._pad_scores(scores, data.batch)
        return F.log_softmax(logits, dim=-1) 