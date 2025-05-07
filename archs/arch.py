import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import pdb

class DistanceEmbedding(nn.Sequential):
    def __init__(self, hidden_dim: int = 32):
        super().__init__(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()
        )

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return super().forward(d).squeeze(-1)
    
class RGCN(nn.Module):
    """
    Edge attributes are expected in the form (distance, direction) where:
        - distance is a float (edge weight)
        - direction is an integer (edge type)
    """
    def __init__(self,
        node_feature_dim: int,
        hidden_feature_dim: int,
        num_relations: int,
        num_distance_bins: int = 3,
        distance_embedding_dim: int = 32,
        num_bases: int | None = None,
        aggr: str = "mean"
    ):
        super().__init__()

        self.num_distance_bins = num_distance_bins
        total_relations = num_relations * self.num_distance_bins

        self.conv1 = RGCNConv(
            in_channels=node_feature_dim,
            out_channels=hidden_feature_dim,
            num_relations=total_relations,
            num_bases=num_bases,
            aggr=aggr
        )

        self.conv2 = RGCNConv(
            in_channels=hidden_feature_dim,
            out_channels=hidden_feature_dim,
            num_relations=total_relations,
            num_bases=num_bases,
            aggr=aggr
        )

        self.distance_embedding = DistanceEmbedding(distance_embedding_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_feature_dim))
    
    @staticmethod
    def _split(edge_attr: torch.Tensor):
        distances = edge_attr[:, 0].unsqueeze(-1) 
        relation_types = edge_attr[:, 1].long()
        return distances, relation_types

    def forward(self, data, x_override=None):
        x = data.x if x_override is None else x_override
        distances, directions = self._split(data.edge_attr)
        distance_vals = distances.squeeze(-1).contiguous()
        max_d = float(distance_vals.max().item())
        edges = torch.linspace(0.0, max_d, self.num_distance_bins + 1,device=distance_vals.device)[1:-1]
        distance_bins = torch.bucketize(distance_vals, edges)
        combined_edge_info = directions * self.num_distance_bins + distance_bins
        # first hop
        x = F.relu(
            self.conv1(x, data.edge_index, edge_type=combined_edge_info)
        )

        # second hop
        x = self.conv2(x, data.edge_index, edge_type=combined_edge_info)
        return x + self.bias