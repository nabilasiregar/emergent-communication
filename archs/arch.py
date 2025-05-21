import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple, Union
from torch_geometric.nn import FastRGCNConv
from torch_geometric.typing import (
    Adj,
    OptTensor
)
import pdb

class WeightedFastRGCNConv(FastRGCNConv):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         num_relations=num_relations, num_bases=num_bases,
                         num_blocks=num_blocks, aggr=aggr, **kwargs)

    # override the forward method to accept edge_weight and pass it to propagate
    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj,
                edge_type: OptTensor = None,
                edge_weight: OptTensor = None):
        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        # propagate_type: (x: Tensor, edge_type: OptTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type,
                             edge_weight=edge_weight,
                             size=size)

        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_type: Tensor,
                edge_index_j: Tensor,
                edge_weight: OptTensor = None) -> Tensor:
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)
        if self.num_blocks is not None:
             raise NotImplementedError("Block-diagonal decomposition with this minimal WeightedFastRGCNConv "
                                       "requires more specific handling. Consider num_bases or no decomposition.")
        
        # No regularization/Basis-decomposition
        if not torch.is_floating_point(x_j):
            weight_index = edge_type * weight.size(1) + edge_index_j
            return weight.view(-1, self.out_channels)[weight_index]
        original_message = torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)
        
        if edge_weight is not None:
            if edge_weight.ndim == 1: # ensure broadcasting
                original_message = original_message * edge_weight.view(-1, 1)
            else:
                original_message = original_message * edge_weight
        return original_message 
class RGCN(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 aggr: str = 'mean'):
        super().__init__()
        
        self.conv1 = WeightedFastRGCNConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases,
            aggr=aggr
        )
        self.conv2 = WeightedFastRGCNConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=num_bases,
            aggr=aggr
        )
    
    def forward(self, data, x_override=None):
        x = data.x if x_override is None else x_override
        edge_index  = data.edge_index
        edge_weight = data.edge_attr[:, 0]
        edge_type   = data.edge_attr[:, 1].long()
        # normalize weights to [0,1]
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)

        h = F.relu(self.conv1(x, edge_index, edge_type, edge_weight))
        h = self.conv2(h, edge_index, edge_type, edge_weight)

        return h