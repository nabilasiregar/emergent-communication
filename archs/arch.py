import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index
from torch_geometric.index import index2ptr
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
import pdb
class WeightedRGCNConv(RGCNConv):
    """
    Same as RGCNConv but also takes a continuous edge_weight per edge.
    We override forward method to pass edge_weight into propagate,
    and override message method to scale the message by edge_weight.
    """
    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index: Adj, edge_type: OptTensor = None, edge_weight: Optional[Tensor] = None) -> Tensor:
        if edge_weight is None:
            return super().forward(x, edge_index, edge_type)

        if isinstance(x, tuple):
            x_l, x_r = x
        else:
            x_l = x_r = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)
        
        if self.num_blocks is not None:
            if not torch.is_floating_point(x_r):
                raise ValueError("Block-diag decomposition needs float inputs.")

            for i in range(self.num_relations):
                mask = edge_type == i
                if mask.sum() == 0:
                    continue
                sub_edge = masked_edge_index(edge_index, mask)
                h = self.propagate(
                    sub_edge,
                    x=x_l,
                    edge_weight=edge_weight[mask],
                    size=size,
                )
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum("abc,bcd->abd", h, weight[i])
                out += h.reshape(-1, self.out_channels)
        else:
            for i in range(self.num_relations):
                mask = edge_type == i
                if mask.sum() == 0:
                    continue
                sub_edge = masked_edge_index(edge_index, mask)

                if not torch.is_floating_point(x_r):
                    out += self.propagate(
                        sub_edge,
                        x=weight[i, x_l],
                        edge_weight=edge_weight[mask],
                        size=size,
                    )
                else:
                    h = self.propagate(
                        sub_edge,
                        x=x_l,
                        edge_weight=edge_weight[mask],
                        size=size,
                    )
                    out += h @ weight[i]

        if self.root is not None:
            out += self.root[x_r] if not torch.is_floating_point(x_r) else x_r @ self.root
        if self.bias is not None:
            out += self.bias

        return out
    
    def message(self, x_j: Tensor, edge_weight: Optional[Tensor] = None, edge_type_ptr: OptTensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
class RGCN(nn.Module):
    def __init__(self, num_node_features, embedding_dim, num_relations, hidden_channels=None):
        super(RGCN, self).__init__()
        hidden_channels = embedding_dim if hidden_channels is None else hidden_channels
        self.conv1 = WeightedRGCNConv(in_channels=num_node_features, out_channels=hidden_channels, num_relations=num_relations)
        self.conv2 = WeightedRGCNConv(in_channels=hidden_channels, out_channels=embedding_dim, num_relations=num_relations)

    def forward(self, data, x_override=None):
        x           = data.x if x_override is None else x_override
        edge_index  = data.edge_index
        edge_weight = data.edge_attr[:, 0]
        edge_type   = data.edge_attr[:, 1].long()
        # normalize weights, 1e-8 to avoid null division
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)

        h = F.relu(self.conv1(x, edge_index, edge_type, edge_weight))
        h = self.conv2(h, edge_index, edge_type, edge_weight)
        return h