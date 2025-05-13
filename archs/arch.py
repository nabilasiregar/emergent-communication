import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.typing import Adj, OptTensor

class WeightedRGCNConv(RGCNConv):
    """
    One-shot vectorized RGCN: uses discrete direction via W_r and continuous distance scaling
    in two fused operations (batched matmul + scatter_add).
    """
    def forward(self,
                x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj,
                edge_type: Tensor,
                edge_weight: Tensor) -> Tensor:
        if isinstance(x, tuple):
            x_l, x_r = x
        else:
            x_l = x_r = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        source, target = edge_index
        N = x_l.size(0)

        # gather source node features for each edge
        xj = x_l[source] 

        # select weight matrix per edge according to its direction
        W_e = self.weight[edge_type]

        m = torch.einsum('ei,eio->eo', xj, W_e)

        # scale messages by continuous edge weight (distance)
        m = m * edge_weight.view(-1, 1)

        # scatter add messages to target nodes
        out = torch.zeros(N, self.out_channels, device=xj.device)
        out.scatter_add_(0, target.unsqueeze(-1).expand(-1, self.out_channels), m)

        if self.root is not None:
            out = out + (x_r @ self.root)
        if self.bias is not None:
            out = out + self.bias

        return out

class RGCN(nn.Module):
    def __init__(self, num_node_features, embedding_dim, num_relations, hidden_channels=None):
        super().__init__()
        hidden_channels = embedding_dim if hidden_channels is None else hidden_channels
        self.conv1 = WeightedRGCNConv(
            in_channels=num_node_features,
            out_channels=hidden_channels,
            num_relations=num_relations
        )
        self.conv2 = WeightedRGCNConv(
            in_channels=hidden_channels,
            out_channels=embedding_dim,
            num_relations=num_relations
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
