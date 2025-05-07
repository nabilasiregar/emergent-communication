import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch import RGCN
from helpers import strip_node_types
from torch.distributions import Categorical
import pdb

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, num_rel: int, num_distance_bins: int):
        super().__init__()
        self.rgcn = RGCN(
            in_dim, hid_dim, num_rel, num_distance_bins,
            distance_embedding_dim=8,
            num_bases=None
        )

    def forward(self, data, x_override=None):
        return self.rgcn(data, x_override)
    
class BeeSender(nn.Module):
    def __init__(self, n_features, embedding_dim, hidden_dim, n_relations):
        super().__init__()
        self.encoder = RGCN(n_features, embedding_dim, n_relations)
        # vector that gives relative displacement
        self.fc = nn.Linear(2 * embedding_dim, hidden_dim)
        self.direction_head = nn.Linear(hidden_dim, n_relations)
        self.continuous_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, aux_input):
        data = aux_input["data"]
        # sender agent knows both nest and food positions
        nest, food = aux_input["nest_tensor"], aux_input["food_tensor"]
        node = self.encoder(data)
        # like calculating a distance from nest node to food node 
        h = torch.tanh(self.fc(torch.cat([node[nest], node[food]], -1)))
        logits = self.direction_head(h)
        mu, logvar = self.continuous_head(h).chunk(2, -1)
        return {"direction_logits": logits, "mu": mu, "logvar": logvar}

class BeeReceiver(nn.Module):
    """
    Expects message [direction, distance] from sender agent. Scores every node relative to the nest.
    """
    def __init__(self, n_features, embedding_dim, n_relations, keep_dims=()):
        super().__init__()
        self.encoder = RGCN(n_features, embedding_dim, n_relations)
        self.direction_embedding = nn.Embedding(n_relations, embedding_dim)
        self.distance_fc = nn.Linear(1, embedding_dim)
        self.merge = nn.Linear(2 * embedding_dim, embedding_dim)
        self.keep_dims = keep_dims
    
    def forward(self, message, _receiver_input, aux_input):
        data = aux_input["data"]
        nest = aux_input["nest_tensor"]
        direction_soft, distance = message[:, :-1], message[:, -1].unsqueeze(-1)

        direction_vec  = direction_soft @ self.direction_embedding.weight
        distance_vec = F.relu(self.distance_fc(distance))
        message_vec  = self.merge(torch.cat([direction_vec, distance_vec], -1))

        x_clean = strip_node_types(data.x, self.keep_dims) 
        node = self.encoder(data, x_override=x_clean)

        relative = node - node[nest][data.batch]
        scores = (relative * message_vec[data.batch]).sum(-1)

        parts = [scores[data.batch == g] for g in range(data.batch.max()+1)]
        logits = nn.utils.rnn.pad_sequence(parts, batch_first=True,
                                           padding_value=-float("inf"))
        return F.log_softmax(logits, -1)

class HumanSender(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, hidden_size, num_rel, num_distance_bins):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim, embed_dim, num_rel, num_distance_bins)
        self.fc  = nn.Linear(2 * embed_dim, hidden_size)

    def forward(self, x, aux_input):
        data       = aux_input["data"]
        nest_id   = aux_input["nest_tensor"]
        food_id   = aux_input["food_tensor"]

        node_emb   = self.encoder(data)
        nest_emb   = node_emb[nest_id]
        food_emb   = node_emb[food_id]

        h0 = torch.tanh(self.fc(torch.cat([nest_emb, food_emb], dim=-1)))
        return h0

class HumanReceiver(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        embed_dim,
        hidden_size,
        num_rel,
        num_distance_bins,
        keep_dims=(),
    ):
        super().__init__()
        self.keep_dims = keep_dims
        self.encoder = GraphEncoder(node_feat_dim, embed_dim, num_rel, num_distance_bins)
        self.message_fc = nn.Linear(hidden_size, embed_dim)

    @staticmethod
    def _pad_scores(scores: torch.Tensor, batch: torch.Tensor):
        B = int(batch.max().item()) + 1
        parts = [scores[batch == g] for g in range(B)]
        return nn.utils.rnn.pad_sequence(
            parts, batch_first=True, padding_value=float("-inf")
        )

    def forward(self, x, _receiver_input, aux_input):
        data       = aux_input["data"]
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