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

    def forward(self, h_t, x, aux_input):
        data       = aux_input["data"]
        nest_id   = aux_input["nest_tensor"]

        x_clean    = strip_node_types(data.x, self.keep_dims)
        node_emb   = self.encoder(data, x_override=x_clean)
        nest_emb   = node_emb[nest_id]
        rel_node   = node_emb - nest_emb[data.batch]

        message_emb    = self.message_fc(h_t)

        scores     = (rel_node * message_emb[data.batch]).sum(-1)
        logits     = self._pad_scores(scores, data.batch)
        return F.log_softmax(logits, dim=-1) 