import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool

class SenderAgent(nn.Module):
    def __init__(self, feat_size, embedding_size, hidden_size, vocab_size, num_relations=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.rgcn1 = RGCNConv(feat_size, embedding_size, num_relations=num_relations)
        self.rgcn2 = RGCNConv(embedding_size, hidden_size, num_relations=num_relations)

        self.init_h = nn.Linear(hidden_size, hidden_size)
        self.init_c = nn.Linear(hidden_size, hidden_size)

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, prev_symbol, prev_state=None, _aux_input=None):
        if _aux_input is None:
            raise ValueError("No aux input!")
        if prev_state is None:
            graph_data = _aux_input["graph_data"]
            x = graph_data.x
            edge_index = graph_data.edge_index
            batch = getattr(graph_data, 'batch', None)

            h = self.rgcn1(x, edge_index)
            h = F.relu(h)
            h = self.rgcn2(h, edge_index)
            h = F.relu(h)

            if batch is not None:
                pooled = global_mean_pool(h, batch)
            else:
                pooled = h.mean(dim=0, keepdim=True)

            hidden_0 = self.init_h(pooled)
            cell_0 = self.init_c(pooled)
            prev_state = (hidden_0, cell_0)

        (hidden, cell) = prev_state
        # produce discrete token distribution
        logits = self.output_layer(hidden)
        logits = F.log_softmax(logits, dim=1)

        return logits, (hidden, cell)


class ReceiverAgent(nn.Module):
    def __init__(self, feat_size, embedding_size, hidden_size, vocab_size, num_relations=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.rgcn1 = RGCNConv(feat_size, embedding_size, num_relations=num_relations)
        self.rgcn2 = RGCNConv(embedding_size, embedding_size, num_relations=num_relations)

        self.hidden_proj = nn.Linear(hidden_size, embedding_size)

    def forward(self, last_hidden, _input=None, _aux_input=None):
        graph_data = _aux_input["graph_data"]
        x = graph_data.x
        edge_index = graph_data.edge_index

        # node embeddings
        h = self.rgcn1(x, edge_index)
        h = F.relu(h)
        h = self.rgcn2(h, edge_index)
        h = F.relu(h)  # shape [num_nodes, embedding_size]

        message_embed = self.hidden_proj(last_hidden)  # shape [batch_size, embedding_size]
        message_embed = message_embed.squeeze(0) 

        # dot product each node embedding with the message
        scores = torch.matmul(h, message_embed.T)
        log_probs = F.log_softmax(scores, dim=0)
        return log_probs
