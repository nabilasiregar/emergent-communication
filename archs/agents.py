import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch import GNN
from torch.distributions import Categorical
import pdb

def strip_node_types(x, keep_dims):
    out = torch.zeros_like(x)
    out[:, keep_dims] = x[:, keep_dims]
    return out
class BeeSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations):
        super(BeeSender, self).__init__()
        self.gnn = GNN(num_node_features, embedding_size, num_relations)

        # to process nest/food embeddings + distance
        self.init_fc = nn.Linear(2 * embedding_size + 1, hidden_size)
        # for discrete token
        self.discrete_head = nn.Linear(hidden_size, num_relations)
        # for continuous token
        self.continuous_head = nn.Linear(hidden_size, 2)

    def forward(self, x, aux_input):
        data = aux_input['data']
        nest_tensor = aux_input['nest_tensor']
        food_tensor = aux_input['food_tensor']

        h = self.gnn(data)
        nest, food = h[nest_tensor], h[food_tensor]

        latent_distance = torch.norm(food - nest, dim=-1, keepdim=True)
        hidden = torch.tanh(self.init_fc(torch.cat([nest, food, latent_distance], dim=-1)))
        
        discrete_logits = self.discrete_head(hidden)
        mu, logvar = self.continuous_head(hidden).chunk(2, dim=-1)
        return {'discrete_logits': discrete_logits, 'mu': mu, 'logvar': logvar}

class BeeReceiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, vocab_size, num_relations, keep_dims=[]):
        super().__init__()
        self.keep_dims = keep_dims
        self.gnn = GNN(num_node_features, embedding_size, num_relations)
        self.discrete_embed = nn.Embedding(vocab_size, embedding_size)
        self.continuous_fc = nn.Linear(1, embedding_size)
        self.message_fc = nn.Linear(2 * embedding_size, embedding_size)

    def forward(self, message, _, aux_input):
        data = aux_input['data']
        x_stripped = strip_node_types(data.x, self.keep_dims)
        node_emb = self.gnn(data, x_override=x_stripped)

        discrete_emb = self.discrete_embed(message[:,0].long())
        continuous_emb = F.relu(self.continuous_fc(message[:,1].unsqueeze(1)))
        message_repr = self.message_fc(torch.cat([discrete_emb, continuous_emb], dim=-1))

        logits_per_graph = []
        for graph_idx in range(data.batch.max().item() + 1):
            node_indices = (data.batch == graph_idx).nonzero(as_tuple=True)[0]
            scores = node_emb[node_indices] @ message_repr[graph_idx].unsqueeze(-1)
            logits_per_graph.append(scores.squeeze(-1))

        logits = nn.utils.rnn.pad_sequence(
            logits_per_graph,
            batch_first=True,
            padding_value=-float('inf')
        )
        return logits
    
class HumanSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations, vocab_size, max_len):
        super().__init__()
        self.gnn = GNN(num_node_features, embedding_size, num_relations)
        self.fc_hidden = nn.Linear(2 * embedding_size, hidden_size)
        self.max_len = max_len

    def forward(self, x, aux_input):
        data, nest_tensor, food_tensor = aux_input['data'], aux_input['nest_tensor'], aux_input['food_tensor']
        h = self.gnn(data)

        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = F.relu(self.fc_hidden(combined))

        return hidden

class HumanReceiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size, num_relations, keep_dims):
        super().__init__()
        self.keep_dims = keep_dims
        self.gnn = GNN(num_node_features, embedding_size, num_relations)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, input, aux_input):
        data = aux_input['data']
        x_stripped = strip_node_types(data.x, self.keep_dims)
        h = self.gnn(data, x_stripped)
        
        message = message.long()
        message_emb = self.embed(message)

        _, hidden = self.rnn(message_emb)
        hidden = hidden.squeeze(0)

        hidden_proj = self.fc_output(hidden)

        # compute logits for each node per graph
        num_graphs = data.batch.max().item() + 1
        logits = []
        for i in range(num_graphs):
            nodes_in_graph = (data.batch == i).nonzero(as_tuple=True)[0]
            node_embeddings = h[nodes_in_graph]
            graph_logits = node_embeddings @ hidden_proj[i]
            logits.append(graph_logits)

        # pad logits to handle variable number of nodes per graph (future use)
        logits = nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=float('-inf'))

        categorical_distribution = Categorical(logits=logits)

        # sample one node index per graph
        sample = categorical_distribution.sample()
        log_prob = categorical_distribution.log_prob(sample)
        entropy = categorical_distribution.entropy()

        return sample, log_prob, entropy
