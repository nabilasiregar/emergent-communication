import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch import RGCN
from torch.distributions import Categorical, Normal
from torch.nn.utils.rnn import pad_sequence
import pdb

def strip_node_types(x, keep_dims):
    if not keep_dims:                      
        return torch.zeros(x.size(0), 1, device=x.device)
    return x[:, keep_dims] 
class BeeSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations):
        super(BeeSender, self).__init__()
        self.num_relations = num_relations

        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)

        # to process nest/food embeddings + distance
        self.init_fc = nn.Linear(2 * embedding_size + 1, hidden_size)
        # for discrete token
        self.discrete_head = nn.Linear(hidden_size, num_relations)
        # for continuous token
        self.continuous_mu = nn.Linear(hidden_size, 1)
        self.continuous_logvar = nn.Linear(hidden_size, 1)

    def forward(self, x, aux_input):
        data = aux_input['data']
        nest_tensor = aux_input['nest_tensor']
        food_tensor = aux_input['food_tensor']

        node_representation, _ = self.rgcn(data)
        nest_embed = node_representation[nest_tensor]
        food_embed = node_representation[food_tensor] 
        distance = torch.norm(food_embed - nest_embed, dim=-1, keepdim=True)

        # hidden representation
        hidden_input = torch.cat([nest_embed, food_embed, distance], dim=-1)
        hidden = torch.tanh(self.init_fc(hidden_input))

        # discrete token
        direction_logits = self.discrete_head(hidden) 
        direction_dist = Categorical(logits=direction_logits)
        direction_token = direction_dist.sample().float().unsqueeze(1)
        direction_log_prob = direction_dist.log_prob(direction_token)
        direction_entropy = direction_dist.entropy()

        # continuous token
        mu = self.continuous_mu(hidden)
        logvar = self.continuous_logvar(hidden)
        std = torch.exp(0.5 * logvar)
        distance_dist = Normal(mu, std)
        distance_token = distance_dist.rsample()
        distance_log_prob = distance_dist.log_prob(distance_token).squeeze(1)
        distance_entropy = distance_dist.entropy().squeeze(1)

        message = torch.cat([direction_token, distance_token], dim=1)
        log_prob = direction_log_prob + distance_log_prob
        entropy = direction_entropy + distance_entropy
   
        return message, log_prob, entropy
    
class HumanSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations, vocab_size, max_len):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)
        self.fc_hidden = nn.Linear(2 * embedding_size, hidden_size)
        self.max_len = max_len

    def forward(self, x, aux_input):
        data, nest_tensor, food_tensor = aux_input['data'], aux_input['nest_tensor'], aux_input['food_tensor']
        h, _ = self.rgcn(data)

        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = F.relu(self.fc_hidden(combined))

        return hidden

class HumanReceiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size, num_relations, keep_dims):
        super().__init__()
        self.keep_dims = keep_dims
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, input, aux_input):
        data = aux_input['data']
        x_stripped = strip_node_types(data.x, self.keep_dims)
        h, _ = self.rgcn(data, x_stripped)
        
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

class BeeReceiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size, num_relations, keep_dims):
        super(BeeReceiver, self).__init__()
        self.keep_dims = keep_dims
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)
        self.discrete_embed = nn.Embedding(vocab_size, embedding_size)
        self.continuous_fc = nn.Linear(1, embedding_size)
        self.message_fc = nn.Linear(2 * embedding_size, hidden_size)
        
    def forward(self, message, x, _aux_input):
        data = _aux_input['data']
        x_stripped = strip_node_types(data.x, self.keep_dims)
        node_emb, _ = self.rgcn(data, x_stripped)
        
        discrete_token = message[:, 0].long() 
        discrete_emb = self.discrete_embed(discrete_token)
        continuous_token = message[:, 1].unsqueeze(1)
        continuous_emb = F.relu(self.continuous_fc(continuous_token))

        combined = torch.cat([discrete_emb, continuous_emb], dim=1)
        message_repr = F.relu(self.message_fc(combined))
        
        batch_size = data.batch.max().item() + 1
        scores = []
        for i in range(batch_size):
            nodes_in_graph = (data.batch == i).nonzero(as_tuple=True)[0]
            nodes_i = node_emb[nodes_in_graph]
            score_i = nodes_i @ message_repr[i].unsqueeze(1)
            scores.append(score_i.squeeze(1))
        
        # in case each graph has different number of nodes
        logits = pad_sequence(scores, batch_first=True, padding_value=float('-inf'))

        return logits