import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch import RGCN
from torch.distributions import Categorical
import pdb

class BeeSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations, num_distance_bins=10):
        super().__init__()
        self.num_relations = num_relations
        self.num_distance_bins = num_distance_bins
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)
        self.init_fc = nn.Linear(2 * embedding_size + 1, hidden_size)

    def forward(self, x, aux_input):
        data = aux_input['data']
        nest_tensor = aux_input['nest_tensor']
        food_tensor = aux_input['food_tensor']


        node_reps, _ = self.rgcn(data)
        nest_embed = node_reps[nest_tensor]   # [batch, embedding_size]
        food_embed = node_reps[food_tensor]   # [batch, embedding_size]
        distance = torch.norm(food_embed - nest_embed, dim=-1, keepdim=True)

        init_vec = torch.cat([nest_embed, food_embed, distance], dim=-1)
        hidden = torch.tanh(self.init_fc(init_vec))  # [batch, hidden_size]
        
        return hidden
    
class HumanSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations, vocab_size, max_len):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)
        self.fc_hidden = nn.Linear(2 * embedding_size, hidden_size)
        self.max_len = max_len
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, sender_input, aux_input):
        data, nest_tensor, food_tensor = aux_input['data'], aux_input['nest_tensor'], aux_input['food_tensor']
        h, _ = self.rgcn(data)

        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = F.relu(self.fc_hidden(combined))

        hidden_seq = hidden.unsqueeze(1).repeat(1, self.max_len, 1)
        logits = self.fc_out(hidden_seq)

        return logits

class Receiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size, num_relations):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, receiver_input, aux_input):
        data = aux_input['data']
        h, _ = self.rgcn(data)

        message = message.long()
        message_emb = self.embed(message)

        _, hidden = self.rnn(message_emb)
        hidden = hidden.squeeze(0)

        hidden_proj = self.fc_output(hidden)
        logits = (h * hidden_proj[data.batch]).sum(dim=1)

        batch_indices = data.batch
        num_graphs = batch_indices.max().item() + 1
        logits_per_graph = torch.full((num_graphs, h.size(0)), float('-inf'), device=h.device)
        logits_per_graph[batch_indices, torch.arange(h.size(0))] = logits

        dist = Categorical(logits=logits_per_graph)
        sample = dist.sample() 
        log_probs = dist.log_prob(sample)
        entropy = dist.entropy()

        return sample, log_probs, entropy
