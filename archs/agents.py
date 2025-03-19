import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from archs.arch import RGCN
import pdb

class BeeSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations=num_relations)
        self.fc = nn.Linear(2 * embedding_size, hidden_size)

    def forward(self, x, _aux_input):
        x = _aux_input
        data = x['data']
        nest_tensor = x['nest_tensor']
        food_tensor = x['food_tensor']

        h = self.rgcn(data)
        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = F.relu(self.fc(combined))
        
        return hidden


class HumanSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, num_relations):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations=num_relations)
        self.fc = nn.Linear(2 * embedding_size, hidden_size)

    def forward(self, x, _aux_input):
        x = _aux_input
        data = x['data']
        nest_tensor = x['nest_tensor']
        food_tensor = x['food_tensor']

        h = self.rgcn(data)
        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = combined.unsqueeze(1).repeat(1, self.max_len, 1)

        return hidden


class Receiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size,
                 num_relations, communication='human'):
        super().__init__()
        self.communication = communication
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations=num_relations)
        # self.fc_msg = nn.Linear(embedding_size, embedding_size)

    def forward(self, message, x, _aux_input):
        data = _aux_input['data']
        node_embeddings = self.rgcn(data)
        # message_representation = self.fc_msg(message)
        logits = torch.matmul(node_embeddings, message.unsqueeze(-1)).squeeze(-1)
        distribution = Categorical(logits=logits)
        sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        entropy = distribution.entropy()
        return sample, logits, entropy