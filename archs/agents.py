import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch import RGCN
from vector_quantize_pytorch import VectorQuantize

class BeeSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size,
                 direction_vocab_size, num_relations):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations=num_relations)
        self.fc = nn.Linear(2 * embedding_size, hidden_size)
        
        # token a: discrete direction
        self.direction_head = nn.Linear(hidden_size, direction_vocab_size)
        # token b: continuous distance
        self.distance_head = nn.Linear(hidden_size, 1)

    def forward(self, x, _aux_input):
        data, nest_tensor, food_tensor = _aux_input
        h = self.rgcn(data)
        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = F.relu(self.fc(combined))

        token_a_logit = self.direction_head(hidden)
        token_b_continuous = self.distance_head(hidden)

        return F.log_softmax(token_a_logit, dim=-1), token_b_continuous


class HumanSender(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size, num_relations, max_len=6):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations=num_relations)
        self.token_decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc_vocab = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len

    def forward(self, x, _aux_input):
        data, nest_tensor, food_tensor = _aux_input
        h = self.rgcn(data)
        nest_embed = h[nest_tensor]
        food_embed = h[food_tensor]

        combined = torch.cat([nest_embed, food_embed], dim=-1)
        hidden = combined.unsqueeze(1).repeat(1, self.max_len, 1)

        outputs, _ = self.token_decoder(hidden)
        logits = self.fc_vocab(outputs)

        return F.log_softmax(logits, dim=-1)


class Receiver(nn.Module):
    def __init__(self, num_node_features, embedding_size, hidden_size, vocab_size, num_relations=2):
        super().__init__()
        self.rgcn = RGCN(num_node_features, embedding_size, num_relations=num_relations)
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, x, _aux_input):
        data, nest_tensor, _ = _aux_input
        h = self.rgcn(data)
        nest_embed = h[nest_tensor]

        # message: [batch, seq_len] discrete tokens
        msg_embed = self.token_embedding(message.argmax(dim=-1))
        rnn_out, _ = self.rnn(msg_embed)
        hidden = self.fc_hidden(rnn_out[:, -1, :])

        logits = torch.matmul(h, hidden.unsqueeze(-1)).squeeze(-1)
        return F.log_softmax(logits, dim=-1)


