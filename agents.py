import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class SenderAgent(nn.Module):
    def __init__(self, gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
                 message_vocab_size, message_length):
        super().__init__()
        self.encoder = GraphEncoder(gnn_in_channels, gnn_hidden_channels, gnn_out_channels)
        # for a discrete channel, produce logits for each token in the vocabulary per message slot
        self.message_length = message_length
        self.fc = nn.Linear(gnn_out_channels, message_vocab_size * message_length)
        self.message_vocab_size = message_vocab_size

    def forward(self, data, labels=None):
        node_embeddings = self.encoder(data)  # shape: [num_nodes, gnn_out_channels]
        global_embedding = torch.mean(node_embeddings, dim=0)  # shape: [gnn_out_channels]
        message_logits = self.fc(global_embedding)  # shape: [message_vocab_size * message_length]
        message_logits = message_logits.view(self.message_length, self.message_vocab_size)
        return message_logits  # these logits can be turned into discrete tokens via Gumbel-Softmax

class ReceiverAgent(nn.Module):
    def __init__(self, gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
                 message_vocab_size, message_length, num_nodes):
        super().__init__()
        self.encoder = GraphEncoder(gnn_in_channels, gnn_hidden_channels, gnn_out_channels)
        # embedding layer for interpreting the discrete message tokens
        self.message_embedding = nn.Embedding(message_vocab_size, gnn_out_channels)
        self.message_length = message_length
        # final classifier: outputs logits over the nodes (to select Food among nodes)
        self.fc = nn.Linear(gnn_out_channels, num_nodes)

    def forward(self, data, message, labels=None):
        # encode the graph
        node_embeddings = self.encoder(data)
        global_graph_embedding = torch.mean(node_embeddings, dim=0)  # shape: [gnn_out_channels]
        # decode the message
        message_embeds = self.message_embedding(message)  # shape: [message_length, gnn_out_channels]
        # aggregate message embeddings
        message_rep = torch.mean(message_embeds, dim=0)  # shape: [gnn_out_channels]
        # combine the message representation with the global graph embedding
        combined = global_graph_embedding + message_rep
        # predict the food node: output logits for each node
        logits = self.fc(combined)  # shape: [num_nodes]
        return logits
