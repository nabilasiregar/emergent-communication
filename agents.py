import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from data_loader import get_dataloader
from environment import Environment

class GraphSender(nn.Module):
    def __init__(self, feat_size, embedding_size, hidden_size, vocab_size=12, temp=1.0, edge_attr_dim=9):
        super(GraphSender, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.temp = temp

        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feat_size * embedding_size)
        )
        self.conv1 = NNConv(feat_size, embedding_size, self.edge_nn1, aggr='mean')

        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size * hidden_size)
        )
        self.conv2 = NNConv(embedding_size, hidden_size, self.edge_nn2, aggr='mean')

        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, _aux_input=None):
        graph = _aux_input["graph_data"]
        node_features = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        h = self.conv1(node_features, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)

        if hasattr(graph, "batch"):
            pooled = global_mean_pool(h, graph.batch)
        else:
            pooled = h.mean(dim=0, keepdim=True)

        logits = self.lin(pooled) / self.temp
        logits = F.log_softmax(logits, dim=1)
        return logits

class GraphReceiver(nn.Module):
    def __init__(self, feat_size, embedding_size, vocab_size, edge_attr_dim=9):
        super(GraphReceiver, self).__init__()
        self.embedding_size = embedding_size

        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feat_size * embedding_size)
        )
        self.conv1 = NNConv(feat_size, embedding_size, self.edge_nn1, aggr='mean')

        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size * embedding_size)
        )
        self.conv2 = NNConv(embedding_size, embedding_size, self.edge_nn2, aggr='mean')

        self.signal_embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, signal, x, _aux_input=None):
        graph = _aux_input["graph_data"]
        node_features = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        h = self.conv1(node_features, edge_index, edge_attr)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)  # h: (num_nodes, embedding_size)

        signal_emb = self.signal_embedding(signal)  # shape: (batch_size, embedding_size)

        if signal_emb.dim() == 2 and signal_emb.size(0) == 1:
            signal_emb = signal_emb.squeeze(0)

        scores = torch.matmul(h, signal_emb)  # (num_nodes,)
        log_probs = F.log_softmax(scores, dim=0)
        return log_probs

if __name__ == '__main__':
    env = Environment(num_distractors=2)
    # env.visualize_environment()
    pyg_data = get_dataloader(env.graph)

    aux_input = {"graph_data": pyg_data}

    feat_size = 3         # node/location types (one-hot)
    embedding_size = 16
    hidden_size = 32
    vocab_size = 12       # communication vocabulary size
    edge_attr_dim = 9     # [distance (1) + one-hot direction (8)]
    
    sender = GraphSender(feat_size, embedding_size, hidden_size, vocab_size, temp=1.0, edge_attr_dim=edge_attr_dim)
    receiver = GraphReceiver(feat_size, embedding_size, vocab_size, edge_attr_dim=edge_attr_dim)

    # --- Test Sender ---
    sender_output = sender(None, _aux_input=aux_input)
    print("Sender output (logits over vocabulary):")
    print(sender_output)

    # create a dummy signal for the receiver
    # by choosing the token with the highest probability from the sender
    sender_token = sender_output.argmax(dim=1)  # tensor of shape (batch_size,)
    print("\nSender token (argmax):", sender_token)

    # --- Test Receiver ---
    receiver_output = receiver(sender_token, None, _aux_input=aux_input)
    print("\nReceiver output (log_probs over nodes):")
    print(receiver_output)
