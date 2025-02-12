import torch
import torch.nn as nn
import torch.optim as optim
import egg.core as core
from agents import SenderAgent, ReceiverAgent
from helpers import convert_graph_to_tensors
from environment import Environment
from torch_geometric.data import Batch
import pdb

def loss_function(receiver_logits, target):
    criterion = nn.CrossEntropyLoss()
    # cross-entropy on receiver logits vs the target index
    return criterion(receiver_logits.unsqueeze(0), target)

if __name__ == '__main__':
    # ------------------------------
    # Hyperparameters and Setup
    # ------------------------------
    # GNN feature dimensions
    gnn_in_channels = 11 # 3 (node type one-hot) + 8 (direction one-hot)
    gnn_hidden_channels = 16
    gnn_out_channels = 16

    # Communication channel parameters
    message_vocab_size = 10   # vocab size for messages
    message_length = 5        # num of tokens per message

    # Hyperparameters
    lr = 0.001
    num_epochs = 20

    # Create the environment and convert its graph to a PyG Data object
    env = Environment(num_distractors=3)
    data = convert_graph_to_tensors(env.graph)
    print("Type of data:", type(data))

    batched_data = Batch.from_data_list([data])

    # Node 0 is the Nest, Node 1 is the Food (target)
    target = torch.tensor([1])
    num_nodes = env.graph.number_of_nodes()

    # ---------------
    # Gumbel-Softmax
    # ---------------
    sender = SenderAgent(gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
                         message_vocab_size, message_length)
    sender = core.GumbelSoftmaxWrapper(sender, temperature=1.0)
    
    receiver = ReceiverAgent(gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
                             message_vocab_size, message_length, num_nodes)
    receiver = core.SymbolReceiverWrapper(receiver, message_vocab_size, agent_input_size=gnn_out_channels)
    game = core.SymbolGameGS(sender, receiver, loss_function)

    # ------------
    # Reinforce
    # ------------
    # sender = SenderAgent(gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
    #                      message_vocab_size, message_length)
    # sender = core.ReinforceWrapper(sender)
    
    # receiver = ReceiverAgent(gnn_in_channels, gnn_hidden_channels, gnn_out_channels,
    #                          message_vocab_size, message_length, num_nodes)
    # receiver = core.SymbolReceiverWrapper(receiver, message_vocab_size, agent_input_size=gnn_out_channels)
    # receiver = core.ReinforceDeterministicWrapper(receiver)
    # game = core.SymbolGameReinforce(sender, receiver, loss_function,
    #                                 sender_entropy_coeff=0.05, receiver_entropy_coeff=0.0)
    # ------------------------------
    # Optimizer and Simple Training Loop
    # ------------------------------
    optimizer = optim.Adam(game.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        print("Before sending to game:", type(batched_data))
        receiver_logits, message_indices = game(batched_data, target)
        print("After game and before loss calculation:", type(batched_data))
        loss = loss_function(receiver_logits, target)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            print("Sender's message:", message_indices)
