import torch
from torch.utils.data import DataLoader
from graph.dataset import collate_fn
from archs.agents import BeeSender, Receiver

data = torch.load("data/graph_dataset.pt")
train_set = data['train_set']
val_set = data['val_set']

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)

num_node_features = 3
embedding_size = 64
hidden_size = 64
direction_vocab_size = 8  # for token a
num_relations = 2        
vocab_size = 10  # for receiver        

sender = BeeSender(num_node_features, embedding_size, hidden_size,
                   direction_vocab_size, num_relations)
receiver = Receiver(num_node_features, embedding_size, hidden_size, vocab_size, num_relations)

for batched_data, nest_tensor, food_tensor in train_loader:
    aux_input = (batched_data, nest_tensor, food_tensor)
    
    token_a, token_b = sender(None, aux_input)
    
    print("Token direction:", token_a)
    print("Token distance:", token_b)
    print("Food node indices:", food_tensor)
    
    break
