import torch
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import networkx as nx
from environment import Environment

def graph_to_data(graph):
    """
    Convert a NetworkX graph from Environment to a PyG Data object
    
    Node Features:
      - [type_one_hot(3), x_position, y_position]
        => total of 5 features per node
    
    Edge Index: 
      - undirected, so each edge (u, v) is stored twice (u->v and v->u)

    Edge Attributes:
      - None (we do not store distance/direction on edges, so agent
        can compute them itself)
    """
    num_nodes = graph.number_of_nodes()
    
    type_mapping = {
        'Nest':        [1, 0, 0],
        'Food':        [0, 1, 0],
        'Distractor':  [0, 0, 1]
    }
    
    node_features = []
    for node_id in range(num_nodes):
        node_data = graph.nodes[node_id]
        node_type = node_data.get('type', 'Distractor')
        type_feat = type_mapping[node_type]
        
        position = node_data.get('position', (0, 0))
        x_pos, y_pos = float(position[0]), float(position[1])
        
        node_feat = type_feat + [x_pos, y_pos]
        node_features.append(node_feat)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    src_nodes = []
    dst_nodes = []
    for u, v in graph.edges():
        src_nodes.append(u)
        dst_nodes.append(v)
        src_nodes.append(v)
        dst_nodes.append(u)
    
    # shape: [2, num_edges*2]
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    
    edge_attr = None
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def create_dataset(num_graphs=10, num_distractors=4, connect_threshold=50.0):
    dataset = []
    for i in range(num_graphs):
        env = Environment(num_distractors=num_distractors,
                          connect_threshold=connect_threshold,
                          plot=False)
        network = env.graph
        data = graph_to_data(network)
        dataset.append(data)
    return dataset

if __name__ == '__main__':
    full_dataset = create_dataset(num_graphs=50, num_distractors=3)
    
    train_size = 40
    validation_size = 10
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader   = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    
    for batch in train_loader:
        print(batch)
        break