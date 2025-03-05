import math
import random
import networkx as nx
import torch
from torch.utils.data import DataLoader, random_split
from environment import Environment
import pdb

def prepare_dictionary(graph):
    """
    Convert the graph into an adjacency dictionary
    
    For each node, store:
      - type: Node type as a string ('Nest', 'Food', or 'Distractor')
      - neighbors: A list of tuples (neighbor_id, edge_attr) where edge_attr is a dict
                   containing "distance" and "direction"
    """
    adj_dict = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get("type", "Distractor")
        neighbors = []

        for neighbor in graph.neighbors(node):
            edge_data = graph[node][neighbor]
            attr = {
                "distance": edge_data.get("distance"),
                "direction": edge_data.get("direction")
            }
            neighbors.append((neighbor, attr))
        adj_dict[node] = {"type": node_type, "neighbors": neighbors}
    return adj_dict
    
def convert_dict_to_tensors(adj_dict, current_node_id=0):
    type_mapping = {'Nest': [1, 0, 0], 'Food': [0, 1, 0], 'Distractor': [0, 0, 1]}
    direction_mapping = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}
    
    node_type = adj_dict["type"][0] if isinstance(adj_dict["type"], list) else adj_dict["type"]
    features = [type_mapping.get(node_type, [0, 0, 1])]
    
    edge_index = [[], []]
    edge_attrs = []
    
    for neighbor in adj_dict.get("neighbors", []):
        neighbor_id_tensor, attr = neighbor
        
        if isinstance(neighbor_id_tensor, torch.Tensor):
            neighbor_id = int(neighbor_id_tensor.item())
        else:
            neighbor_id = int(neighbor_id_tensor)
        edge_index[0].append(current_node_id)
        edge_index[1].append(neighbor_id)
        
        if torch.is_tensor(attr["distance"]):
            distance = attr["distance"][0].item() if attr["distance"].dim() > 0 else attr["distance"].item()
        else:
            distance = attr["distance"]
        direction_str = attr["direction"][0] if isinstance(attr["direction"], list) else attr["direction"]
        direction = direction_mapping.get(direction_str, 0)
        edge_attrs.append([distance, direction])
    
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else None
    return x, edge_index, edge_attr


def create_dataset(num_graphs=10, num_nodes=6, edge_connection_prob=0.3, max_tries=100, plot=False):
    dataset = []
    for _ in range(num_graphs):
        env = Environment(num_nodes=num_nodes, 
                          edge_connection_prob=edge_connection_prob,
                          max_tries=max_tries,
                          plot=plot)
        adj_dict = prepare_dictionary(env.graph)
        dataset.append(adj_dict)
    return dataset

if __name__ == '__main__':
    full_dataset = create_dataset(num_graphs=50, num_nodes=6, edge_connection_prob=0.3, max_tries=100, plot=False)
    
    train_size = 40
    validation_size = 10
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    
    # for batch in train_loader:
    #     print(batch)
    #     break
    for batch in train_loader:
        dictionary = batch[0]
        node_t, edge_i, edge_atr = convert_dict_to_tensors(dictionary)
        
        print("Feature tensor x:", node_t) # node type
        print("Edge index:", edge_i) # tells the connection
        print("Edge attributes:", edge_atr) # [distance, direction] of each connection
        break