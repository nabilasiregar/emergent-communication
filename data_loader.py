import math
import random
import networkx as nx
import torch
from torch.utils.data import DataLoader, random_split
from environment import Environment
import pdb
from torch_geometric.data import Data

def prepare_dictionary(graph):
    """
    Convert the graph into an adjacency dictionary.
    
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

def convert_graph_dict_to_tensors(graph_dict):
    """
    Converts the graph dictionary into tensors.
    
    Returns:
      - x: Node feature tensor of shape [num_nodes, feature_dim]
      - edge_index: Tensor of shape [2, num_edges]
      - edge_attr: Tensor of shape [num_edges, 2] (distance and direction)
    """
    type_mapping = {'Nest': [1, 0, 0], 'Food': [0, 1, 0], 'Distractor': [0, 0, 1]}
    direction_mapping = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}
    
    nodes = sorted(graph_dict.keys())
    features = []
    for node in nodes:
        node_data = graph_dict[node]
        node_type = node_data["type"][0] if isinstance(node_data["type"], list) else node_data["type"]
        features.append(type_mapping.get(node_type, [0, 0, 1]))
    
    edge_index = [[], []]
    edge_attrs = []
    for node in nodes:
        node_data = graph_dict[node]
        for neighbor in node_data.get("neighbors", []):
            neighbor_id, attr = neighbor
            if isinstance(neighbor_id, torch.Tensor):
                neighbor_id = int(neighbor_id.item())
            else:
                neighbor_id = int(neighbor_id)
            edge_index[0].append(node)
            edge_index[1].append(neighbor_id)
            
            if torch.is_tensor(attr["distance"]):
                if attr["distance"].dim() > 0:
                    distance = attr["distance"][0].item()
                else:
                    distance = attr["distance"].item()
            else:
                distance = attr["distance"]
            
            direction_str = attr["direction"][0] if isinstance(attr["direction"], list) else attr["direction"]
            direction = direction_mapping.get(direction_str, 0)
            edge_attrs.append([distance, direction])
    
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else None
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

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

def custom_collate_fn(batch):
    return batch[0]

if __name__ == '__main__':
    full_dataset = create_dataset(num_graphs=50, num_nodes=6, edge_connection_prob=0.3, max_tries=100, plot=False)
    
    train_size = 40
    validation_size = 10
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    for batch in train_loader:
        graph_data = convert_graph_dict_to_tensors(batch)
        
        print("Feature tensor x:", graph_data.x)      # node type
        print("Edge index:", graph_data.edge_index)     # graph connectivity
        print("Edge attributes:", graph_data.edge_attr)   # [distance, direction] for each edge
        break
