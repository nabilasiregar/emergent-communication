import math
import random
import networkx as nx
import torch
from torch.utils.data import DataLoader, random_split
from environment import Environment

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
    
    for batch in train_loader:
        print(batch)
        break