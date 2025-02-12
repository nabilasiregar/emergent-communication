import torch
from torch_geometric.data import Data
import networkx as nx
import pdb

def convert_graph_to_tensors(graph):
    """
    Convert a NetworkX graph with node and edge attributes to a PyTorch Geometric Data object.
    
    Expected node attributes:
      - type: a string that can be 'Nest', 'Food', or 'Distractor'
      - direction: a discretized direction (e.g. one of "N", "NE", "E", "SE", "S", "SW", "W", "NW")
      
    Expected edge attributes:
      - distance: a float representing the distance between nodes
    """
    num_nodes = graph.number_of_nodes()

    # build node feature matrix
    # for each node, we combine a one-hot encoding of the node type and one-hot encoding of the direction
    node_features = []
    
    # define one-hot encoding mappings for node type
    type_mapping = {
        'Nest': [1, 0, 0],
        'Food': [0, 1, 0],
        'Distractor': [0, 0, 1]
    }

    directions_list = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    
    for node in range(num_nodes):
        node_data = graph.nodes[node]
        # get type feature; default to Distractor if type is missing
        type_feat = type_mapping.get(node_data.get('type', 'Distractor'), [0, 0, 1])
        
        # get direction feature as a one-hot vector for 8 possible directions
        direction_feat = [0] * 8
        direction = node_data.get('direction', None)
        if direction in directions_list:
            idx = directions_list.index(direction)
            direction_feat[idx] = 1
        else:
            # if no valid direction is provided, leave the one-hot vector as all zeros
            pass

        node_features.append(type_feat + direction_feat)

    # create a tensor for node features
    x = torch.tensor(node_features, dtype=torch.float)

    # build the edge_index tensor
    edges = list(graph.edges())
    src_nodes = []
    dst_nodes = []
    for u, v in edges:
        src_nodes.append(u)
        dst_nodes.append(v)
        # in an undirected graph, each edge in NetworkX appears only once but for PyG, 
        # we add both directions to ensure every node receives messages from all its neighbors
        src_nodes.append(v)
        dst_nodes.append(u)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    # build edge features (edge_attr)
    edge_features = []
    for u, v in edges:
        distance = graph[u][v].get('distance', 0.0)
        edge_features.append([distance])
        edge_features.append([distance])
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    global GLOBAL_EDGE_INDEX, GLOBAL_EDGE_ATTR
    GLOBAL_EDGE_INDEX = edge_index
    GLOBAL_EDGE_ATTR = edge_attr
    
    return data

if __name__ == '__main__':
    try:
        from environment import Environment
        env = Environment(num_distractors=2)
        G = env.graph
        print("Loaded environment from environment.py")
    except ImportError:
        print("environment.py not found")
    
    pyg_data = convert_graph_to_tensors(G)
    
    print("PyG Data object:")
    print("Node features (x):")
    print(pyg_data.x)
    print("\nEdge index:")
    print(pyg_data.edge_index)
    if pyg_data.edge_attr is not None:
        print("\nEdge attributes (edge_attr):")
        print(pyg_data.edge_attr)
