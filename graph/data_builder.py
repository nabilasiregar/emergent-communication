import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, random_split
from environment import Environment
import pdb

class DataConverter:
    """
    Converts the graph environment to a PyG data object where:
      - x is the node feature matrix (one-hot encoding of node types)
      - edge_index is the connectivity tensor
      - edge_attr is a tensor with columns [distance, direction]
    """
    NODE_TYPES = {"nest": 0, "food": 1, "distractor": 2}
    DIRECTIONS = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}
    
    @staticmethod
    def convert(env: Environment) -> Data:
        nodes = sorted(env.directed_graph.nodes())
        
        node_types_list = [env.directed_graph.nodes[n]['node_type'] for n in nodes]
        numeric_node_types = [DataConverter.NODE_TYPES[nt] for nt in node_types_list]
        node_types_tensor = torch.tensor(numeric_node_types, dtype=torch.long)
        num_node_types = len(DataConverter.NODE_TYPES)
        x = F.one_hot(node_types_tensor, num_classes=num_node_types).float()
        
        edges = list(env.directed_graph.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        distances = []
        directions = []
        for u, v in edges:
            edge_data = env.directed_graph.edges[u, v]
            distances.append(edge_data['distance'])
            direction = edge_data['direction']
            if isinstance(direction, str):
                directions.append(DataConverter.DIRECTIONS[direction])
            else:
                directions.append(direction)
        if distances:
            distances_tensor = torch.tensor(distances, dtype=torch.float).view(-1, 1)
            directions_tensor = torch.tensor(directions, dtype=torch.long).view(-1, 1)
            edge_attr = torch.cat([distances_tensor, directions_tensor], dim=1)
        else:
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def get_nest_and_food_indices(data: Data) -> (int, int):
    """
    Utility function to extract the nest and food node indices from a Data object.
    """
    node_types = torch.argmax(data.x, dim=1)
    nest_indices = (node_types == DataConverter.NODE_TYPES["nest"]).nonzero(as_tuple=True)[0]
    food_indices = (node_types == DataConverter.NODE_TYPES["food"]).nonzero(as_tuple=True)[0]
    return nest_indices.item(), food_indices.item()

if __name__ == '__main__':
    env = Environment(num_nodes=4, connection_prob=0.5)
    data = DataConverter.convert(env)
    print("Converted Data:")
    print("x:", data.x)
    print("edge_index:", data.edge_index)
    print("edge_attr:", data.edge_attr)
    nest, food = get_nest_and_food_indices(data)
    print(f"Nest ID: {nest}. Food ID: {food}")
