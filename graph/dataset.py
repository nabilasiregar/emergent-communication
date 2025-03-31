import argparse
import torch
from torch.utils.data import Dataset,random_split
from graph.environment import Environment
from graph.data_builder import DataConverter, get_nest_and_food_indices
import pdb
class GraphDataset(Dataset):
    def __init__(self, num_samples: int = 100, num_nodes: int = 6, connection_prob: float = 0.3):
        self.samples = []
        for _ in range(num_samples):
            env = Environment(num_nodes=num_nodes, connection_prob=connection_prob)
            data = DataConverter.convert(env)
            nest_id, food_id = get_nest_and_food_indices(data)
            self.samples.append((data, nest_id, food_id))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        return self.samples[index]

def create_dataset(num_samples=100, num_nodes=6, connection_prob=0.3, train_ratio=0.8):
    # later add a handler so user can only initiate with minimum 2 number of nodes (food and nest node)
    dataset = GraphDataset(num_samples=num_samples, num_nodes=num_nodes, connection_prob=connection_prob)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_set = [train_dataset[i] for i in range(len(train_dataset))]
    val_set = [val_dataset[i] for i in range(len(val_dataset))]

    return train_set, val_set

def main():
    parser = argparse.ArgumentParser(description="Generate and split graph dataset")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Total number of graph samples to generate (default: 500)")
    parser.add_argument("--num_nodes", type=int, default=5,
                        help="Number of nodes in each graph (default: 5)")
    parser.add_argument("--connection_prob", type=float, default=0.3,
                        help="Probability of connection between nodes (default: 0.3)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of samples for the training set (default: 0.8)")
    parser.add_argument("--train_output", type=str, default="train_data.pt",
                        help="File path to save the training dataset (default: train_data.pt)")
    parser.add_argument("--test_output", type=str, default="test_data.pt",
                        help="File path to save the test dataset (default: test_data.pt)")
    
    args = parser.parse_args()
    
    train_set, test_set = create_dataset(args.num_samples, args.num_nodes, args.connection_prob, args.train_ratio)
    
    torch.save(train_set, args.train_output)
    torch.save(test_set, args.test_output)

if __name__ == '__main__':
    main()