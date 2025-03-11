import torch
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader, random_split
from graph.environment import Environment
from graph.data_builder import DataConverter, get_nest_and_food_indices

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

def collate_fn(batch):
    data_list = [sample[0] for sample in batch]
    nest_tensor = torch.tensor([sample[1] for sample in batch])
    food_tensor = torch.tensor([sample[2] for sample in batch])

    batch_data = Batch.from_data_list(data_list)

    sender_input = torch.zeros(len(batch), dtype=torch.long) 
    receiver_input = None
    labels = food_tensor # what the receiver predicts
    aux_input = {
        'data': batch_data,
        'nest_tensor': nest_tensor,
        'food_tensor': food_tensor
    }

    return sender_input, labels, receiver_input, aux_input


def create_dataset(file_path, num_samples=100, num_nodes=6, connection_prob=0.3, train_ratio=0.8):
    dataset = GraphDataset(num_samples=num_samples, num_nodes=num_nodes, connection_prob=connection_prob)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    train_set = [train_dataset[i] for i in range(len(train_dataset))]
    val_set = [val_dataset[i] for i in range(len(val_dataset))]
    
    torch.save({
        'train_set': train_set,
        'val_set': val_set
    }, file_path)

    # return train_loader, val_loader

if __name__ == '__main__':
    # train_loader, val_loader = create_dataset(num_samples=50, num_nodes=6, connection_prob=0.3, batch_size=5)
    
    # print("TRAINING BATCH:")
    # for batched_data, start_tensor, target_tensor in train_loader:
    #     print("Batched Graph:")
    #     print("x:", batched_data.x)
    #     print("edge_index:", batched_data.edge_index)
    #     print("edge_attr:", batched_data.edge_attr)
    #     print("Nest ID:", start_tensor)
    #     print("Food ID:", target_tensor)
    #     break
    create_dataset("graph_dataset.pt", num_samples=500, num_nodes=6)