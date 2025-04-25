import torch
from torch_geometric.data import Batch
import pdb

def collate_fn(batch):
    data_list = [sample[0] for sample in batch]
    batch_data = Batch.from_data_list(data_list)

    nest_tensor = torch.tensor([
        batch_data.ptr[i] + sample[1] for i, sample in enumerate(batch)
    ], dtype=torch.long)

    food_tensor = torch.tensor([
        batch_data.ptr[i] + sample[2] for i, sample in enumerate(batch)
    ], dtype=torch.long)

    labels = torch.tensor([sample[2] for sample in batch], dtype=torch.long)
    sender_input = torch.zeros(len(batch), dtype=torch.long)
    receiver_input = None
   
    aux_input = {
        'data': batch_data,
        'nest_tensor': nest_tensor,
        'food_tensor': food_tensor
    }
    return sender_input, labels, receiver_input, aux_input