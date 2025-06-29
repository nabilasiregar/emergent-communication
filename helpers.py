import numpy as np
import random
import torch
from torch_geometric.data import Batch
import pdb

def collate_fn(batch):
    datas, nests, foods = zip(*batch)
    
    batch_data = Batch.from_data_list(datas)
    
    nest_tensor = torch.tensor(
        [batch_data.ptr[i] + nests[i] for i in range(len(nests))],
        dtype=torch.long
    )
    food_tensor = torch.tensor(
        [batch_data.ptr[i] + foods[i] for i in range(len(foods))],
        dtype=torch.long
    )
    labels  = torch.tensor(foods, dtype=torch.long)
    sender_input   = batch_data.x
    receiver_input = None

    aux_input = {
        "data": batch_data,
        "nest_tensor": nest_tensor,
        "food_tensor": food_tensor,
        "nest_idx": torch.tensor(nests, dtype=torch.long),
        "food_idx": torch.tensor(foods, dtype=torch.long)
    }
   
    return sender_input, labels, receiver_input, aux_input

def strip_node_types(x, keep_dims):
    out = torch.zeros_like(x)
    out[:, keep_dims] = x[:, keep_dims]
    return out

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False