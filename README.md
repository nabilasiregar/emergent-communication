## Environment Setup
### 1. Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install EGG (editable mode)
```
git clone git@github.com:facebookresearch/EGG.git
cd EGG
pip install -e .
cd ..
```

### 4. Apply local modification in EGG
Important: the following manual edit to EGG/core/interaction.py is required:
change line 209
```
 aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
```
to
```
values = [x.aux_input[k] for x in interactions]
                try:
                    aux_input[k] = _check_cat(values)
                except Exception:
                    aux_input[k] = values
```

After making the changes, reinstall EGG to reflect the modifications:
```
cd EGG
pip install -e .
cd ..
```

## Environment
To visualize the environment, run:
```
python -m graph.environment
```

## Graph representation
The graph is represented as a Data object where x is a node feature matrix using one-hot encoding to indicate the type of each node (nest, food or distractor), edge_index is a 2-row tensor that defines the directed edges between nodes (which nodes are connected and in what direction), and edge_attr holds the distance and direction between connected nodes. For example, node 0 is identified as the nest, node 2 as the food source, and nodes 1 and 3 as distractors, based on their one-hot encodings in x. Each edge has an associated distance (a float) and direction (encoded as an integer) in edge_attr.

To see the graph representation, run:
```
python -m graph.data_builder
```

## Create dataset
- num_samples: how many graph/network to be created
- num_nodes: total number of nodes within the network. 2 will always be reserved for food and nest node, the rest are distractors.
- train_ratio: the train/test split ratio
- train_output: output filepath for train set
- test_output: output filepath for test set

Example:
```
python -m graph.dataset --num_samples 10000 --num_nodes 10 --train_ratio 0.8 --train_output data/samples:10_000_train_data_totalnodes:10.pt --test_output data/samples:10_000_test_data_totalnodes:10.pt
```

## Running the game
For experimentation and fine-tuning, do not pass `--final_run` flag to the arguments. This will set aside a portion of the training data to create a validation set, and the test data will remain unused.
For the final run, include the `--final_run` flag to evaluate the model on the test set.
Example:
```
python -m game --communication_type bee --mode gs --train_data data/samples:10_000_train_data_totalnodes:10.pt --validation_data data/samples:10_000_test_data_totalnodes:10.pt --temperature 2.5 --lr 0.005 --sender_hidden 64 --receiver_hidden 64  --sender_embedding 32 --receiver_embedding 32 --vocab_size 8 --n_epochs 100 --final_run

python -m game --communication_type human --mode gs --train_data data/samples:10_000_train_data_totalnodes:10.pt --validation_data data/samples:10_000_test_data_totalnodes:10.pt --sender_hidden 128 --receiver_hidden 128 --sender_embedding 64 --receiver_embedding 64 --sender_cell 'gru' --receiver_cell 'gru' --vocab_size 100 --temperature 2.5 --lr 0.001 --max_len 10 --n_epochs 100 --final_run
```

