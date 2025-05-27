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
try:
    aux_input[k] = _check_cat([x.aux_input[k] for x in interactions])
except Exception as e:
    aux_input[k] = None
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

## Vocab size
### Bee-like
vocab_size = 1 + 8 + 10  # 1 for EOS, 8 directions, 10 distance bins
max_len = 2 # distance + direction
### Human-like
vocab_size = 50  # 8 directions, 10 distance words, 5 verbs, etc.
max_len = 10

## Experiment
### Logging data
Add
```
DataLogger(save_path="logs/experiment.json")
```
in callbacks

### Running game
Example:
```
python -m game --communication_type bee --mode gs --train_data data/samples:10_000_train_data_totalnodes:10.pt --validation_data data/samples:10_000_test_data_totalnodes:10.pt --temperature 1.5 --n_epochs 100

python -m game --communication_type human --mode gs --train_data data/samples:10_000_train_data_totalnodes:10.pt --validation_data data/samples:10_000_test_data_totalnodes:10.pt --max_len 10 --vocab_size 4 --temperature 1.5 --lr 0.0001 --n_epochs 50
```

