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
python -m graph.dataset --num_samples 500 --num_nodes 5 --train_ratio 0.8 --train_output data/train_data.pt --test_output data/test_data.pt
```

## Vocab size
### Bee-like
vocab_size = 1 + 8 + 10  # 1 for EOS, 8 directions, 10 distance bins
max_len = 2 # distance + direction
### Human-like
vocab_size = 50  # 8 directions, 10 distance words, 5 verbs, etc.
max_len = 10

