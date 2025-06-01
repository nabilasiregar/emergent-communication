import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from graph.data_builder import DataConverter
from collections import Counter
import pdb

node_types = {v: k for k, v in DataConverter.NODE_TYPES.items()}
directions = {v: k for k, v in DataConverter.DIRECTIONS.items()}

def build_graph(dataset_path):
    """
    Loads the dataset and for each sample, build the graph
    """
    dataset = torch.load(dataset_path)
    graphs_data = []

    for data, nest_id, food_id in dataset:
        x_np = data.x.argmax(dim=1).numpy()
        edge_idx = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy()

        G = nx.DiGraph()
        # add nodes with the correct node_type
        for node_id, t_idx in enumerate(x_np):
            G.add_node(node_id, node_type=node_types[int(t_idx)])

        # add edges with attributes
        for e in range(edge_idx.shape[1]):
            u, v       = int(edge_idx[0, e]), int(edge_idx[1, e])
            distance       = float(edge_attr[e, 0])
            direction_idx    = int(edge_attr[e, 1])
            G.add_edge(u, v, distance=distance, direction=directions[direction_idx])

        graphs_data.append((G, nest_id, food_id))

    return graphs_data

def analyze_graphs(graphs_data, label):
    records = []
    for idx, (G, nest_id, food_id) in enumerate(graphs_data):
        # get the shortest path between food and nest nodes in terms of num of hops (edges)
        path_hops = nx.shortest_path(G, source=food_id, target=nest_id)
        d_fn_hops = len(path_hops) - 1

        # shortest distance between food and nest nodes
        d_fn_distance = nx.shortest_path_length(
            G, source=food_id, target=nest_id, weight="distance"
        )

        num_distractors_on_path = sum(
            1 for v in path_hops[1:-1]
            if G.nodes[v]["node_type"] == "distractor"
        )

        num_possible_paths = sum(
            1 for _ in nx.all_simple_paths(G, source=food_id, target=nest_id)
        )

        # density measures how many edges exist in the graph compared to the maximum possible number of edges
        # diameter is he longest shortest path (in hops) between any two nodes in the entire graph (different 
        # from d_fn_hops which is only between nest and food nodes)

        records.append({
            "dataset": label,
            "sample": idx,
            "d_fn_hops": d_fn_hops,
            "d_fn_distance": d_fn_distance,
            "num_distractors_on_path": num_distractors_on_path,
            "num_possible_paths": num_possible_paths,
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G.to_undirected()),
            "diameter": nx.diameter(G.to_undirected())
        })

    df = pd.DataFrame(records)
    print(f"\n=== Statistics for {label} ===")
    print(df.describe())
    print(df.head())

def compute_centrality_measures(graphs_data, label):
    records = []
    for idx, (G, nest_id, food_id) in enumerate(graphs_data):
        degree_cent    = nx.degree_centrality(G)
        betweenness_cent   = nx.betweenness_centrality(G)
        close_cent  = nx.closeness_centrality(G)

        avg_deg    = sum(degree_cent.values()) / len(degree_cent)
        max_deg    = max(degree_cent.values())
        avg_betw   = sum(betweenness_cent.values()) / len(betweenness_cent)
        max_betw   = max(betweenness_cent.values())
        avg_close  = sum(close_cent.values()) / len(close_cent)
        max_close  = max(close_cent.values())

        records.append({
            "dataset": label,
            "sample": idx,
            "avg_degree_centrality": avg_deg,
            "max_degree_centrality": max_deg,
            "avg_betweenness_centrality": avg_betw,
            "max_betweenness_centrality": max_betw,
            "avg_closeness_centrality": avg_close,
            "max_closeness_centrality": max_close
        })

    df = pd.DataFrame(records)
    print(f"\n=== Centrality stats for {label} ===")
    print(df.describe())

def identify_hub_nodes(graphs_data, label):
    """
    For each sample, finds the top-1 hub node by degree centrality,
    then checks whether that node is the nest, food, or a distractor.

    At the end, prints a summary count across all samples.
    """
    role_counts = Counter()

    for idx, (G, nest_id, food_id) in enumerate(graphs_data):
        deg_cent = nx.degree_centrality(G)
        top_node = max(deg_cent.items(), key=lambda x: x[1])[0]

        if top_node == nest_id:
            role = "nest"
        elif top_node == food_id:
            role = "food"
        else:
            role = G.nodes[top_node].get("node_type", "unknown")

        role_counts[role] += 1

    total = sum(role_counts.values())
    print(f"\n=== Hub summary for {label} ===")
    for role, count in role_counts.items():
        print(f"  {role:12s}: {count:3d} ({count / total:.1%})")


if __name__ == "__main__":
    dataset_paths = [
        "data/samples:10_000_train_data_totalnodes:5.pt",
        "data/samples:10_000_train_data_totalnodes:10.pt",
        "data/samples:10_000_train_data_totalnodes:20.pt"
    ]

    for path in dataset_paths:
        label = path.split("/")[-1]
        graphs_data = build_graph(path)

        analyze_graphs(graphs_data, label)
        compute_centrality_measures(graphs_data, label)
        identify_hub_nodes(graphs_data, label)
