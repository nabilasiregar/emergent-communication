import networkx as nx
import random
import torch
from typing import List, Tuple, Set
from graph.environment import Environment
from graph.data_builder import DataConverter, get_nest_and_food_indices

def bin_distance(distance: float) -> str:
    """Bin continuous distances into discrete categories"""
    if distance < 30:
        return "very close"
    elif distance < 70:
        return "a bit further"
    else:
        return "far away"

def extract_path_components(env) -> List[Tuple[str, str]]:
    """Extract (direction, distance_bin) pairs from shortest path"""
    G = env.directed_graph.to_undirected()
    nest = next(n for n,d in G.nodes(data=True) if d["node_type"] == "nest")
    food = next(n for n,d in G.nodes(data=True) if d["node_type"] == "food")
    
    path = nx.shortest_path(G, nest, food)
    
    components = []
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        direction = env.directed_graph.edges[u, v]["direction"]
        distance = env.directed_graph.edges[u, v]["distance"]
        distance_bin = bin_distance(distance)
        components.append((direction, distance_bin))
    
    return components

def hop_length(env) -> int:
    """Return num of hops (edges) from nest to food"""
    G = env.directed_graph.to_undirected()
    nest = next(n for n,d in G.nodes(data=True) if d["node_type"] == "nest")
    food = next(n for n,d in G.nodes(data=True) if d["node_type"] == "food")
    return nx.shortest_path_length(G, nest, food)

def generate_compositional_dataset(
    num_samples: int, 
    target_hops: int, 
    allowed_combinations: Set[Tuple[str, str]],
    num_nodes: int = 10,
    extra_degree: float = 0.5,
    world_size: Tuple[int, int] = (100, 100),
    seed: int = None
) -> List[Tuple]:
    """
    Generate dataset with only specific direction-distance combinations
    
    Args:
        num_samples: Number of samples to generate
        target_hops: Required number of hops in shortest path
        allowed_combinations: Set of (direction, distance_bin) tuples allowed
        num_nodes: Number of nodes in each graph
        extra_degree: Extra degree parameter for graph generation
        world_size: Size of 2D world
        rng: Random number generator
    
    Returns:
        List of (data, nest_id, food_id) tuples
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    samples = []
    tries = 0
    
    while len(samples) < num_samples:
        env = Environment(
            num_nodes=num_nodes,
            extra_degree=extra_degree,
            world_size=world_size,
            seed=rng.randint(0, 2**32-1),
            allow_adjacent_nest_food=True
        )
        
        if hop_length(env) != target_hops:
            tries += 1
            if tries % 5000 == 0:
                print(f"Searching for {target_hops}-hop paths... {len(samples)}/{num_samples}, {tries} tries")
            continue
            
        # check if path uses only allowed combinations
        path_components = extract_path_components(env)
        if all(comp in allowed_combinations for comp in path_components):
            data = DataConverter.convert(env)
            nest_id, food_id = get_nest_and_food_indices(data)
            samples.append((data, nest_id, food_id))
            print(f"Found valid sample {len(samples)}/{num_samples}: {path_components}")
        else:
            tries += 1
            if tries % 5000 == 0:
                print(f"Filtering combinations... {len(samples)}/{num_samples}, {tries} tries")
    
    print(f"Dataset generation complete. Total tries: {tries}")
    return samples

def generate_systematic_datasets(
    train_samples: int = 1000,
    test_samples: int = 200,
    target_hops: int = 3,
    seed: int = 42,
    **kwargs
) -> Tuple[List, List, List]:
    """
    Generate train/test datasets for Systematic Recombination
    
    Returns:
        (train_data, test_novel_directions, test_novel_distances)
    """
    
    train_combinations = set()

    cardinal_dirs = ["N", "E", "S", "W"]
    seen_distances = ["very close", "far away"]
    
    for direction in cardinal_dirs:
        for distance in seen_distances:
            train_combinations.add((direction, distance))
    
    print(f"Training combinations: {train_combinations}")
    
    # Test set 1: Novel directions with seen distances
    test_novel_directions = set()
    diagonal_dirs = ["NE", "SE", "SW", "NW"]
    
    for direction in diagonal_dirs:
        for distance in seen_distances:
            test_novel_directions.add((direction, distance))
    
    print(f"Test novel directions: {test_novel_directions}")
    
    # Test set 2: Seen directions with novel distance
    test_novel_distances = set()
    novel_distance = "a bit further"
    
    for direction in cardinal_dirs:
        test_novel_distances.add((direction, novel_distance))
    
    print(f"Test novel distances: {test_novel_distances}")
    
    print("\n=== Generating Training Data ===")
    train_data = generate_compositional_dataset(
        train_samples, target_hops, train_combinations, seed=seed, **kwargs
    )
    
    print("\n=== Generating Test Data (Novel Directions) ===")
    test_data_directions = generate_compositional_dataset(
        test_samples, target_hops, test_novel_directions, seed=seed, **kwargs
    )
    
    print("\n=== Generating Test Data (Novel Distances) ===")
    test_data_distances = generate_compositional_dataset(
        test_samples, target_hops, test_novel_distances, seed=seed, **kwargs
    )
    
    return train_data, test_data_directions, test_data_distances

if __name__ == "__main__":
    train_data, test_novel_directions, test_novel_distances = generate_systematic_datasets(
        train_samples=2000,
        test_samples=300,
        target_hops=3,
        num_nodes=10,
        extra_degree=0.5,
        world_size=(100, 100),
        seed=42
    )

    torch.save(train_data, 'data/systematic_10nodes_train.pt')
    torch.save(test_novel_directions, 'data/systematic_10nodes_test_directions.pt')
    torch.save(test_novel_distances, 'data/systematic_10nodes_test_distances.pt')
    
    print(f"\nDataset Summary:")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples (novel directions): {len(test_novel_directions)}")
    print(f"Test samples (novel distances): {len(test_novel_distances)}")
    