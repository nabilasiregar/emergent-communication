import torch, random, argparse, math
from graph.environment import Environment
from graph.data_builder import DataConverter, get_nest_and_food_indices
from utils.helpers import hop_length

def generate_set(num_samples: int,
                 target_hops: int,
                 num_nodes: int,
                 extra_degree: float,
                 seed: int = 42):
    """Produce num_samples graphs whose nest to food path length = target_hops"""
    rng = random.Random(seed)
    samples = []
    tries = 0
    while len(samples) < num_samples:
        env = Environment(num_nodes=num_nodes,
                          extra_degree=extra_degree,
                          world_size=(100,100),
                          seed=rng.randint(0,2**32-1),
                          allow_adjacent_nest_food=True)
        if hop_length(env) != target_hops:
            tries += 1
            if tries % 5000 == 0:
                print(f"Still searching... {len(samples)}/{num_samples}")
            continue
        data = DataConverter.convert(env)
        nest_id, food_id = get_nest_and_food_indices(data)
        samples.append((data, nest_id, food_id))
    return samples

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data", type=str)
    p.add_argument("--train_atomic", type=int, default=8000,
                   help="num of graphs (train)")
    p.add_argument("--val_atomic", type=int, default=1000,
                   help="num of graphs (val)")
    p.add_argument("--test_compositionality", type=int, default=4096,
                   help="num of graphs (zero-shot test)")
    p.add_argument("--num_nodes_atomic", type=int, default=5,
                   help="num_nodes for train/val graphs")
    p.add_argument("--num_nodes_composionality", type=int, default=5,
                   help="num_nodes for zero-shot graphs")
    p.add_argument("--extra_degree", type=float, default=0.5)
    args = p.parse_args()

    print("Generating atomic train data...")
    train_atomic = generate_set(args.train_atomic, 1, args.num_nodes_atomic, args.extra_degree, seed=123)
    print("Generating atomic validation data...")
    val_atomic   = generate_set(args.val_atomic, 1, args.num_nodes_atomic, args.extra_degree, seed=456)
    print("Generating test data...")
    test_comp    = generate_set(args.test_compositionality, 2, args.num_nodes_composionality, args.extra_degree, seed=789)

    torch.save(train_atomic, f"{args.out_dir}/atomic_train.pt")
    torch.save(val_atomic, f"{args.out_dir}/atomic_val.pt")
    torch.save(test_comp, f"{args.out_dir}/compositionality_test.pt")
    print("Saved datasets to", args.out_dir)
