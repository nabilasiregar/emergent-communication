import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_best_acc(seed: int, base_dir: str, species: str, game_size: int):
    """
    seed: integer seed
    species: 'bee' or 'human'
    game_size: 10 or 5; filenames:
       10 nodes: systematic_dirs_{species}_seed{seed}.csv / systematic_dist_{species}_seed{seed}.csv
       5 nodes: 5nodes_systematic_dirs_{species}_seed{seed}.csv / 5nodes_systematic_dist_{species}_seed{seed}.csv
    Returns: train_acc, novel_distances_acc, novel_directions_acc
    """
    if game_size == 10:
        dir_file = os.path.join(base_dir, f"systematic_dirs_{species}_seed{seed}.csv")
        dist_file = os.path.join(base_dir, f"systematic_dist_{species}_seed{seed}.csv")
    elif game_size == 5:
        dir_file = os.path.join(base_dir, f"5nodes_systematic_dirs_{species}_seed{seed}.csv")
        dist_file = os.path.join(base_dir, f"5nodes_systematic_dist_{species}_seed{seed}.csv")
    else:
        raise ValueError("Unsupported game_size (must be 5 or 10)")

    if not os.path.isfile(dir_file):
        raise FileNotFoundError(f"Missing file for novel directions: {dir_file}")
    if not os.path.isfile(dist_file):
        raise FileNotFoundError(f"Missing file for novel distances: {dist_file}")

    df_dir = pd.read_csv(dir_file)
    df_dist = pd.read_csv(dist_file)

    # pick epoch with highest train accuracy (from dirs file)
    train_rows = df_dir[df_dir["mode"] == "train"]
    if train_rows.empty:
        raise ValueError(f"No train rows in {dir_file}")
    max_train_acc = train_rows["acc"].max()
    candidates = train_rows[train_rows["acc"] == max_train_acc]
    best_train_row = candidates.sort_values("epoch").iloc[0]
    best_train_epoch = int(best_train_row["epoch"])
    train_acc = float(best_train_row["acc"])

    def test_acc_at_epoch(df, epoch):
        test_rows = df[df["mode"] == "test"]
        if test_rows.empty:
            raise ValueError("No test rows in dataframe.")
        exact = test_rows[test_rows["epoch"] == epoch]
        if not exact.empty:
            return float(exact.iloc[0]["acc"])
        earlier = test_rows[test_rows["epoch"] < epoch]
        if not earlier.empty:
            return float(earlier.sort_values("epoch").iloc[-1]["acc"])
        return float(test_rows.sort_values("epoch").iloc[0]["acc"])

    novel_dirs_acc = test_acc_at_epoch(df_dir, best_train_epoch)
    novel_dists_acc = test_acc_at_epoch(df_dist, best_train_epoch)

    return train_acc, novel_dists_acc, novel_dirs_acc  # Training, Novel Distances, Novel Directions

def aggregate(seeds, base_dir, species, game_size):
    cats = ["Training", "Novel Distances", "Novel Directions"]
    results = {cat: [] for cat in cats}
    for seed in seeds:
        train_acc, novel_dists_acc, novel_dirs_acc = load_best_acc(seed, base_dir, species, game_size)
        results["Training"].append(train_acc)
        results["Novel Distances"].append(novel_dists_acc)
        results["Novel Directions"].append(novel_dirs_acc)
    return results

def compute_mean_ci(data_list, confidence=0.95):
    a = np.array(data_list, dtype=float)
    n = len(a)
    if n < 2:
        return float(a.mean()), 0.0
    mean = a.mean()
    sem = stats.sem(a, ddof=1)
    t_multiplier = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return mean, sem * t_multiplier

def plot_grouped(seeds, base_dir, species_colors=None, figsize=(12,6), save_path=None):
    """
    Plots grouped bars: categories on x-axis, within each category four bars:
      Bee-10, Bee-5, Human-10, Human-5.
    Orange shades for bee, blue shades for human.
    """
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
    })

    if species_colors is None:
        species_colors = {"bee": "#ff9933", "human": "#1f77b4"}

    # define shades: darker for 10-node, lighter for 5-node
    color_map = {
        ("bee", 10): "#d66300",    # darker orange
        ("bee", 5): "#ffbf80",     # lighter orange
        ("human", 10): "#1f4f8a",  # darker blue
        ("human", 5): "#8fbde5",   # lighter blue
    }

    categories = ["Training", "Novel Distances", "Novel Directions"]
    species_list = ["bee", "human"]
    game_sizes = [10, 5] 

    stats_data = {}  
    for sp in species_list:
        for size in game_sizes:
            agg = aggregate(seeds, base_dir, sp, size)
            stats_data[(sp, size)] = {}
            for cat in categories:
                mean, ci = compute_mean_ci(agg[cat], confidence=0.95)
                stats_data[(sp, size)][cat] = (mean, ci)

    x = np.arange(len(categories))
    bars_per_group = len(species_list) * len(game_sizes)  # 4
    total_width = 0.9
    bar_width = total_width / bars_per_group
    # order within group: Bee-10, Bee-5, Human-10, Human-5
    offsets = {
        ("bee", 10): -1.5 * bar_width,
        ("bee", 5): -0.5 * bar_width,
        ("human", 10): 0.5 * bar_width,
        ("human", 5): 1.5 * bar_width,
    }

    fig, ax = plt.subplots(figsize=figsize)

    for (sp, size), cat_stats in stats_data.items():
        means = [cat_stats[cat][0] for cat in categories]
        cis = [cat_stats[cat][1] for cat in categories]
        positions = x + offsets[(sp, size)]
        label = f"({sp.capitalize()}, {size})"
        bars = ax.bar(
            positions,
            means,
            yerr=cis,
            capsize=6,
            width=bar_width,
            label=label,
            color=color_map.get((sp, size)),
            edgecolor="black"
        )
        for pos, mean in zip(positions, means):
            ax.annotate(f"{mean:.3f}",
                        xy=(pos, mean),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)

    ax.legend(title="(Communication Channel, Game Size)",
              loc="upper center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=2,
              frameon=True)

    fig.subplots_adjust(bottom=0.30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    seeds = [42, 27, 31, 123, 2025]
    base_dir = "logs/csv/2025-08-04"
    plot_grouped(seeds, base_dir)





