import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re

def plot_accuracies_over_epochs(
    csv_files: list,
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy",
    title: str = "Mean Accuracy Over Epochs Across Different Seeds"
):
    all_dfs = []
    for path in csv_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: {path!r}")
        df = pd.read_csv(path)
        seed_label = os.path.splitext(os.path.basename(path))[0]
        df["seed"] = seed_label
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    combined["epoch"] = combined["epoch"].astype(int)
    combined["acc"] = combined["acc"].astype(float)
    combined["mode"] = combined["mode"].astype(str)

    stats = combined.groupby(["mode", "epoch"])["acc"].agg(["mean", "std"]).reset_index()
    train_stats = stats[stats["mode"] == "train"].sort_values("epoch")
    test_stats = stats[stats["mode"] == "test"].sort_values("epoch")

    plt.figure(figsize=(8, 6))

    if not train_stats.empty:
        epochs_train = train_stats["epoch"].to_numpy()
        mean_train = train_stats["mean"].to_numpy()
        std_train = train_stats["std"].fillna(0).to_numpy() 
        plt.plot(
            epochs_train,
            mean_train,
            label="Train",
            linewidth=2
        )
        plt.fill_between(
            epochs_train,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.2
        )

    if not test_stats.empty:
        epochs_test = test_stats["epoch"].to_numpy()
        mean_test = test_stats["mean"].to_numpy()
        std_test = test_stats["std"].to_numpy()
        plt.plot(
            epochs_test,
            mean_test,
            label="Test",
            linewidth=2
        )
        plt.fill_between(
            epochs_test,
            mean_test - std_test,
            mean_test + std_test,
            alpha=0.2
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_losses_over_epochs(
    csv_files: list,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    title: str = "Mean Loss Over Epochs Across Different Seeds"
):
    all_dfs = []
    for path in csv_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: {path!r}")
        df = pd.read_csv(path)
        seed_label = os.path.splitext(os.path.basename(path))[0]
        df["seed"] = seed_label
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    combined["epoch"] = combined["epoch"].astype(int)
    combined["loss"] = combined["loss"].astype(float)
    combined["mode"] = combined["mode"].astype(str)
   
    stats = combined.groupby(["mode", "epoch"])["loss"].agg(["mean", "std"]).reset_index()
    train_stats = stats[stats["mode"] == "train"].sort_values("epoch")
    test_stats = stats[stats["mode"] == "test"].sort_values("epoch")

    plt.figure(figsize=(8, 6))

    if not train_stats.empty:
        epochs_train = train_stats["epoch"].to_numpy()
        mean_train = train_stats["mean"].to_numpy()
        std_train = train_stats["std"].to_numpy()
        plt.plot(
            epochs_train,
            mean_train,
            label="Train",
            linewidth=2
        )
        plt.fill_between(
            epochs_train,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.2
        )

    if not test_stats.empty:
        epochs_test = test_stats["epoch"].to_numpy()
        mean_test = test_stats["mean"].to_numpy()
        std_test = test_stats["std"].to_numpy()
        plt.plot(
            epochs_test,
            mean_test,
            label="Test",
            linewidth=2
        )
        plt.fill_between(
            epochs_test,
            mean_test - std_test,
            mean_test + std_test,
            alpha=0.2
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    # plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_final_accuracy_by_gamesize(
    csv_files: list,
    ylabel: str = "Accuracy",
    title: str = "Test communication success rate for different game sizes"
):
    """
    This function plots test accuracy at last epoch for every gamesize. 
    The primary grouping on the x-axis is the agent type. 
    Within each group, bars are colored to represent different game sizes. 
    A legend is provided to map colors to game sizes.
    """
    results = {}
    
    for path in csv_files:
        if not os.path.isfile(path):
            print(f"Warning: File not found, skipping. {path!r}")
            continue
            
        match = re.search(r'gamesize(\d+)_(\w+)_gs_seed', path)
        if not match:
            print(f"Warning: Could not parse file, skipping. Name must contain 'gamesize<N>_<agent>_gs_seed'. {path!r}")
            continue
        
        game_size = int(match.group(1))
        agent_type = match.group(2)
        
        try:
            df = pd.read_csv(path)
            last_epoch = df['epoch'].max()
            last_test_acc = df[(df['epoch'] == last_epoch) & (df['mode'] == 'test')]['acc'].iloc[0]
            
            results.setdefault(agent_type, {}).setdefault(game_size, []).append(last_test_acc)
        except Exception as e:
            print(f"Warning: Could not process file {path!r}. Error: {e}")

    if not results:
        print("No data to plot. Check file paths and naming conventions.")
        return

    agent_types = sorted(results.keys())
    all_game_sizes = sorted(list(set(gs for agent in results.values() for gs in agent.keys())))
    
    n_agents = len(agent_types)
    n_gamesizes = len(all_game_sizes)

    colors_to_use = ['#76c893', '#1a759f', '#d9ed92']
    plot_background_color = '#f0f0f0'

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor(plot_background_color)
    
    bar_width = 0.8 / n_gamesizes
    agent_indices = np.arange(n_agents)

    for i, game_size in enumerate(all_game_sizes):
        offset = (i - (n_gamesizes - 1) / 2) * bar_width
        
        means = [np.mean(results[agent].get(game_size, [np.nan])) for agent in agent_types]
        stds = [np.std(results[agent].get(game_size, [np.nan])) for agent in agent_types]

        color = colors_to_use[i % len(colors_to_use)]
        
        ax.bar(agent_indices + offset, means, width=bar_width, 
               yerr=stds, color=color, label=f'{game_size}', capsize=5)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(agent_indices)
    ax.set_xticklabels(agent_types, fontsize=12)
    ax.set_xlabel("")

    ax.set_ylim(0, 1.05)
    
    ax.legend(title="Game Size", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


if __name__ == "__main__":
    csv_list = [
        "logs/csv/2025-06-16/gamesize5_bee_gs_seed42.csv",
        "logs/csv/2025-06-16/gamesize5_bee_gs_seed123.csv",
        "logs/csv/2025-06-16/gamesize5_bee_gs_seed2025.csv",
        "logs/csv/2025-06-16/gamesize10_bee_gs_seed42.csv",
        "logs/csv/2025-06-16/gamesize10_bee_gs_seed123.csv",
        "logs/csv/2025-06-16/gamesize10_bee_gs_seed2025.csv",
        "logs/csv/2025-06-16/gamesize20_bee_gs_seed42.csv",
        "logs/csv/2025-06-16/gamesize20_bee_gs_seed123.csv",
        "logs/csv/2025-06-16/gamesize20_bee_gs_seed2025.csv",
        "logs/csv/2025-06-16/gamesize5_human_gs_seed42.csv",
        "logs/csv/2025-06-16/gamesize5_human_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize5_human_gs_seed2025.csv",
        "logs/csv/2025-06-16/gamesize10_human_gs_seed42.csv",
        "logs/csv/2025-06-16/gamesize10_human_gs_seed123.csv",
        "logs/csv/2025-06-16/gamesize10_human_gs_seed2025.csv",
        "logs/csv/2025-06-16/gamesize20_human_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize20_human_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize20_human_gs_seed2025.csv",

    ]

    # plot_accuracies_over_epochs(
    #     csv_files=csv_list,
    #     title="Mean Accuracy Over Epochs in Bee with Lr 0.001"
    # )

    # plot_losses_over_epochs(csv_files=csv_list, title="Mean Loss Over Epochs in Bee with Lr 0.001")

    plot_final_accuracy_by_gamesize(csv_files=csv_list)
