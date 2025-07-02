import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re
import json

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

def plot_accuracy_over_epochs_by_condition(
    csv_files: list,
    xlabel: str = "Epoch",
    ylabel: str = "Test Accuracy",
    title: str = "Impact of Max Length on Human Agent Learning"
):
    """
    Plots mean test accuracy over epochs for different experimental conditions.

    This function reads multiple CSV files, determines the experimental condition
    from the filename and groups the data by these conditions. It then plots a separate
    learning curve for each condition, averaged over any different seeds.
    """
    all_dfs = []

    label_map = {
        'maxlen10_human_gs': 'maxlen = 10',
        'maxlen2_human_gs':  'maxlen = 2',
        'maxlen4_human_gs':  'maxlen = 4',
        'maxlen6_human_gs':  'maxlen = 6'
    }

    for path in csv_files:
        if not os.path.isfile(path):
            print(f"Warning: File not found, skipping. {path!r}")
            continue
            
        df = pd.read_csv(path)
        
        basename = os.path.basename(path)
        match = re.search(r'(.*)_seed\d+', basename)
        if match:
            raw_label = match.group(1)
        else:
            raw_label = os.path.splitext(basename)[0]
            
        condition_label = label_map.get(raw_label, raw_label)
            
        df["condition"] = condition_label
        all_dfs.append(df)
        
    if not all_dfs:
        print("No valid CSV files to plot.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    combined["epoch"] = combined["epoch"].astype(int)
    combined["acc"] = combined["acc"].astype(float)
    combined["mode"] = combined["mode"].astype(str)
    
    test_data = combined[combined["mode"] == "test"].copy()
    stats = test_data.groupby(["condition", "epoch"])["acc"].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    try:
        conditions = sorted(stats["condition"].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
    except (AttributeError, TypeError):
        conditions = sorted(stats["condition"].unique())

    colors = plt.cm.get_cmap('viridis', len(conditions))

    for i, condition in enumerate(conditions):
        condition_stats = stats[stats["condition"] == condition].sort_values("epoch")
        
        if not condition_stats.empty:
            epochs = condition_stats["epoch"].to_numpy()
            mean_acc = condition_stats["mean"].to_numpy()
            std_acc = condition_stats["std"].fillna(0).to_numpy()
            
            ax.plot(
                epochs,
                mean_acc,
                label=condition,
                linewidth=2,
                color=colors(i / len(conditions))
            )
            ax.fill_between(
                epochs,
                mean_acc - std_acc,
                mean_acc + std_acc,
                alpha=0.2,
                color=colors(i / len(conditions))
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, title="Conditions")
    
    plt.tight_layout()
    plt.show()

def create_test_accuracy_bar_plot(file_paths, title: str = "Test Communication Success", figsize: tuple = (15, 8), save_path: str = None):
    def extract_last_epoch_test_accuracy(file_path):
        """Extract test accuracy from the last epoch in a csv file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                test_data = df[df['mode'] == 'test']
                if test_data.empty:
                    print(f"Warning: No test data found in {file_path}")
                    return None
                
                last_test_acc = test_data['acc'].iloc[-1]
                return last_test_acc
            
            else:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # find the last line that contains test accuracy
                last_test_acc = None
                for line in reversed(lines):
                    if 'test_acc' in line:
                        try:
                            # try to parse as JSON
                            data = json.loads(line.strip())
                            if 'test_acc' in data:
                                last_test_acc = data['test_acc']
                                break
                        except json.JSONDecodeError:
                            # if not JSON, try to parse as key-value pairs
                            if 'test_acc:' in line:
                                parts = line.split('test_acc:')
                                if len(parts) > 1:
                                    acc_str = parts[1].strip().split()[0]
                                    last_test_acc = float(acc_str)
                                    break
                
                if last_test_acc is None:
                    print(f"Warning: Could not find test_acc in {file_path}")
                    return None
                
                return last_test_acc
                
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            return None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def parse_filename(file_path, comm_type):
        """Parse filename to extract gamesize and maxlen parameters"""
        filename = file_path.split('/')[-1]
        
        # Extract gamesize
        gamesize_match = re.search(r'gamesize(\d+)', filename)
        gamesize = int(gamesize_match.group(1)) if gamesize_match else None
        
        if comm_type == 'human':
            # Extract maxlen for human
            maxlen_match = re.search(r'maxlen(\d+)', filename)
            maxlen = int(maxlen_match.group(1)) if maxlen_match else None
            return gamesize, maxlen
        else:
            # For bee, no maxlen
            return gamesize, None

    bee_colors = {
            5: '#FFD1A1',
            10: '#FF9933',
            20: '#E66100'
        }

    human_colors = {
        (2, 5): '#CCE5FF',
        (4, 5): '#99CCFF',
        (6, 5): '#66b2ff', 
        (10, 5): '#0073E6',
        (2, 10): '#CCF2F2',
        (4, 10): '#99E6E6',
        (6, 10): '#66D9D9', 
        (10, 10): '#00B3B3',
        (2, 20): '#F2E6FF',
        (4, 20): '#DABFFF',
        (6, 20): '#B399FF', 
        (10, 20): '#9966FF'
    }

    # Group files by communication type and parameters
    grouped_results = {}
    
    for comm_type, paths in file_paths.items():
        if comm_type not in grouped_results:
            grouped_results[comm_type] = {}
        
        # Group by parameters
        for path in paths:
            gamesize, maxlen = parse_filename(path, comm_type)
            
            if comm_type == 'human':
                param_key = f"maxlen{maxlen}_gamesize{gamesize}"
                legend_label = f"(maxlen={maxlen}, gamesize={gamesize})"
            else:
                param_key = f"gamesize{gamesize}"
                legend_label = f"(gamesize={gamesize})"
            
            if param_key not in grouped_results[comm_type]:
                grouped_results[comm_type][param_key] = {
                    'paths': [],
                    'legend': legend_label
                }
            
            grouped_results[comm_type][param_key]['paths'].append(path)
    
    # Extract accuracies for each group
    final_results = {}
    
    for comm_type, param_groups in grouped_results.items():
        final_results[comm_type] = {}
        
        for param_key, param_data in param_groups.items():
            accuracies = []
            
            for path in param_data['paths']:
                acc = extract_last_epoch_test_accuracy(path)
                if acc is not None:
                    accuracies.append(acc)
            
            if len(accuracies) > 0:
                final_results[comm_type][param_key] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0,
                    'values': accuracies,
                    'n': len(accuracies),
                    'legend': param_data['legend']
                }
                print(f"{comm_type} {param_key}: {accuracies} -> mean={final_results[comm_type][param_key]['mean']:.4f}, std={final_results[comm_type][param_key]['std']:.4f}")
            else:
                print(f"Warning: No valid accuracies found for {comm_type} {param_key}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bee_results = []
    human_results = []
    
    for comm_type in ['bee', 'human']:
        if comm_type in final_results:
            sorted_params = sorted(final_results[comm_type].keys())
            
            for param_key in sorted_params:
                data = final_results[comm_type][param_key]
                
                if comm_type == 'bee':
                    gamesize = int(param_key.split('gamesize')[1])
                    color = bee_colors.get(gamesize, '#d62728') 
                    
                    bee_results.append({
                        'gamesize': gamesize,
                        'mean': data['mean'],
                        'std': data['std'],
                        'color': color,
                        'n': data['n']
                    })
                else:
                    maxlen = int(param_key.split('maxlen')[1].split('_')[0])
                    gamesize = int(param_key.split('gamesize')[1])
                    color = human_colors.get((maxlen, gamesize), '#d62728') 
                    
                    human_results.append({
                        'maxlen': maxlen,
                        'gamesize': gamesize,
                        'mean': data['mean'],
                        'std': data['std'],
                        'color': color,
                        'n': data['n']
                    })
    
    bee_results.sort(key=lambda x: x['gamesize'])
    human_results.sort(key=lambda x: (x['gamesize'], x['maxlen']))

    x_pos = np.arange(2) 
    width = 0.12
    
    bee_bars = []
    for i, data in enumerate(bee_results):
        bar = ax.bar(x_pos[0] + i * width - (len(bee_results) - 1) * width / 2, 
                    data['mean'], width, yerr=data['std'], capsize=3,
                    color=data['color'], alpha=0.8, edgecolor='black', linewidth=1)
        bee_bars.append((bar, data['gamesize']))
    
    human_bars = []
    for i, data in enumerate(human_results):
        bar = ax.bar(x_pos[1] + i * width - (len(human_results) - 1) * width / 2, 
                    data['mean'], width, yerr=data['std'], capsize=3,
                    color=data['color'], alpha=0.8, edgecolor='black', linewidth=1)
        human_bars.append((bar, (data['maxlen'], data['gamesize'])))
    
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Bee', 'Human'], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)  
    
    bee_legend_elements = []
    for bar, gamesize in bee_bars:
        bee_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=bar[0].get_facecolor(), 
                                               edgecolor='black', label=str(gamesize)))
    
    human_legend_elements = []
    for bar, (maxlen, gamesize) in human_bars:
        human_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=bar[0].get_facecolor(), 
                                                 edgecolor='black', label=f'({maxlen}, {gamesize})'))
    
    leg1 = ax.legend(
        handles=bee_legend_elements,
        title='Bee (Game Size)',
        fontsize=12,
        title_fontsize=14,
        ncol=3,
        labelspacing=0.6,
        handletextpad=0.4,
        columnspacing=1.0,
        loc='upper center',
        bbox_to_anchor=(0.25, -0.15),
        frameon=False
    )
    ax.add_artist(leg1)
    
    leg2 = ax.legend(
        handles=human_legend_elements,
        title='Human (Max Len, Game Size)',
        fontsize=12,
        title_fontsize=14,
        ncol=4,
        labelspacing=0.6,
        handletextpad=0.4,
        columnspacing=1.0,
        loc='upper center',
        bbox_to_anchor=(0.75, -0.15),
        frameon=False
    )
    plt.subplots_adjust(bottom=0.1)

    all_means = [data['mean'] for data in bee_results] + [data['mean'] for data in human_results]
    all_stds = [data['std'] for data in bee_results] + [data['std'] for data in human_results]
    
    if all_means:
        y_min = min(all_means) - max(all_stds) - 0.05
        y_max = max(all_means) + max(all_stds) + 0.05
        ax.set_ylim(max(0, y_min), min(1, y_max))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig, ax


if __name__ == "__main__":
    csv_list = [
        # "logs/csv/2025-06-16/gamesize5_bee_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize5_bee_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize5_bee_gs_seed2025.csv",
        # "logs/csv/2025-06-16/gamesize10_bee_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize10_bee_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize10_bee_gs_seed2025.csv",
        # "logs/csv/2025-06-16/gamesize20_bee_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize20_bee_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize20_bee_gs_seed2025.csv",
        # "logs/csv/2025-06-16/gamesize5_human_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize5_human_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize5_human_gs_seed2025.csv",
        # "logs/csv/2025-06-16/gamesize10_human_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize10_human_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize10_human_gs_seed2025.csv",
        # "logs/csv/2025-06-16/gamesize20_human_gs_seed42.csv",
        # "logs/csv/2025-06-16/gamesize20_human_gs_seed123.csv",
        # "logs/csv/2025-06-16/gamesize20_human_gs_seed2025.csv",
        # "logs/csv/2025-06-16/maxlen2_human_gs_seed42.csv",
        # "logs/csv/2025-06-16/maxlen4_human_gs_seed42.csv",
        # "logs/csv/2025-06-16/maxlen6_human_gs_seed42.csv"
        "logs/csv/2025-06-22/maxlen2_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen2_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen2_human_gs_seed2025.csv",
        # "logs/csv/2025-06-22/maxlen4_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen4_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen4_human_gs_seed2025.csv",
        "logs/csv/2025-06-22/maxlen6_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen6_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen6_human_gs_seed2025.csv",
        "logs/csv/2025-06-22/maxlen10_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen10_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen10_human_gs_seed2025.csv"
    ]

    # plot_accuracies_over_epochs(
    #     csv_files=csv_list,
    #     title="Mean Accuracy Over Epochs in Human"
    # )

    # plot_losses_over_epochs(csv_files=csv_list, title="Mean Loss Over Epochs in Human")

    # plot_final_accuracy_by_gamesize(csv_files=csv_list)
    # plot_accuracy_over_epochs_by_condition(csv_files=csv_list)

    files = {
        'bee': [
            'logs/csv/2025-06-30/gamesize5_bee_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize5_bee_gs_seed123.csv', 
            'logs/csv/2025-06-30/gamesize5_bee_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize10_bee_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize10_bee_gs_seed123.csv', 
            'logs/csv/2025-06-30/gamesize10_bee_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize20_bee_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_bee_gs_seed123.csv', 
            'logs/csv/2025-06-30/gamesize20_bee_gs_seed2025.csv'
        ],
        'human': [
            'logs/csv/2025-06-29/gamesize5_maxlen2_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen2_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen2_human_gs_seed2025.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen4_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen4_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen4_human_gs_seed2025.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen6_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen6_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen6_human_gs_seed2025.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen10_human_gs_seed2025.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen2_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen2_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen2_human_gs_seed2025.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen4_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen4_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen4_human_gs_seed2025.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen6_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen6_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen6_human_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen2_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen2_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen2_human_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen4_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen4_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen4_human_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen6_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen6_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen6_human_gs_seed2025.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen10_human_gs_seed2025.csv'
        ]
    }

    create_test_accuracy_bar_plot(
        file_paths=files,
        title="Test Communication Success",
        save_path="test_communication_success.png"
    )
