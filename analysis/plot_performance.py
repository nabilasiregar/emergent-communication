import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re
import json
import math

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

def extract_last_epoch_test_accuracy(file_path):
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
            
            last_test_acc = None
            for line in reversed(lines):
                if 'test_acc' in line:
                    try:
                        data = json.loads(line.strip())
                        if 'test_acc' in data:
                            last_test_acc = data['test_acc']
                            break
                    except json.JSONDecodeError:
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

def process_files(file_paths, extract_params_func, get_color_func):
    grouped_results = {}
    
    for path in file_paths:
        params = extract_params_func(path)
        param_key = f"maxlen{params[0]}_vocab{params[1]}" if len(params) == 2 else f"gamesize{params[0]}"
        
        if param_key not in grouped_results:
            grouped_results[param_key] = {
                'paths': [],
                'params': params
            }
        
        grouped_results[param_key]['paths'].append(path)
    
    final_results = {}
    
    for param_key, param_data in grouped_results.items():
        accuracies = []
        
        for path in param_data['paths']:
            acc = extract_last_epoch_test_accuracy(path)
            if acc is not None:
                accuracies.append(acc)
        
        if len(accuracies) > 0:
            params = param_data['params']
            final_results[param_key] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0,
                'values': accuracies,
                'n': len(accuracies),
                'params': params,
                'color': get_color_func(*params)
            }
            print(f"{param_key}: {accuracies} -> mean={final_results[param_key]['mean']:.4f}, std={final_results[param_key]['std']:.4f}")
        else:
            print(f"Warning: No valid accuracies found for {param_key}")
    
    return final_results

def create_test_accuracy_bar_plot(file_paths, save_path=None):
    def parse_filename_bee_human(file_path, comm_type):
        filename = file_path.split('/')[-1]
        
        gamesize_match = re.search(r'gamesize(\d+)', filename)
        gamesize = int(gamesize_match.group(1)) if gamesize_match else None
        
        if comm_type == 'human':
            maxlen_match = re.search(r'maxlen(\d+)', filename)
            maxlen = int(maxlen_match.group(1)) if maxlen_match else None
            return gamesize, maxlen
        else:
            return gamesize, None
    
    def get_color_bee_human(gamesize, maxlen=None):
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
        
        if maxlen is None:
            return bee_colors.get(gamesize, '#d62728')
        else:
            return human_colors.get((maxlen, gamesize), '#d62728')
    
    grouped_results = {}
    
    for comm_type, paths in file_paths.items():
        if comm_type not in grouped_results:
            grouped_results[comm_type] = {}
        
        for path in paths:
            gamesize, maxlen = parse_filename_bee_human(path, comm_type)
            
            if comm_type == 'human':
                param_key = f"maxlen{maxlen}_gamesize{gamesize}"
            else:
                param_key = f"gamesize{gamesize}"
            
            if param_key not in grouped_results[comm_type]:
                grouped_results[comm_type][param_key] = {
                    'paths': [],
                    'gamesize': gamesize,
                    'maxlen': maxlen
                }
            
            grouped_results[comm_type][param_key]['paths'].append(path)
    
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
                    'gamesize': param_data['gamesize'],
                    'maxlen': param_data['maxlen']
                }
                print(f"{comm_type} {param_key}: {accuracies} -> mean={final_results[comm_type][param_key]['mean']:.4f}, std={final_results[comm_type][param_key]['std']:.4f}")
            else:
                print(f"Warning: No valid accuracies found for {comm_type} {param_key}")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bee_results = []
    human_results = []
    
    for comm_type in ['bee', 'human']:
        if comm_type in final_results:
            sorted_params = sorted(final_results[comm_type].keys())
            
            for param_key in sorted_params:
                data = final_results[comm_type][param_key]
                
                if comm_type == 'bee':
                    gamesize = data['gamesize']
                    color = get_color_bee_human(gamesize)
                    
                    bee_results.append({
                        'gamesize': gamesize,
                        'mean': data['mean'],
                        'std': data['std'],
                        'color': color,
                        'n': data['n']
                    })
                else:
                    maxlen = data['maxlen']
                    gamesize = data['gamesize']
                    color = get_color_bee_human(gamesize, maxlen)
                    
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
    
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Bee', 'Human'], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)  
    
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
        fontsize=16,
        title_fontsize=16,
        ncol=3,
        labelspacing=0.6,
        handletextpad=0.4,
        columnspacing=1.0,
        loc='upper center',
        bbox_to_anchor=(0.25, -0.15),
        frameon=False
    )
    ax.add_artist(leg1)
    
    max_per_col = 4
    ncols_human = math.ceil(len(human_legend_elements) / max_per_col)
    leg2 = ax.legend(
        handles=human_legend_elements,
        title='Human (Max Len, Game Size)',
        fontsize=16,
        title_fontsize=16,
        ncol=ncols_human,
        labelspacing=0.6,
        handletextpad=0.4,
        columnspacing=1.0,
        loc='upper center',
        bbox_to_anchor=(0.75, -0.1),
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

def create_human_vocab_maxlen_plot(file_paths, save_path=None):
    def parse_filename_vocab_maxlen(file_path):
        filename = file_path.split('/')[-1]
        
        maxlen = 10
        vocab_size = 100
        
        maxlen_match = re.search(r'maxlen(\d+)', filename)
        if maxlen_match:
            maxlen = int(maxlen_match.group(1))
        
        vocab_match = re.search(r'vocab(\d+)', filename)
        if vocab_match:
            vocab_size = int(vocab_match.group(1))
        
        vocab_size_match = re.search(r'vocab_size(\d+)', filename)
        if vocab_size_match:
            vocab_size = int(vocab_size_match.group(1))
        
        return maxlen, vocab_size
    
    def get_color_vocab_maxlen(maxlen, vocab_size):
        color_map = {
            (2, 20): '#99CCFF',
            (2, 50): '#66b2ff', 
            (2, 100): '#0073E6',
            (4, 20): '#CCF2F2',
            (4, 50): '#99E6E6',
            (4, 100): '#66D9D9', 
            (6, 20): '#FFD1A1',
            (6, 50): '#FFB366',
            (6, 100): '#FF9933',
            (10, 20): '#B399FF', 
            (10, 50): '#9966FF',
            (10, 100): '#8000FF'
        }
        
        return color_map.get((maxlen, vocab_size), '#808080')
    
    final_results = process_files(file_paths, parse_filename_vocab_maxlen, get_color_vocab_maxlen)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sorted_keys = sorted(final_results.keys(), 
                        key=lambda x: (final_results[x]['params'][0], final_results[x]['params'][1]))
    
    x_pos = np.arange(len(sorted_keys))
    width = 0.8
    
    bars = []
    for i, param_key in enumerate(sorted_keys):
        data = final_results[param_key]
        bar = ax.bar(x_pos[i], data['mean'], width, 
                    yerr=data['std'], capsize=3,
                    color=data['color'], alpha=0.8, 
                    edgecolor='black', linewidth=1)
        bars.append((bar, data['params'][0], data['params'][1]))
    
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.tick_params(axis='y', labelsize=16)
    
    legend_elements = []
    for bar, maxlen, vocab_size in bars:
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                                           facecolor=bar[0].get_facecolor(), 
                                           edgecolor='black', 
                                           label=f'({maxlen}, {vocab_size})'))
    
    max_per_col = 3
    ncols = math.ceil(len(legend_elements) / max_per_col)
    
    leg = ax.legend(
        handles=legend_elements,
        title='Human (Max Len, Vocab Size)',
        fontsize=16,
        title_fontsize=16,
        ncol=ncols,
        labelspacing=0.6,
        handletextpad=0.4,
        columnspacing=1.0,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        frameon=False
    )
    
    all_means = [data['mean'] for data in final_results.values()]
    all_stds = [data['std'] for data in final_results.values()]
    
    if all_means:
        y_min = min(all_means) - max(all_stds) - 0.05
        y_max = max(all_means) + max(all_stds) + 0.05
        ax.set_ylim(max(0, y_min), min(1, y_max))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig, ax

def create_ablation_test_accuracy_plot(ablation_files, save_path: str = None):
    """
    Create a bar plot showing the effect of ablations on test accuracy at the last epoch.
    """
    def classify_ablation_type(file_path):
        """Classify the ablation type based on filename"""
        filename = file_path.split('/')[-1].lower()
        
        if 'zerodistance' in filename:
            return 'zeroed-out distance'
        elif 'binneddistance' in filename or 'binnedistance' in filename:
            return 'binned distance'
        else:
            return 'baseline'
    
    ablation_colors = {
        'baseline': '#0073e6',
        'zeroed-out distance': '#ff9933',
        'binned distance': '#9966ff'
    }
    
    grouped_results = {}
    
    for comm_type, paths in ablation_files.items():
        if comm_type not in grouped_results:
            grouped_results[comm_type] = {}
        
        # Group by ablation type
        for path in paths:
            ablation_type = classify_ablation_type(path)
            
            if ablation_type not in grouped_results[comm_type]:
                grouped_results[comm_type][ablation_type] = []
            
            grouped_results[comm_type][ablation_type].append(path)
    
    final_results = {}
    
    for comm_type, ablation_groups in grouped_results.items():
        final_results[comm_type] = {}
        
        for ablation_type, paths in ablation_groups.items():
            accuracies = []
            
            for path in paths:
                acc = extract_last_epoch_test_accuracy(path)
                if acc is not None:
                    accuracies.append(acc)
            
            if len(accuracies) > 0:
                final_results[comm_type][ablation_type] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0,
                    'values': accuracies,
                    'n': len(accuracies)
                }
                print(f"{comm_type} {ablation_type}: {accuracies} -> mean={final_results[comm_type][ablation_type]['mean']:.4f}, std={final_results[comm_type][ablation_type]['std']:.4f}")
            else:
                print(f"Warning: No valid accuracies found for {comm_type} {ablation_type}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ablation_order = ['baseline', 'zeroed-out distance', 'binned distance']
    
    x_pos = np.arange(2)
    width = 0.25
    
    for i, ablation_type in enumerate(ablation_order):
        bee_mean = bee_std = human_mean = human_std = 0
        bee_n = human_n = 0
        
        if 'bee' in final_results and ablation_type in final_results['bee']:
            bee_data = final_results['bee'][ablation_type]
            bee_mean = bee_data['mean']
            bee_std = bee_data['std']
            bee_n = bee_data['n']
        
        if 'human' in final_results and ablation_type in final_results['human']:
            human_data = final_results['human'][ablation_type]
            human_mean = human_data['mean']
            human_std = human_data['std']
            human_n = human_data['n']
        
        color = ablation_colors[ablation_type]
        
        if bee_n > 0:
            ax.bar(x_pos[0] + i * width - width, bee_mean, width, 
                   yerr=bee_std, capsize=3, color=color, alpha=0.8, 
                   edgecolor='black', linewidth=1, label=ablation_type if i == 0 else "")
        
        if human_n > 0:
            ax.bar(x_pos[1] + i * width - width, human_mean, width, 
                   yerr=human_std, capsize=3, color=color, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Bee', 'Human'], fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    
    legend_elements = []
    for ablation_type in ablation_order:
        if any(ablation_type in final_results.get(comm_type, {}) for comm_type in ['bee', 'human']):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                               facecolor=ablation_colors[ablation_type], 
                                               edgecolor='black', alpha=0.8,
                                               label=ablation_type.title()))
    
    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.10),
            ncol=len(legend_elements),
            fontsize=16,
            frameon=False
        )
    
    all_means = []
    all_stds = []
    for comm_type in final_results.values():
        for data in comm_type.values():
            all_means.append(data['mean'])
            all_stds.append(data['std'])
    
    if all_means:
        y_min = max(0, min(all_means) - max(all_stds) - 0.05)
        y_max = min(1, max(all_means) + max(all_stds) + 0.05)
        ax.set_ylim(y_min, y_max)
    
    ax.set_axisbelow(True)
    
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
            'logs/csv/2025-07-02/gamesize5_bee_gs_seed42.csv',
            'logs/csv/2025-07-02/gamesize5_bee_gs_seed123.csv', 
            'logs/csv/2025-07-02/gamesize5_bee_gs_seed2025.csv',
            'logs/csv/2025-07-02/gamesize5_bee_gs_seed31.csv', 
            'logs/csv/2025-07-02/gamesize5_bee_gs_seed27.csv',
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed42.csv',
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed123.csv', 
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed2025.csv',
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed31.csv', 
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed27.csv',
            'logs/csv/2025-07-02/gamesize20_bee_gs_seed42.csv',
            'logs/csv/2025-07-02/gamesize20_bee_gs_seed123.csv', 
            'logs/csv/2025-07-02/gamesize20_bee_gs_seed2025.csv',
            'logs/csv/2025-07-02/gamesize20_bee_gs_seed31.csv', 
            'logs/csv/2025-07-02/gamesize20_bee_gs_seed27.csv'
        ],
        'human': [
            'logs/csv/2025-06-29/gamesize5_maxlen2_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen2_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen2_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen2_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen2_human_gs_seed27.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen4_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen4_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen4_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen4_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen4_human_gs_seed27.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen6_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen6_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen6_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen6_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen6_human_gs_seed27.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize5_maxlen10_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen10_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize5_maxlen10_human_gs_seed27.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen2_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen2_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen2_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen2_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen2_human_gs_seed27.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen4_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen4_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen4_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen4_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen4_human_gs_seed27.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen6_human_gs_seed42.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen6_human_gs_seed123.csv',
            'logs/csv/2025-06-29/gamesize10_maxlen6_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen6_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen6_human_gs_seed27.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen10_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen10_human_gs_seed27.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen2_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen2_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen2_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen2_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen2_human_gs_seed27.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen4_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen4_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen4_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen4_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen4_human_gs_seed27.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen6_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen6_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen6_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen6_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen6_human_gs_seed27.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize20_maxlen10_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen10_human_gs_seed31.csv',
            'logs/csv/2025-07-07/gamesize20_maxlen10_human_gs_seed27.csv'
        ]
    }

    create_test_accuracy_bar_plot(
        file_paths=files,
        save_path="test_communication_success.png"
    )

    vary_maxlen_vocab_filepaths = [
        "logs/csv/2025-07-07/maxlen2_vocab20_human_gs_seed42.csv",
        "logs/csv/2025-07-07/maxlen2_vocab20_human_gs_seed123.csv",
        "logs/csv/2025-07-07/maxlen2_vocab20_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen2_vocab20_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen2_vocab20_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen2_vocab50_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen2_vocab50_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen2_vocab50_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen2_vocab50_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen2_vocab50_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen2_vocab100_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen2_vocab100_human_gs_seed31.csv",
        "logs/csv/2025-06-22/maxlen2_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen2_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen2_human_gs_seed2025.csv",

        "logs/csv/2025-07-08/maxlen4_vocab20_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen4_vocab20_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen4_vocab20_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen4_vocab20_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen4_vocab20_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen4_vocab50_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen4_vocab50_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen4_vocab50_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen4_vocab50_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen4_vocab50_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen4_vocab100_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen4_vocab100_human_gs_seed31.csv",
        "logs/csv/2025-06-22/maxlen4_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen4_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen4_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen6_vocab20_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen6_vocab20_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen6_vocab20_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen6_vocab20_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen6_vocab20_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen6_vocab50_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen6_vocab50_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen6_vocab50_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen6_vocab50_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen6_vocab50_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen6_vocab100_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen6_vocab100_human_gs_seed31.csv",
        "logs/csv/2025-06-22/maxlen6_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen6_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen6_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen10_vocab20_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen10_vocab20_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen10_vocab20_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen10_vocab20_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen10_vocab20_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen10_vocab50_human_gs_seed42.csv",
        "logs/csv/2025-07-08/maxlen10_vocab50_human_gs_seed123.csv",
        "logs/csv/2025-07-08/maxlen10_vocab50_human_gs_seed2025.csv",
        "logs/csv/2025-07-08/maxlen10_vocab50_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen10_vocab50_human_gs_seed31.csv",
        "logs/csv/2025-07-08/maxlen10_vocab100_human_gs_seed27.csv",
        "logs/csv/2025-07-08/maxlen10_vocab100_human_gs_seed31.csv",
        "logs/csv/2025-06-22/maxlen10_human_gs_seed42.csv",
        "logs/csv/2025-06-23/maxlen10_human_gs_seed123.csv",
        "logs/csv/2025-06-23/maxlen10_human_gs_seed2025.csv"
    ]

    fig, ax = create_human_vocab_maxlen_plot(
        vary_maxlen_vocab_filepaths,
        save_path="human_communication_maxlen_vocab_plot.png"
    )

    ablation_files = {
        'bee': [
            # baseline
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed42.csv',
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed123.csv', 
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed2025.csv',
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed31.csv', 
            'logs/csv/2025-07-02/gamesize10_bee_gs_seed27.csv',
            # zeroed-out distance
            "logs/csv/2025-07-02/zerodistance_bee_gs_seed27.csv",
            "logs/csv/2025-07-02/zerodistance_bee_gs_seed31.csv",
            "logs/csv/2025-07-02/zerodistance_bee_gs_seed42.csv",
            "logs/csv/2025-07-02/zerodistance_bee_gs_seed123.csv",
            "logs/csv/2025-07-02/zerodistance_bee_gs_seed2025.csv",
            # binned distance
            "logs/csv/2025-07-02/binneddistance_bee_gs_seed27.csv",
            "logs/csv/2025-07-02/binneddistance_bee_gs_seed31.csv",
            "logs/csv/2025-07-02/binneddistance_bee_gs_seed42.csv",
            "logs/csv/2025-07-02/binneddistance_bee_gs_seed123.csv",
            "logs/csv/2025-07-02/binneddistance_bee_gs_seed2025.csv"

        ],
        'human': [
            # baseline
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed42.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed123.csv',
            'logs/csv/2025-06-30/gamesize10_maxlen10_human_gs_seed2025.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen10_human_gs_seed27.csv',
            'logs/csv/2025-07-07/gamesize10_maxlen10_human_gs_seed31.csv',
            # zeroed-out distance
            "logs/csv/2025-07-02/zerodistance_human_gs_seed27.csv",
            "logs/csv/2025-07-02/zerodistance_human_gs_seed31.csv",
            "logs/csv/2025-07-02/zerodistance_human_gs_seed42.csv",
            "logs/csv/2025-07-02/zerodistance_human_gs_seed123.csv",
            "logs/csv/2025-07-02/zerodistance_human_gs_seed2025.csv",
            # binned distance
            "logs/csv/2025-07-02/binneddistance_human_gs_seed27.csv",
            "logs/csv/2025-07-02/binneddistance_human_gs_seed31.csv",
            "logs/csv/2025-07-02/binneddistance_human_gs_seed42.csv",
            "logs/csv/2025-07-02/binneddistance_human_gs_seed123.csv",
            "logs/csv/2025-07-02/binneddistance_human_gs_seed2025.csv"
        ]
    }

    create_ablation_test_accuracy_plot(
        ablation_files,
        save_path="ablation_study.png"
    )
