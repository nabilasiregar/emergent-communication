import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    csv_list = [
        "logs/csv/2025-06-11/bee_gs_seed42_without_tanh.csv",
        "logs/csv/2025-06-11/bee_gs_seed123_without_tanh.csv",
        "logs/csv/2025-06-11/bee_gs_seed2025_without_tanh.csv"
    ]

    plot_accuracies_over_epochs(
        csv_files=csv_list,
        title="Mean Accuracy"
    )

    plot_losses_over_epochs(csv_files=csv_list, title="Mean Loss")
