import argparse
import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import egg.core as core

import game
from utils.helpers import set_seed, collate_fn


class MetricsCallback(core.Callback):
    """
    Callback to track the best validation loss during a trial.
    """
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')

    def on_validation_end(self, loss: float, logs: dict, epoch: int):
        if loss < self.best_loss:
            self.best_loss = loss


class OptunaPruningCallback(core.Callback):
    """
    Callback to integrate Optuna's trial pruning mechanism with EGG's training loop.
    """
    def __init__(self, trial: optuna.trial.Trial):
        super().__init__()
        self.trial = trial
        self.step = 0

    def on_validation_end(self, loss: float, logs: dict, epoch: int):
        self.step += 1
        self.trial.report(loss, self.step)

        if self.trial.should_prune():
            message = f"Trial was pruned at step {self.step} with loss {loss}."
            raise TrialPruned(message)


def objective(trial: optuna.trial.Trial, cfg: dict, train_dataset: Dataset):
    hidden_size = trial.suggest_categorical("hidden_size", cfg["hidden_choices"])
    lr = trial.suggest_float("lr", *cfg["lr_range"], log=True)
    temperature = trial.suggest_float("temperature", *cfg["temperature_range"])

    seed = 42
    hp = {
        "communication_type": cfg["communication_type"],
        "random_seed":        seed,
        "train_data":         cfg["train_data"],
        "mode":               cfg["mode"],
        "n_epochs":           cfg["n_epochs"],
        "batch_size":         cfg["batch_size"],
        "lr":                 lr,
        "temperature":        temperature,
        "sender_hidden":      hidden_size,
        "receiver_hidden":    hidden_size
    }
    cli_params = [f"--{k}={v}" for k, v in hp.items()]
    opts = game.get_params(cli_params)
    set_seed(opts.random_seed)

    n_samples = len(train_dataset)
    val_size = int(cfg["validation_split"] * n_samples)
    train_size = n_samples - val_size
    train_ds, val_ds = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=opts.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cfg.get("num_workers", 4))

    model, callbacks = game.get_game(opts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = core.build_optimizer(model.parameters())

    metrics_cb = MetricsCallback()
    pruning_cb = OptunaPruningCallback(trial)

    trainer = core.Trainer(
        game=model,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        device=device,
        callbacks=callbacks + [metrics_cb, pruning_cb]
    )

    try:
        trainer.train(n_epochs=opts.n_epochs)
    except TrialPruned:
        return float('inf')

    core.close()

    return metrics_cb.best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_tune_bee.yaml", help="Config file for tuning")
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    dataset = torch.load(cfg["train_data"])

    pruner = MedianPruner(n_warmup_steps=cfg["n_warmup_steps"])
    sampler = TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)

    print(f"Starting Optuna Study: {cfg['n_trials']} trials")
    study.optimize(lambda trial: objective(trial, cfg, dataset), n_trials=cfg["n_trials"])

    print("\n\n--- STUDY COMPLETE ---")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("Best Hyperparameters")
    for key, value in study.best_params.items():
        print(f"  {key.ljust(20)}: {value}")


if __name__ == "__main__":
    main()