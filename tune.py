"""
Reads default arguments and default hyperparams from an external YAML or JSON file (default: config.yaml).  
It runs N independent seeds for every candidate value and saves everything under logs/<timestamp>/** so runs never overwrite.

Usage example:
```bash
python experiment.py \
    --config config.yaml \
    --param learning_rate \
    --values 1e-4 5e-4 1e-3 \
    --num_runs 3
```
This will run 3×3 = 9 trainings.
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Any, List, Tuple

import yaml
import numpy as np
import optuna
from optuna.samplers import GridSampler
import torch

import game
import egg.core as core
from torch.utils.data import DataLoader, random_split
from helpers import collate_fn
from analysis.callbacks import DataLogger

import pdb

def load_config(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text()
    return yaml.safe_load(text)


def load_defaults(config: dict, tune_param: str) -> Tuple[List[str], Any]:
    """
    Turn every key!=tune_param into CLI flags [--key, value].
    Return that list and the config[tune_param] as default.
    """
    cli: List[str] = []
    default_value = config.get(tune_param)
    for k, v in config.items():
        if k == tune_param:
            continue
        # this is to get validation set from train set for tuning
        if k == "validation_split":
            continue
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        else:
            cli += [f"--{k}", str(v)]
    return cli, default_value


def get_final_acc(log_path: pathlib.Path) -> float:
    """Read the last test-mode acc from the DataLogger"""
    if not log_path.exists():
        return 0.0
    with log_path.open() as fh:
        try:
            records = json.load(fh)
        except json.JSONDecodeError:
            return 0.0
    acc = 0.0
    for rec in records:
        mode = rec.get("mode")
        if mode in ("test", "validation"):
            acc = float(rec.get("acc", 0.0))
    return acc


def run(
    base_cli: List[str],
    param_name: str,
    param_value: Any,
    run_tag: str,
    log_dir: pathlib.Path,
    config: dict
) -> float:
    """
    Launch one training run with base_cli + --param_name param_value.
    Returns the test accuracy from DataLogger.
    """
    cli = base_cli + [f"--{param_name}", str(param_value)]
    opts = game.get_params(cli)

    # set a reproducible but distinct seed per run
    opts.seed += hash(run_tag) % 10_000

    full_train = torch.load(opts.train_data)
    val_split = config.get("validation_split", None)
    if val_split is not None:
        # get validation set from train set not test set
        total = len(full_train)
        val_count = int(total * val_split)
        train_count = total - val_count
        train_ds, val_ds = random_split(
            full_train,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(opts.seed),
        )
    else:
        train_ds = full_train
        val_ds = torch.load(opts.validation_data)
    
    train_loader = DataLoader(train_ds, batch_size=opts.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=opts.batch_size,
                              shuffle=False,collate_fn=collate_fn)
    model, callbacks = game.get_game(opts)

    json_log = log_dir / f"{param_name}{param_value}_{run_tag}.json"
    callbacks.append(DataLogger(save_path=str(json_log)))
    callbacks.append(core.InteractionSaver(
        train_epochs=[],
        test_epochs=[opts.n_epochs],
        checkpoint_dir=str(log_dir / f"{param_name}{param_value}_{run_tag}")
    ))

    callbacks.append(core.ConsoleLogger(as_json=True))

    trainer = core.Trainer(
        game=model,
        optimizer=core.build_optimizer(model.parameters()),
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks
    )
    trainer.train(n_epochs=opts.n_epochs)
    core.close()

    return get_final_acc(json_log)


def make_objective(
    param_name: str,
    values: List[Any],
    base_cli: List[str],
    config: dict,
    log_root: pathlib.Path,
    num_runs: int
):
    def _objective(trial: optuna.trial.Trial) -> float:
        val = trial.suggest_categorical(param_name, values)
        accs = []
        for run_num in range(1, num_runs + 1):
            tag = f"run{run_num}"
            try:
                acc = run(base_cli, param_name, val, tag, log_root, config)
            except Exception as e:
                print(f"Error for {param_name}={val} run{run_num}: {e}")
                acc = 0.0
            accs.append(acc)
        return float(np.mean(accs))
    return _objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--param",
        required=True,
        help="Hyperparam to tune",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        required=True,
        help="List of values to try for that parameter",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Independent runs per value (different seeds)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=len(typed_values),
        help="How many Optuna trials to run in total",
    )

    args = parser.parse_args()

    settings = load_config(pathlib.Path(args.config))
    base_cli, _ = load_defaults(settings, args.param)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = pathlib.Path("logs") / stamp
    log_root.mkdir(parents=True, exist_ok=True)

    # convert CLI values to same type as default
    default_val = settings.get(args.param)
    typed_values = [type(default_val)(v) for v in args.values]

    study = optuna.create_study(
        direction="maximize",
        study_name=f"tune_{args.param}"
    )
    study.optimize(
        make_objective(…),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\nBest hyper-parameters:", study.best_params)
    print("Best mean validation acc:", study.best_value)


if __name__ == "__main__":
    main()
