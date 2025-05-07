import argparse
import json
import pathlib
from datetime import datetime
from typing import Any, List, Tuple

import yaml
import numpy as np
import torch
import random

import game
from game import set_seed
import egg.core as core
from torch.utils.data import DataLoader, random_split
from helpers import collate_fn
from analysis.callbacks import DataLogger


def load_config(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def load_defaults(config: dict, tune_param: str) -> Tuple[List[str], Any]:
    """
    Turn every key != tune_param and != validation_split into CLI flags,
    return that list and the config[tune_param] default.
    """
    cli: List[str] = []
    default_value = config.get(tune_param)
    for k, v in config.items():
        if k in (tune_param, 'validation_split'):
            continue
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        else:
            cli += [f"--{k}", str(v)]
    return cli, default_value


def get_final_acc(log_path: pathlib.Path) -> float:
    if not log_path.exists():
        return 0.0
    with log_path.open() as fh:
        try:
            records = json.load(fh)
        except json.JSONDecodeError:
            return 0.0
    acc = 0.0
    for rec in records:
        if rec.get('mode') in ('validation', 'test'):
            acc = float(rec.get('acc', 0.0))
    return acc


def run(
    base_cli: List[str],
    param_name: str,
    param_value: Any,
    run_idx: int,
    log_root: pathlib.Path,
    config: dict
) -> float:
    run_tag = f"run{run_idx}"
    cli = base_cli + [f"--{param_name}", str(param_value)]
    opts = game.get_params(cli)

    opts.random_seed = opts.seed
    set_seed(opts.seed)
    print(opts)

    # split train/validation
    full_train = torch.load(opts.train_data)
    vs = config.get('validation_split', None)
    if vs is not None:
        n_val = int(len(full_train) * vs)
        n_trn = len(full_train) - n_val
        train_ds, val_ds = random_split(
            full_train,
            [n_trn, n_val],
            generator=torch.Generator().manual_seed(opts.seed),
        )
    else:
        train_ds = full_train
        val_ds = torch.load(opts.validation_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opts.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model, callbacks = game.get_game(opts)

    # set up logging paths using run_tag
    run_dir = log_root / f"{param_name}{param_value}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json_log = run_dir / f"{param_name}{param_value}_{run_tag}.json"

    callbacks.append(DataLogger(save_path=str(json_log)))
    callbacks.append(core.InteractionSaver(
        train_epochs=[],
        test_epochs=[opts.n_epochs],
        checkpoint_dir=str(run_dir),
    ))
    callbacks.append(core.ConsoleLogger(as_json=True))

    trainer = core.Trainer(
        game=model,
        optimizer=core.build_optimizer(model.parameters()),
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
    )
    trainer.train(n_epochs=opts.n_epochs)
    core.close()

    return get_final_acc(json_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--param', required=True,
                        help='Hyperparam to tune')
    parser.add_argument('--values', nargs='+', required=True,
                        help='List of values to try')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs per value')
    args = parser.parse_args()

    config = load_config(pathlib.Path(args.config))
    base_cli, default_val = load_defaults(config, args.param)

    values = [type(default_val)(v) for v in args.values]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root = pathlib.Path('logs') / stamp
    log_root.mkdir(parents=True, exist_ok=True)

    for val in values:
        accs = []
        for i in range(1, args.n_runs + 1):
            acc = run(base_cli, args.param, val, i, log_root, config)
            accs.append(acc)
        avg = float(np.mean(accs))
        print(f"{args.param}={val} avg_acc={avg:.4f} over runs={accs}")

if __name__ == '__main__':
    main()
