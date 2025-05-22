import argparse
import yaml
import optuna
from optuna.pruners import MedianPruner
import torch
import egg.core as core
from torch.utils.data import DataLoader, random_split
import game as game_module
from helpers import set_seed, collate_fn
from egg.core import Callback

def get_args():
    p = argparse.ArgumentParser(description="Tune for max accuracy")
    p.add_argument("--config", type=str, default="config.yaml",
                   help="YAML file")
    args, _ = p.parse_known_args()
    return args

class AccCB(Callback):
    def __init__(self):
        super().__init__()
        self.acc = 0.0
    def on_validation_end(self, loss, logs, epoch):
        a = logs.aux.get("acc", 0.0)
        # logs.aux["acc"] is per-sample; take the mean
        self.acc = a.mean().item() if torch.is_tensor(a) else float(a)

def objective(trial, dataset, cfg):
    lr   = trial.suggest_categorical("lr",   cfg["lr_choices"])
    temp = trial.suggest_categorical("temperature", cfg["temp_choices"])

    cli = []
    hp = dict(
        communication_type=cfg.get("communication_type","bee"),
        seed=cfg["seed"],
        train_data=cfg["train_data"],
        mode=cfg.get("mode","gs"),
        n_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
        lr=lr,
        temperature=temp
    )
    for k,v in hp.items():
        cli += [f"--{k}", str(v)]

    egg_opts = game_module.get_params(cli)
    set_seed(egg_opts.seed)

    total = len(dataset)
    n_val = int(total * cfg["validation_split"])
    n_train = total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(egg_opts.seed)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(train_ds, batch_size=egg_opts.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=cfg.get("num_workers",4),
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=egg_opts.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=cfg.get("num_workers",4),
                            pin_memory=True)

    model, callbacks = game_module.get_game(egg_opts)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=egg_opts.lr)

    acc_cb = AccCB()
    trainer = core.Trainer(
        game=model,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        device=device,
        callbacks=callbacks + [acc_cb]
    )

    trainer.train(n_epochs=egg_opts.n_epochs)
    core.close()
    return acc_cb.acc

def main():
    args = get_args()
    cfg  = yaml.safe_load(open(args.config))
    dataset = torch.load(cfg["train_data"])
    pruner = MedianPruner(n_warmup_steps=5)  # don’t prune the first 5 steps

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda t: objective(t, dataset, cfg),
                   n_trials=cfg.get("n_trials", 20))

    print("== Best Hyperparameters ==")
    print(f"  lr          = {study.best_params['lr']}")
    print(f"  temperature = {study.best_params['temperature']}")
    print(f"  val_accuracy= {study.best_value:.4f}")

if __name__ == "__main__":
    main()

