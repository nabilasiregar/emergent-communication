import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import egg.core as core
from archs.agents import HumanSender, HumanReceiver, BeeSender, BeeReceiver
from utils.wrapper import MixedSymbolReceiverWrapper, MixedSymbolSenderWrapper
from utils.early_stopper import EarlyStopperLoss
from helpers import collate_fn, set_seed
from analysis.logger import CsvLogger
from egg.core import Trainer, build_optimizer
from egg.core.callbacks import (
    ConsoleLogger,
    InteractionSaver,
    TemperatureUpdater
)
from egg.core.language_analysis import (
    PrintValidationEvents
)
import pdb
def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication_type", choices=["bee", "human"], default="bee")

    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--train_data", type=str, default="data/train_data.pt", help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default="data/test_data.pt", help="Path to the validation data"
    )
    parser.add_argument(
        "--final_run",
        default=False,
        action="store_true",
        help="If this flag is passed, use the full training set and the separate validation file. Otherwise, split train into train/val.",
    )
   
    # arguments concerning the training method
    parser.add_argument(
        "--mode", choices=["rf", "gs"], default="gs",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: gs)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=32,
        help="Size of the hidden layer of Sender (default: 128)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=32,
        help="Size of the hidden layer of Receiver (default: 128)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=32,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 128)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=32,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 128)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    args = core.init(parser, params)

    if args.communication_type == "bee":
        args.grad_norm = 1.0 # to solve the explosive gradient problem

    # automatically get num of node features
    train_dataset = torch.load(args.train_data)
    sample_data = train_dataset[0][0]  # get the first graph sample
    args.num_node_features = sample_data.x.size(1)
    
    args.num_relations = 8 # because we discretized direction to 8 classes

    return args

def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    acc = (labels == receiver_output).float()
    return -acc, {"acc": acc}

def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float()
    return nll, {"acc": acc}


def get_game(opts):
    keep_dims = [0] # receiver does not get node‑type one‑hots apart from nest node
    if opts.communication_type == "bee":
        sender = BeeSender(
            opts.num_node_features,
            opts.sender_embedding,
            opts.sender_hidden,
            opts.num_relations
        )

        receiver = BeeReceiver(
            opts.num_node_features,
            opts.receiver_embedding,
            opts.receiver_hidden,
            opts.num_relations,
            keep_dims=keep_dims
        )
    else:
        sender = HumanSender(
            node_feat_dim = opts.num_node_features,
            embed_dim     = opts.sender_embedding,
            hidden_size   = opts.sender_hidden,
            num_rel       = opts.num_relations
    )

        receiver = HumanReceiver(
            node_feat_dim = opts.num_node_features,
            embed_dim     = opts.receiver_embedding,
            hidden_size   = opts.receiver_hidden,
            num_rel       = opts.num_relations,
            keep_dims=keep_dims
        )

    if opts.mode.lower() == "gs":
        if opts.communication_type == "bee":
            sender = MixedSymbolSenderWrapper(sender,
                                hidden_size=opts.sender_hidden,
                                vocab_size=opts.vocab_size,
                                temperature=opts.temperature,
                                straight_through=False)
            receiver = MixedSymbolReceiverWrapper(receiver,
                                vocab_size=opts.vocab_size,
                                agent_input_size=opts.receiver_hidden
        )
            game = core.SymbolGameGS(sender, receiver, loss_nll)
        else:
            sender = core.RnnSenderGS(
                sender,
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden,
                max_len=opts.max_len,
                temperature=opts.temperature,
                cell=opts.sender_cell
            )
            receiver = core.RnnReceiverGS(
                receiver,
                vocab_size=opts.vocab_size,
                embed_dim=opts.receiver_embedding,
                hidden_size=opts.receiver_hidden,
                cell=opts.receiver_cell
            )
            game = core.SenderReceiverRnnGS(sender, receiver, loss_nll)
            
        callbacks = []
    return game, callbacks


def perform_training(opts, train_loader, val_loader, game, callbacks, device, experiment_name):
    optimizer = core.build_optimizer(game.parameters())

    # for creating a timestamped folder
    timestamp_str = datetime.now().strftime("%Y-%m-%d") 
        
    callbacks = [
        ConsoleLogger(print_train_loss=True, as_json=True),
        InteractionSaver(
            train_epochs=list(range(1, opts.n_epochs + 1)),
            test_epochs=list(range(1, opts.n_epochs + 1)),
            checkpoint_dir=f"logs/interactions/{timestamp_str}/{experiment_name}",
            aggregated_interaction=False
        ),
        CsvLogger(log_dir=f"logs/csv/{timestamp_str}", filename=experiment_name),
        TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.5),
        EarlyStopperLoss(
        patience=10,
        min_delta=0.0,
        validation=True,
        verbose=True
    )
    ]

    if opts.print_validation_events:
        callbacks.append(PrintValidationEvents(opts.n_epochs))

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        device=device,
        callbacks=callbacks
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

def main(params, experiment_name=None):
    device = "cpu"
    opts = get_params(params)
    set_seed(opts.random_seed)

    if experiment_name is None:
        experiment_name = f"{opts.communication_type}_{opts.mode}_seed{opts.random_seed}"
    
    train_dataset = torch.load(opts.train_data)

    if opts.final_run:
        print("Mode: FULL TRAINING SET + TEST SET")
        val_dataset = torch.load(opts.validation_data)
        train_loader = DataLoader(
            train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True,
            persistent_workers=True, num_workers=4, generator=torch.Generator().manual_seed(opts.random_seed)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=opts.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True,
            persistent_workers=True, num_workers=4, generator=torch.Generator().manual_seed(opts.random_seed)
        )
    else:
        print("Mode: EXPERIMENT")
        n_samples = len(train_dataset)
        train_size = int(0.8 * n_samples)
        val_size = n_samples - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(opts.random_seed)
        )

        train_loader = DataLoader(
            train_subset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=True, num_workers=4,
            generator=torch.Generator().manual_seed(opts.random_seed)
        )
        val_loader = DataLoader(
            val_subset, batch_size=opts.batch_size, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=True, num_workers=4,
            generator=torch.Generator().manual_seed(opts.random_seed)
        )
    game, callbacks = get_game(opts)
    game.to(device)
    perform_training(opts, train_loader, val_loader, game, callbacks, device, experiment_name)
if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
