import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import egg.core as core
from archs.agents import HumanSender, HumanReceiver, BeeSender, BeeReceiver
from wrappers.wrapper import BeeGSWrapper
from helpers import collate_fn
from analysis.callbacks import DataLogger

import pdb

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication_type", choices=["bee", "human"], default="bee")

    # arguments concerning the input data and how they are processed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train_data", type=str, default="data/train_data.pt", help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default="data/test_data.pt", help="Path to the validation data"
    )
   
    # arguments concerning the training method
    parser.add_argument(
        "--mode", choices=["rf", "gs"], default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)"
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
    parser.add_argument(
        "--distance_bins",
        type=int,
        default=10,
        help="Number of bins to discretize continuous distances for the RGCNConv",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    args = core.init(parser, params)

    # automatically get num of node features
    train_dataset = torch.load(args.train_data)
    sample_data = train_dataset[0][0]  # get the first graph sample
    args.num_node_features = sample_data.x.size(1)
    
    args.num_relations = 8 # because we discretized direction to 8 classes

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        vocab_size = opts.num_relations
        max_len = 2

        sender = BeeSender(
            opts.num_node_features,
            opts.sender_embedding,
            opts.sender_hidden,
            opts.num_relations
        )

        receiver = BeeReceiver(
            opts.num_node_features,
            opts.receiver_embedding,
            opts.num_relations,
            keep_dims=keep_dims
        )
    else:
        sender = HumanSender(
            node_feat_dim = opts.num_node_features,
            embed_dim     = opts.sender_embedding,
            hidden_size   = opts.sender_hidden,
            num_rel       = opts.num_relations,
            num_distance_bins = opts.distance_bins
    )

        receiver = HumanReceiver(
            node_feat_dim = opts.num_node_features,
            embed_dim     = opts.receiver_embedding,
            hidden_size   = opts.receiver_hidden,
            num_rel       = opts.num_relations,
            num_distance_bins = opts.distance_bins,
            keep_dims=keep_dims
        )

    if opts.mode.lower() == "gs":
        if opts.communication_type == "bee":
            sender = BeeGSWrapper(sender,
                                  hidden_size=opts.sender_hidden,
                                  max_len=opts.max_len, 
                                  vocab_size=opts.vocab_size, 
                                  temperature=opts.temperature, 
                                  straight_through = False)
            
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
            game = core.SenderReceiverRnnGS(sender, receiver, loss_nll, length_cost=-0.1)
            
        callbacks = []
        
    elif opts.mode.lower() == "rf":
        if opts.communication_type == "bee":
            sender = BeeReinforceWrapper(sender)
            receiver = core.ReinforceWrapper(receiver)
            game = core.SymbolGameReinforce(
                sender,
                receiver,
                loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,
                receiver_entropy_coeff=0.01
            )
            callbacks = []
        else:
            sender = core.RnnSenderReinforce(
                sender,
                vocab_size,
                opts.sender_embedding,
                opts.sender_hidden,
                max_len,
                cell=opts.sender_cell
            )
            receiver = core.RnnReceiverReinforce(
                receiver,
                vocab_size,
                embed_dim=opts.receiver_embedding,
                hidden_size=opts.receiver_hidden,
                cell=opts.receiver_cell
            )
            game = core.SenderReceiverRnnReinforce(
                sender,
                receiver,
                loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,
                receiver_entropy_coeff=0.01
            )
            callbacks = []
    return game, callbacks


def perform_training(opts, train_loader, val_loader, game, callbacks):
    optimizer = core.build_optimizer(game.parameters())

    saver = core.InteractionSaver(
        train_epochs=[],
        test_epochs=[opts.n_epochs],
        checkpoint_dir="logs/msgs/experiment_bee_without_knowing_nest_totalnodes:10"
    )

    if opts.print_validation_events == True:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=val_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
                # DataLogger(save_path="logs/run3_bee_without_knowing_nest_totalnodes:10.json"),
                # saver
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=val_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True),
               ]
        )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

def main(params):
    opts = get_params(params)
    set_seed(opts.seed)

    train_dataset = torch.load(opts.train_data)
    val_dataset = torch.load(opts.validation_data)

    train_loader = DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(opts.seed)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False, collate_fn=collate_fn,
         generator=torch.Generator().manual_seed(opts.seed)
    )
    game, callbacks = get_game(opts)
    perform_training(opts, train_loader, val_loader, game, callbacks)
if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
