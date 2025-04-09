import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import egg.core as core
from archs.agents import BeeSender, HumanSender, Receiver, BeeReceiver
from helpers import collate_fn
from analysis.callbacks import DataLogger

import pdb

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication_type", choices=["bee", "human"], default="bee")
    # parser.add_argument("--vocab_size", type=int, default=20,
    #                     help="Number of discrete symbols for the communication channel")
    # parser.add_argument("--max_len", type=int, default=2,
    #                     help="Max length of the message")
    # arguments concerning the input data and how they are processed
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
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
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

def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    acc = (receiver_output == labels).float()
    return -acc, {"acc": acc}

def get_game(opts):
    if opts.communication_type == "bee":
        sender = BeeSender(
            num_node_features=opts.num_node_features,
            embedding_size=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            num_relations=opts.num_relations
        )
        vocab_size = opts.num_relations + 1
        max_len = 2

        receiver = BeeReceiver(
            num_node_features=opts.num_node_features,
            embedding_size=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            vocab_size=vocab_size,
            num_relations=opts.num_relations
        )
    else:
        sender = HumanSender(
            num_node_features=opts.num_node_features,
            embedding_size=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            num_relations=opts.num_relations,
            vocab_size=opts.vocab_size,
            max_len=opts.max_len
        )
        vocab_size = opts.vocab_size
        max_len = opts.max_len

        receiver = Receiver(
        num_node_features=opts.num_node_features,
        embedding_size=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        vocab_size=vocab_size,
        num_relations=opts.num_relations
    )

    if opts.mode.lower() == "gs":
        sender = core.RnnSenderGS(
            sender,
            vocab_size=vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=max_len,
            temperature=opts.temperature,
            cell=opts.sender_cell
        )
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell
        )
        game = core.SenderReceiverRnnGS(
            sender,
            receiver,
            loss
        )
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    elif opts.mode.lower() == "rf":
        if opts.communication_type == "bee":
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

    if opts.print_validation_events == True:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=val_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs)
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=val_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
        )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

def main(params):
    opts = get_params(params)

    train_dataset = torch.load(opts.train_data)
    val_dataset = torch.load(opts.validation_data)

    train_loader = DataLoader(
        train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False, collate_fn=collate_fn
    )

    game, callbacks = get_game(opts)
    perform_training(opts, train_loader, val_loader, game, callbacks)
if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
