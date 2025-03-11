import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from archs.agents import BeeSender, Receiver
from graph.dataset import collate_fn
from wrappers import CustomSenderWrapper, CustomReceiverWrapper
import pdb


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/graph_dataset.pt")
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--direction_vocab_size", type=int, default=8)
    parser.add_argument("--num_node_features", type=int, default=3)
    parser.add_argument("--mode", choices=["rf", "gs"], default="rf")
    # parser.add_argument("--n_epochs", type=int, default=20)
    # parser.add_argument("--batch_size", type=int, default=32)

    return core.init(parser)


def custom_loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    predicted_node = receiver_output 
    correct_food_node = labels

    reward = (predicted_node == correct_food_node).float()

    loss = -reward

    accuracy = reward.mean().item()

    return loss, {'accuracy': accuracy}


def get_game(opts):
    sender = BeeSender(
        num_node_features=opts.num_node_features,
        embedding_size=opts.embedding_size,
        hidden_size=opts.hidden_size,
        direction_vocab_size=opts.direction_vocab_size,
        num_relations=2
    )

    receiver = Receiver(
        num_node_features=opts.num_node_features,
        embedding_size=opts.embedding_size,
        hidden_size=opts.hidden_size,
        vocab_size=opts.direction_vocab_size,
        num_relations=2
    )

    sender = CustomSenderWrapper(sender)
    receiver = CustomReceiverWrapper(receiver)

    game = core.SymbolGameReinforce(sender, receiver, custom_loss, sender_entropy_coeff=0.01, receiver_entropy_coeff=0.01)
    return game


def perform_training(opts, train_loader, val_loader, game):
    optimizer = torch.optim.Adam(game.parameters(), lr=1e-3)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=[core.ConsoleLogger(print_train_loss=True)]
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()


if __name__ == '__main__':
    opts = parse_arguments()

    dataset = torch.load(opts.data_path)

    train_loader = DataLoader(
        dataset['train_set'], batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset['val_set'], batch_size=opts.batch_size, shuffle=False, collate_fn=collate_fn
    )

    game = get_game(opts)

    perform_training(opts, train_loader, val_loader, game)

