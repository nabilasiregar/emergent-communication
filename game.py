import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import egg.core as core
from egg.core import RnnSenderReinforce, RnnReceiverReinforce, SenderReceiverRnnReinforce
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data_loader import create_dataset
from agents import SenderAgent, ReceiverAgent

def make_egg_data_iter(loader):
    """
    Converts a PyG DataLoader into an infinite generator of EGG samples:
    (sender_input, receiver_input, _aux_input)
    """
    while True:
        for batch in loader:
            yield (None, None, {"graph_data": batch})

def select_node_and_reward(receiver_output, graph_data):
    predicted_node = receiver_output.argmax(dim=0).item()
    node_type = graph_data.x[predicted_node, :3] 
    is_food = (node_type[1].item() == 1.0)
    return 1.0 if is_food else 0.0

def loss_fn(sender_input, message, receiver_input, receiver_output, _aux_input):
    graph_data = _aux_input["graph_data"]
    reward = select_node_and_reward(receiver_output, graph_data)
    logs = {"acc_reward": torch.tensor([reward])}
    loss = torch.zeros(1, requires_grad=True)
    return loss, logs

def main(custom_args, egg_params):
    core.init(params=egg_params)
    opts = core.get_opts()

    max_len = 3 if custom_args.bee_like else custom_args.max_len
    n_epochs = custom_args.n_epochs

    # create dataset
    full_dataset = create_dataset(num_graphs=50, num_distractors=custom_args.game_size, connect_threshold=60.0)
    train_size = 40
    val_size = 10
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_iter = make_egg_data_iter(train_loader)
    val_iter = make_egg_data_iter(val_loader)

    # create agents
    feat_size = 5
    embedding_size = 16
    hidden_size = 32
    vocab_size = custom_args.vocab_size

    sender_core = SenderAgent(feat_size, embedding_size, hidden_size, vocab_size)
    sender = RnnSenderReinforce(
        agent=sender_core,
        vocab_size=vocab_size,
        embed_dim=hidden_size,
        hidden_size=hidden_size,
        max_len=max_len,
        cell='lstm'
    )

    receiver_core = ReceiverAgent(feat_size, embedding_size, hidden_size, vocab_size)
    receiver = RnnReceiverReinforce(
        agent=receiver_core,
        vocab_size=vocab_size,
        embed_dim=hidden_size,
        hidden_size=hidden_size,
        cell='lstm'
    )

    # training
    game = SenderReceiverRnnReinforce(
        sender=sender,
        receiver=receiver,
        loss=loss_fn,
        sender_entropy_coeff=1e-2,
        receiver_entropy_coeff=1e-2
    )
    optimizer = optim.Adam(game.parameters(), lr=custom_args.lr)
    
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_iter,
        validation_data=val_iter,
    )

    trainer.train(n_epochs=opts.n_epochs)
    core.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--model_generation", type=str, required=True)
    parser.add_argument("--game_size", type=int, required=True,
                        help="Number of distractors")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--max_len", type=int, required=True,
                        help="Max message length for human-like mode")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bee_like", action="store_true",
                        help="Use bee-like mode (max_len=3 tokens) instead of human-like mode")
    
    custom_args, leftover = parser.parse_known_args()
    print("Custom parsed arguments:", custom_args)
    print("Leftover EGG parameters:", leftover)

    main(custom_args, leftover)
