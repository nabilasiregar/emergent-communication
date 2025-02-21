import argparse
import torch.optim as optim
import egg.core as core
from agents import GraphSender, GraphReceiver
from torch.utils.data import DataLoader
from data_loader import get_dataloader
from environment import Environment
import pdb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=16, help="Size of the embedding layer")
    parser.add_argument("--hidden_size", type=int, default=32, help="Size of the hidden layers")
    parser.add_argument("--vocab_size", type=int, default=8, help="Size of the vocabulary")
    parser.add_argument("--game_size", type=int, default=1, help="Size of the game (number of agents)")
    parser.add_argument("--feat_size", type=int, default=12, help="Size of the feature vectors")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes for classification")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature parameter for Gumbel-Softmax")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['rf', 'gs'],
        default="rf",
        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf)"
    )
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    
    args = parser.parse_args()
    return args

def play(opts):
    env = Environment(num_distractors=2)
    graph_data = get_dataloader(env.graph)
    dataloader = DataLoader(graph_data, batch_size=opts.batch_size, shuffle=True)

    sender = GraphSender(opts.game_size, opts.feat_size, opts.embedding_size, opts.hidden_size, opts.vocab_size, opts.temp)
    receiver = GraphReceiver(opts.game_size, opts.feat_size, opts.embedding_size, opts.vocab_size, reinforce=(opts.mode == "rf"))
    
    def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        return torch.nn.functional.nll_loss(receiver_output, labels)

    if opts.mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(sender, receiver, loss)
    elif opts.mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=opts.temp)
        receiver = core.GumbelSoftmaxWrapper(receiver, temperature=opts.temp)
        game = core.SymbolGameGS(sender, receiver, loss)
  
    trainer = core.Trainer(
        game=game,
        optimizer=optim.Adam(list(sender.parameters()) + list(receiver.parameters()), lr=0.001),
        train_data=dataloader,
        validation_data=None,
        callbacks=[core.ConsoleLogger()]
    )
    trainer.train()

if __name__ == "__main__":
    opts = parse_arguments()
    play(opts)
