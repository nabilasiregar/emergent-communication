"""
Evaluate a trained bee or human model on the
zero-shot compositional test set.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from utils.helpers import collate_fn, set_seed
import game

def load_model(ckpt_path: str, device="cpu") -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    opts = argparse.Namespace(**ckpt["opts"])
    mdl, _ = game.get_game(opts)
    mdl.load_state_dict(ckpt["game_state"])
    return mdl.to(device).eval()

@torch.no_grad()
def accuracy(model, dataset_path, batch_size, device):
    data   = torch.load(dataset_path)
    loader = DataLoader(data,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn)

    correct = total = 0
    for sender_input, labels, receiver_input, aux_input in loader:
        sender_input = sender_input.to(device)
        labels    = labels.to(device)

        try:
            # bee
            result = model(sender_input, labels, receiver_input, aux_input)
        except TypeError:
            # human
            result = model(sender_input, labels)

        if isinstance(result, tuple):
            _, second, *rest = result
            if hasattr(second, "receiver_output"):
                #bee
                logits = second.receiver_output
            else:
                #human
                logits = second
        else:
            logits = result

        if logits.ndim == 3:
            logits = logits[:, -1, :]

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return 100.0 * correct / total



def main():
    ap = argparse.ArgumentParser(description="Zero-shot TZSG evaluation")
    ap.add_argument("--ckpt", required=True, help="*.pt checkpoint")
    ap.add_argument("--testset", default="data/compos_test.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    set_seed(0)
    mdl = load_model(args.ckpt, args.device)
    acc = accuracy(mdl, args.testset, args.batch_size, args.device)
    print(f"{args.ckpt}: zero-shot accuracy = {acc:.2f} %")

if __name__ == "__main__":
    main()