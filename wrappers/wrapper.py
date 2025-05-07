import torch, torch.nn as nn, torch.nn.functional as F
from egg.core.gs_wrappers import gumbel_softmax_sample
import pdb

class BeeGSWrapper(nn.Module):
    """
    Wraps BeeSender which returns differentiable message tensor
       shape = (B, n_relations + 1)
    """
    def __init__(self, sender_core, temperature=1.0, straight_through=False):
        super().__init__()
        self.core = sender_core
        self.temp = temperature
        self.st  = straight_through

    def forward(self, sinp, aux):
        out = self.core(sinp, aux)
        # discrete token
        y = gumbel_softmax_sample(out["direction_logits"],
                                  self.temp, self.training, self.st)
        # continuous token
        mu, logvar = out["mu"], out["logvar"]
        eps  = torch.randn_like(mu)
        distance = mu + eps * torch.exp(0.5 * logvar)
        return torch.cat([y, distance], -1)

