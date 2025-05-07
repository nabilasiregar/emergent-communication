import torch
import torch.nn as nn
from egg.core.gs_wrappers import gumbel_softmax_sample

class BeeGSWrapper(nn.Module):
    """
    A wrapper that lets each token emerge as either discrete-like
    or continuous-like.
    """
    def __init__(
        self,
        sender: nn.Module,
        hidden_size: int,
        max_len: int = 2,
        vocab_size: int = 8,
        temperature: float = 1.0,
        straight_through: bool = False,
    ):
        super().__init__()
        self.core = sender
        self.max_len = max_len
        self.vocab = vocab_size
        self.tau = temperature
        self.st = straight_through

        self.switch_head = nn.Linear(hidden_size,  self.max_len * 2)
        self.disc_head   = nn.Linear(hidden_size,  self.max_len * self.vocab)
        self.mu_head     = nn.Linear(hidden_size,  self.max_len)
        self.logvar_head = nn.Linear(hidden_size,  self.max_len)

    def forward(self, sinp, aux):
        h   = self.core(sinp, aux)
        B   = h.size(0)

        switch_logits = self.switch_head(h).view(B, self.max_len, 2)
        disc_logits   = self.disc_head(h).view(B, self.max_len, self.vocab)
        mu            = self.mu_head(h).view(B, self.max_len)
        logvar        = self.logvar_head(h).view(B, self.max_len)

        # choose token type: discrete or continuous
        mode = gumbel_softmax_sample(
            switch_logits.view(-1, 2), self.tau, self.training, self.st
        ).view(B, self.max_len, 2)
        disc_mask = mode[..., 0:1]
        cont_mask = mode[..., 1:2]

        # disc
        disc_onehot = gumbel_softmax_sample(
            disc_logits.view(-1, self.vocab), self.tau, self.training, self.st
        ).view(B, self.max_len, self.vocab)
        disc_scalar = disc_onehot.argmax(dim=-1, keepdim=True).float() / (self.vocab - 1)

        # continuous
        eps          = torch.randn_like(mu)
        cont_scalar  = (mu + eps * torch.exp(0.5 * logvar)).unsqueeze(-1)

        token = disc_mask * disc_scalar + cont_mask * cont_scalar

        return token.squeeze(-1)
