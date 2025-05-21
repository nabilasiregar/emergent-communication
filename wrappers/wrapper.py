"""Custom two‑token (discrete a, continuous b) wrappers for an EGG
SymbolGame.  They mimic the API and naming style of GumbelSoftmaxWrapper 
and SymbolReceiverWrapper.
"""
from typing import Optional

import torch
import torch.nn as nn

from egg.core.gs_wrappers import gumbel_softmax_sample, RelaxedEmbedding
import pdb

class MixedSymbolSenderWrapper(nn.Module):
    """Wraps a continuous sender core so it emits one symbol that
    contains a discrete id token_a (Gumbel‑Softmax) and a continuous
    scalar token_b (Gaussian re‑parametrisation).

    The resulting message has dimensionality vocab_size + 1 where first
    vocab_size positions form a one‑hot for token_a, the last dimension is
    the raw scalar token_b.
    """

    def __init__(
        self,
        agent: nn.Module,
        hidden_size: int,
        vocab_size: int = 8,
        temperature: float = 1.0,
        trainable_temperature: bool = False,
        straight_through: bool = False,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.vocab_size = vocab_size
        self.straight_through = straight_through
        if trainable_temperature:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.temperature = temperature

        self.logits_a = nn.Linear(hidden_size, vocab_size)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)

    def forward(self, *args, **kwargs):
        h = self.agent(*args, **kwargs)
        assert h.dim() == 2, "Sender core must return [batch, hidden]"

        onehot_a = gumbel_softmax_sample(
            self.logits_a(h), self.temperature, self.training, self.straight_through
        )

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        eps = torch.randn_like(mu)
        token_b = mu + eps * torch.exp(0.5 * logvar)

        message = torch.cat([onehot_a, token_b], dim=-1)
        return message


class MixedSymbolReceiverWrapper(nn.Module):
    """Inverse of MixedSymbolSenderWrapper

    Splits the incoming message into the one‑hot part (embedded by
    RelaxedEmbedding) and the continuous scalar (passed through a
    tiny linear layer).  The two vectors are summed before feeding the
    wrapped agent.
    """

    def __init__(
        self,
        agent: nn.Module,
        vocab_size: int,
        agent_input_size: int,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.embedding_a = RelaxedEmbedding(vocab_size, agent_input_size)
        self.embedding_b = nn.Linear(1, agent_input_size)

    def forward(self, message: torch.Tensor, receiver_input=None, aux_input=None):
        vocab = self.embedding_a.num_embeddings
        onehot_a = message[:, :vocab].unsqueeze(1) 
        scalar_b = message[:, vocab : vocab + 1]

        emb_a = self.embedding_a(onehot_a)
        emb_b = self.embedding_b(scalar_b).unsqueeze(1)

        merged = torch.relu(emb_a + emb_b).squeeze(1)
      
        return self.agent(merged, receiver_input, aux_input)

