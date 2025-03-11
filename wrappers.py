import torch
import torch.nn as nn
from torch.distributions import Categorical

class CustomSenderWrapper(nn.Module):
    def __init__(self, agent):
        super(CustomSenderWrapper, self).__init__()
        self.agent = agent

    def forward(self, sender_input, aux_input=None):
        direction_logits, distance_pred = self.agent(sender_input, aux_input)

        distr = Categorical(logits=direction_logits)
        entropy = distr.entropy()

        if self.training:
            discrete_sample = distr.sample()
        else:
            discrete_sample = direction_logits.argmax(dim=1)

        log_prob = distr.log_prob(discrete_sample)
        # combine the tokens to 1 tensor message
        message = torch.stack([discrete_sample.float(), distance_pred.squeeze(-1)], dim=1)

        return message, log_prob, entropy

class CustomReceiverWrapper(nn.Module):
    def __init__(self, agent):
        super(CustomReceiverWrapper, self).__init__()
        self.agent = agent

    def forward(self, message, x=None, _aux_input=None):
        logits = self.agent(message, x=None, _aux_input=_aux_input)

        distr = Categorical(logits=logits)
        entropy = distr.entropy()

        if self.training:
            node_sample = distr.sample()
        else:
            node_sample = logits.argmax(dim=1)

        log_prob = distr.log_prob(node_sample)

        return node_sample, log_prob, entropy

