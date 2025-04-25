import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class BeeReinforceWrapper(nn.Module):
    def __init__(self, sender):
        super().__init__()
        self.sender = sender

    def forward(self, *args, **kwargs):
        out = self.sender(*args, **kwargs)
        logits = out['discrete_logits']
        mu, logvar = out['mu'], out['logvar']

        # --- Discrete ---
        distr_d = Categorical(logits=logits)
        if self.training:
            token_d = distr_d.sample()      
        else:
            token_d = logits.argmax(dim=1)  
        logp_d   = distr_d.log_prob(token_d)  
        ent_d    = distr_d.entropy() 

        # --- Continuous ---
        # clamp logvar to avoid numerical issues
        logvar = torch.clamp(logvar, -5.0, 5.0)
        std     = torch.exp(0.5 * logvar)
        distr_c = Normal(mu, std)
        if self.training:
            token_c = distr_c.rsample().squeeze(-1) 
        else:
            token_c = mu.squeeze(-1)            
        logp_c   = distr_c.log_prob(token_c).sum(dim=-1) 
        ent_c    = distr_c.entropy().sum(dim=-1)   

        # discrete token as float in channel 0, continuous in channel 1
        message = torch.stack([token_d.float(), token_c], dim=1)
        log_prob = logp_d + logp_c 
        entropy  = ent_d  + ent_c 

        return message, log_prob, entropy

