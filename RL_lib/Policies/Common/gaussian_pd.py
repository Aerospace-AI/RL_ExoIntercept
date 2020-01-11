import numpy as np
import torch
import rl_utils

class Gaussian_pd(object):

    def __init__(self, action_dim, actions_per_dim):
        self.action_dim = action_dim
        self.actions_per_dim = actions_per_dim

    def logp(self, action, logits, log_vars):

        logp1 = -0.5 * torch.sum(log_vars)
        diff = action - logits 
        logp2 = -0.5 * torch.sum(torch.mul(diff, diff) / torch.exp(log_vars), 1)
        logp3 = -0.5 * np.log(2.0 * np.pi) * self.action_dim
        logp = logp1 + logp2 + logp3
        return logp

    def kl(self, logp, old_logp,  log_vars, masks):

        log_vars = log_vars.detach().numpy()

        logp, old_logp = rl_utils.unpad_list([logp, old_logp],masks)

        kl = 0.5 * np.mean((logp - old_logp)**2)

        return kl

    def entropy(self, logp, log_vars, masks):

        log_vars = log_vars.detach().numpy() 

        logp = rl_utils.unpad_list([logp],masks)

        entropy = 0.5 * (self.action_dim * (np.log(2 * np.pi) + 1) +
                              np.sum(log_vars))

        return entropy

    def sample(self, logits, log_vars, test_mode):

        if test_mode:
            action = logits 
        else:
            sd = np.exp(log_vars / 2.0)
            action = logits + np.random.normal(scale=sd)

        env_action = action

        return action, env_action

    def sd(self, logits, log_vars):
        log_vars = log_vars.detach().numpy()
        sd = np.exp(log_vars / 2.0)
        return sd 

    def from_numpy(self, actions):
        return torch.from_numpy(actions).float()


