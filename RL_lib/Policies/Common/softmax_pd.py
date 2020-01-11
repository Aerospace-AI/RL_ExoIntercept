import numpy as np
import torch
import rl_utils 


class Softmax_pd(object):
    def __init__(self,action_dim, actions_per_dim, min_action=-1, max_action=1.):
        self.action_dim = action_dim
        self.actions_per_dim = actions_per_dim 

        self.action_converter = rl_utils.Action_converter(1,actions_per_dim,min_action=min_action, max_action=max_action)



    def kl(self, old_logp, logp, log_vars, masks):

        logp, old_logp = rl_utils.unpad_list([logp, old_logp],masks)

        kl = 0.5 * np.mean((logp - old_logp)**2)

        return kl

    def entropy(self, logp, log_vars, masks):

        logp, = rl_utils.unpad_list([logp],masks)

        p = np.exp(logp)

        entropy = np.sum( - p * np.log2(p)) / logp.shape[0]
        return entropy

    def logp(self, actions, logits, log_vars):
        logps = []
        logits = torch.split(logits,self.actions_per_dim,dim=1)
        for i in range(actions.shape[1]):
            logp1  = torch.nn.functional.log_softmax(logits[i],dim=1)
            m = logits[i].shape[0]
            a = actions[:,i]
            logp = logp1[torch.arange(m), actions[:,i]]
            logps.append(logp)
        logps = torch.stack(logps, dim=1)
        logp = torch.sum(logps,dim=1) - np.log(self.action_dim)
        return logp

    def sample(self, logits, log_vars, test_mode):
        U = np.random.uniform(low=0.0, high=1.0, size=(self.action_dim,self.actions_per_dim))
        logits = logits.reshape(self.action_dim, self.actions_per_dim)
        if test_mode:
            action = np.argmax(logits,axis=1)
        else:
            action = np.argmax(logits - np.log(-np.log(U)) ,axis=1)
        env_action = np.squeeze(self.action_converter.idx2action(action))
        return np.expand_dims(action,axis=0), env_action

    def sd(self, logits1, log_vars):
        logits = logits1.detach().numpy()
        U = np.random.uniform(low=0.0, high=1.0, size=(logits.shape[0], self.action_dim, self.actions_per_dim))
        logits = logits.reshape(-1, self.action_dim, self.actions_per_dim)
        action = np.argmax(logits - np.log(-np.log(U)) ,axis=-1)
        env_action = np.squeeze(self.action_converter.idx2action(action))
        s = np.mean(np.std(env_action, axis=-1)) 
        return s
    
    def from_numpy(self, actions):
        return torch.from_numpy(actions)

    #def cast(self, actions):
    #    return actions.astype(int)
 
