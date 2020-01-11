import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils


class MLP1(nn.Module):
    def __init__(self, obs_dim, network_scale=10, base_lr=1e-2, recurrent_steps=1, activation=torch.tanh):
        super(MLP1, self).__init__()
        self.activation = activation
        self.obs_dim = obs_dim
        hid1_size = obs_dim * network_scale  # 10 chosen empirically on 'Hopper-v1'
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr =  base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, 1)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))


    def forward(self, x,  s, masks, flags, return_tensor=True, unroll=None):
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        if return_tensor:
            return torch.squeeze(x), self.initial_state_pt
        else:
            return torch.squeeze(x).detach().numpy(), self.initial_state



class GRU1(nn.Module):
    def __init__(self, obs_dim,   network_scale=10, base_lr=9e-3, recurrent_steps=1, activation=torch.tanh, cell=nn.GRUCell):
        super(GRU1, self).__init__()
        self.activation = activation
        self.obs_dim = obs_dim
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = 5 
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, 1)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))


    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)


        if return_tensor:
            return torch.squeeze(x), None 
        else:
            return torch.squeeze(x).detach().numpy(),   s.detach().numpy()


