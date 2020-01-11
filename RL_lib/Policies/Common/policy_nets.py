import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils

class MLP1(nn.Module):
    def __init__(self, obs_dim, act_dim, input_network_scale=10, output_network_scale=10, 
                    activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLP1, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * input_network_scale
        hid3_size = act_dim * output_network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    
    def forward(self, image_obs, vector_obs, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(vector_obs).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x,  log_vars, self.initial_state_pt 
        else:
            return x.detach().numpy(), log_vars.detach().numpy(),  self.initial_state


 
class GRU1(nn.Module):
    def __init__(self, obs_dim,  act_dim, input_network_scale=10, output_network_scale=10, 
                 activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU1, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * input_network_scale
        hid3_size = act_dim * output_network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, image_obs,  vector_obs, states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(vector_obs).float()
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

        log_vars = torch.sum(self.log_vars, 0) - 1.0

        if return_tensor:
            return x,  log_vars, None 
        else:
            return x.detach().numpy(),   log_vars.detach().numpy(), s.detach().numpy()


   
class GRU_CNN1(nn.Module):
    def __init__(self, obs_dim,  act_dim, cnn_layer, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU_CNN1, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid3_size = act_dim * network_scale
        self.cnn1 = cnn_layer
        hid1_size = self.cnn1.size() + obs_dim
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)

        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, image_obs,  vector_obs, states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        xi = torch.from_numpy(image_obs).float()
        xs = torch.from_numpy(vector_obs).float()
        s = torch.from_numpy(states).float()

        xi = self.activation(self.cnn1(xi))
        xi = self.cnn1.flatten(xi)
        x = torch.cat((xi,xs),dim=1)
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

        log_vars = torch.sum(self.log_vars, 0) - 1.0
 
        if return_tensor:
            return x,  log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()
 
    
class GRU_CNN2(nn.Module):
    def __init__(self, obs_dim,  act_dim, cnn_layer, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU_CNN2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid3_size = act_dim * network_scale
        self.cnn1 = cnn_layer
        hid1a_size = obs_dim * network_scale
        self.fc1 =  nn.Linear(obs_dim, hid1a_size)
        hid1_size = self.cnn1.size() + hid1a_size
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)

        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, image_obs,  vector_obs, states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        xi = torch.from_numpy(image_obs).float()
        xs = torch.from_numpy(vector_obs).float()
        s = torch.from_numpy(states).float()
        xs = self.activation(self.fc1(xs))
        xi = self.activation(self.cnn1(xi))
        xi = self.cnn1.flatten(xi)
        x = torch.cat((xi,xs),dim=1)
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

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class GRU_CNN3(nn.Module):
    def __init__(self,  act_dim, cnn_layer, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU_CNN3, self).__init__()
        self.obs_dim = 2  # not used, just so scaler does not barf
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid3_size = act_dim * network_scale
        self.cnn1 = cnn_layer
        hid1_size = self.cnn1.size() 
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)

        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, image_obs,  vector_obs, states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(image_obs).float()
        s = torch.from_numpy(states).float()
        x = self.activation(self.cnn1(x))
        x = self.cnn1.flatten(x)
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

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

