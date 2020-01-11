
"""

    Implements AWR PPO variants
 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rl_utils
import advantage_utils
from time import time
import sklearn.utils
 
class Policy(object):
    """ NN-based policy approximation """
    def __init__(self,   net,  pd, adv_func=None,  std=0.10, weight_by_adv=False, fix_loss=True,  
                 epochs=20, init_func=rl_utils.default_init, 
                 test_mode=False, shuffle=True,   shuffle_by_chunks=False, max_grad_norm=999, 
                 obs_key='observes', scale_vector_obs=True, scale_image_obs=False, scale_actions=True, verbose=False, rollout_limit=1):

        net.apply(init_func)

        self.net = net
        self.pd = pd
        if adv_func is None:
            self.adv_func = advantage_utils.Adv_relu()
        else:
            self.adv_func = adv_func
        self.std = std
        self.weight_by_adv = weight_by_adv
        self.fix_loss = fix_loss
        self.test_mode = test_mode
        self.shuffle = shuffle
        self.shuffle_by_chunks = shuffle_by_chunks

        if self.net.recurrent_steps > 1 and not self.shuffle_by_chunks:
            print('Policy: recurrent steps > 1, disabling shuffle')
            self.shuffle = False
        self.epochs = epochs 
        self.max_grad_norm = max_grad_norm
        self.obs_key = obs_key
        self.grad_monitor = rl_utils.Grad_monitor('Policy', net)
        self.vector_scaler = rl_utils.Scaler(net.obs_dim)
        self.action_scaler = rl_utils.Scaler(net.act_dim)
        self.image_scaler = rl_utils.Image_scaler(net.obs_dim)
        self.scale_image_obs = scale_image_obs
        self.scale_vector_obs = scale_vector_obs
        self.scale_actions = scale_actions

        self.verbose = verbose 
        self.rollout_limit = rollout_limit
        self.rollout_list = []

        if self.net.recurrent_steps > 1:
            self.use_padding = True
        else:
            self.use_padding = False

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.net.lr)

        print('\tTest Mode:         ',self.test_mode)
        print('\tShuffle :          ',self.shuffle)
        print('\tShuffle by Chunks: ',self.shuffle_by_chunks)
        print('\tMax Grad Norm:     ',self.max_grad_norm)
        print('\tRecurrent Steps:   ',self.net.recurrent_steps)
        print('\tRollout Limit:     ',self.rollout_limit)
        print('\tRollout Limit:     ',self.rollout_limit)
        print('\tAdvantage Func:    ',self.adv_func)
        print('\tAdvantage Norm:    ',self.adv_func.normalizer)

    def save_params(self,fname):
        fname = 'policy_' + fname + '.pt'
        param_dict = {}
        param_dict['image_scaler_u']   = self.image_scaler.means
        param_dict['image_scaler_var']  = self.image_scaler.vars
        param_dict['vector_scaler_u']   = self.vector_scaler.means
        param_dict['vector_scaler_var']  = self.vector_scaler.vars
        param_dict['net_state'] = self.net.state_dict()
        torch.save(param_dict, fname)

    def load_params(self,fname):
        fname = 'policy_' + fname + '.pt'
        param_dict = torch.load(fname)
        self.image_scaler.means = param_dict['image_scaler_u']
        self.image_scaler.vars = param_dict['image_scaler_var']
        self.vector_scaler.means = param_dict['vector_scaler_u']
        self.vector_scaler.vars = param_dict['vector_scaler_var']
        self.net.load_state_dict(param_dict['net_state'])



    def sample(self, image_obs, vector_obs, state):


        image_obs = self.image_scaler.apply(image_obs)
        vector_obs = self.vector_scaler.apply(vector_obs)
        logits, log_vars, state = self.net.forward(image_obs, vector_obs, state, np.ones(1), np.zeros(1), return_tensor=False)

        if self.scale_actions:
            action = self.action_scaler.reverse(logits) + np.random.normal(scale=self.std, size=(logits.shape[1]))
        else:
            action = logits + np.random.normal(scale=self.std, size=(logits.shape[1]))
        env_action = action 
        return action, env_action, state 

    def update_scalers(self, rollouts):
        self.image_scaler.update(rollouts['image_observes'])
        self.vector_scaler.update(rollouts['vector_observes'])
        self.action_scaler.update(rollouts['actions'])
        
    def update(self, rollouts, logger):
        if len(self.rollout_list) == self.rollout_limit:
            del self.rollout_list[0]
        self.rollout_list.append(rollouts)
        keys = self.rollout_list[0].keys()
        comb_rollouts = {}
        for k in keys:
            comb_rollouts[k] = np.concatenate([r[k] for r in self.rollout_list])
        self.update1(comb_rollouts, logger)
 
    def update1(self, rollouts, logger):
      
        if self.use_padding:
            key = 'padded_'
        else:
            key = '' 
        image_observes    = rollouts[key + 'image_observes']
        vector_observes    = rollouts[key + 'vector_observes']

        actions     = rollouts[key + 'actions']
        states      = rollouts[key + 'policy_states']
        vtarg       = rollouts[key + 'disc_sum_rew']
        vpred       = rollouts[key + 'vpreds']
        masks       = rollouts[key + 'masks']
        flags       = rollouts[key + 'flags']

        if self.scale_vector_obs:
            vector_observes = self.vector_scaler.apply(vector_observes)
        if self.scale_image_obs:
            image_observes = self.image_scaler.apply(image_observes)
        if self.scale_actions:
            actions = self.action_scaler.apply(actions) 
        vtarg_unp   = rollouts['disc_sum_rew']
        vpred_unp   = rollouts['vpreds']

        actions_pt = self.pd.from_numpy(actions)

        loss =   0

        advantages_unp = vtarg_unp - vpred_unp
        advantages = vtarg - vpred

        print('ADV1: ',  np.mean(advantages), np.std(advantages), np.max(advantages), np.min(advantages))
        advantages = self.adv_func.calc_adv(advantages_unp, advantages)
        print('ADV2: ',  np.mean(advantages), np.std(advantages), np.max(advantages), np.min(advantages))


        t0 = time()
        for e in range(self.epochs):

            if self.shuffle:
                    image_observes, vector_observes, actions, advantages, states, masks, flags,  = \
                            sklearn.utils.shuffle(image_observes, vector_observes, actions, advantages, states, masks, flags  )

            actions_pt = self.pd.from_numpy(actions)

            self.optimizer.zero_grad()
            logits_pt, _,  _ = self.net.forward(image_observes,  vector_observes, states, masks, flags, unroll=True)
            loss = self.calc_loss(torch.from_numpy(actions).float() , logits_pt,  torch.from_numpy(advantages).float(), masks)
            loss.backward()
            if self.max_grad_norm is not None:
                ng = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            else:
                ng = None
            self.optimizer.step()
            self.grad_monitor.add(ng)

        t1 = time()
            

        self.grad_monitor.show()

        if self.verbose:
            print('POLICY ROLLOUT LIST: ',len(self.rollout_list))
            print('POLICY Update: ',t1-t0,observes.shape)
            print('u_adv: ',u_adv)
            print('std_adv: ',std_adv)

        logger.log({'PolicyLoss': loss,
                    'Policy_SD' : self.std}) 
                    #'Policy_Entropy': y,
                    #'Policy_KL': kl})


    def calc_loss(self, actions, logits, advantages, masks):
        if self.fix_loss:
            new_masks = masks * (advantages > 0)
        else:
            new_masks = masks
        if self.use_padding:
            actions, logits, advantages = rl_utils.unpad_list([actions, logits, advantages],new_masks)
        #print(actions.shape, logits.shape, advantages.shape, actions.size(1))
        error = actions - logits
        if self.weight_by_adv:
            advantages /= torch.sum(advantages)
            mse = torch.sum(torch.mul(error,error), dim=1)
            loss = torch.sum( 0.5 * mse * advantages) / actions.size(1)
        else:
            mse = torch.mean(torch.mul(error,error), dim=1)
            loss = 0.5 * torch.mean(mse*advantages)
            #print('LOSS: ', loss)
        return loss

 
