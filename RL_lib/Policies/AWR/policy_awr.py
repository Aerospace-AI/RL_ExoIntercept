
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
    def __init__(self,   net, pd,  adv_func=None, kl_limit=0.1, 
                 epochs=20, init_func=rl_utils.default_init, 
                 test_mode=False, shuffle=True,   shuffle_by_chunks=False, max_grad_norm=999, 
                 obs_key='observes', scale_vector_obs=True, scale_image_obs=False, verbose=False, rollout_limit=1):

        print('AWR Policy: ')
        net.apply(init_func)

        self.net = net
        self.pd = pd

        if adv_func is None:
            self.adv_func = advantage_utils.Adv_relu()
        else:
            self.adv_func = adv_func

        self.test_mode = test_mode
        self.shuffle = shuffle
        self.shuffle_by_chunks = shuffle_by_chunks

        if self.net.recurrent_steps > 1 and not self.shuffle_by_chunks:
            print('Policy: recurrent steps > 1, disabling shuffle')
            self.shuffle = False
        self.kl_limit = kl_limit
        self.epochs = epochs 
        self.max_grad_norm = max_grad_norm
        self.obs_key = obs_key
        self.grad_monitor = rl_utils.Grad_monitor('Policy', net)
        self.vector_scaler = rl_utils.Scaler(net.obs_dim)
        self.image_scaler = rl_utils.Image_scaler(net.obs_dim)
        self.scale_image_obs = scale_image_obs
        self.scale_vector_obs = scale_vector_obs

        self.verbose = verbose 
        self.rollout_limit = rollout_limit
        self.rollout_list = []

        if self.net.recurrent_steps > 1:
            self.use_padding = True
        else:
            self.use_padding = False

        self.calc_loss = self.calc_loss1
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.net.lr)

        print('\tTest Mode:         ',self.test_mode)
        print('\tShuffle :          ',self.shuffle)
        print('\tShuffle by Chunks: ',self.shuffle_by_chunks)
        print('\tMax Grad Norm:     ',self.max_grad_norm)
        print('\tRecurrent Steps:   ',self.net.recurrent_steps)
        print('\tRollout Limit:     ',self.rollout_limit)
        print('\tAdvantage Func:    ',self.adv_func)
        print('\tAdvantage Norm:    ',self.adv_func.normalizer)
        print('\tPD:                ',self.pd)

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

        if self.scale_image_obs:
            image_obs = self.image_scaler.apply(image_obs)
        if self.scale_vector_obs:
            vector_obs = self.vector_scaler.apply(vector_obs)
        logits, log_vars, state = self.net.forward(image_obs, vector_obs, state, np.ones(1), np.zeros(1), return_tensor=False)
        action, env_action = self.pd.sample(logits, log_vars, self.test_mode)
        return action, env_action, state 

    def update_scalers(self, rollouts):
        self.image_scaler.update(rollouts['image_observes'])
        self.vector_scaler.update(rollouts['vector_observes'])

        
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
 
        vtarg_unp   = rollouts['disc_sum_rew']
        vpred_unp   = rollouts['vpreds']

        actions_pt = self.pd.from_numpy(actions)

        with torch.no_grad():
            old_logits_pt, log_vars_pt,  _ = self.net.forward(image_observes,  vector_observes, states, masks, flags)

        old_logp_pt = self.pd.logp(actions_pt, old_logits_pt, log_vars_pt)   
        old_logp = old_logp_pt.detach().numpy() 
        loss, kl, entropy = 0, 0, 0

        advantages_unp = vtarg_unp - vpred_unp
        advantages = vtarg - vpred

        print('ADV1: ',  np.mean(advantages), np.std(advantages), np.max(advantages), np.min(advantages))

        advantages = self.adv_func.calc_adv(advantages_unp, advantages)

        print('ADV2: ',  np.mean(advantages), np.std(advantages), np.max(advantages), np.min(advantages))

        t0 = time()
        for e in range(self.epochs):

            if self.shuffle:
                if self.shuffle_by_chunks:
                    image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp  = \
                            rl_utils.shuffle_list_by_chunks([image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp], self.net.recurrent_steps)
                else:
                    image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp  = \
                            sklearn.utils.shuffle(image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp)

            actions_pt = self.pd.from_numpy(actions)

            self.optimizer.zero_grad()
            logits_pt, log_vars_pt,  _ = self.net.forward(image_observes,  vector_observes, states, masks, flags, unroll=True)
            logp_pt = self.pd.logp(actions_pt, logits_pt, log_vars_pt)
            loss = self.calc_loss(logp_pt,  torch.from_numpy(advantages).float(), masks)
            loss.backward()
            if self.max_grad_norm is not None:
                ng = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            else:
                ng = None
            self.optimizer.step()
            self.grad_monitor.add(ng)

            kl = self.pd.kl(old_logp, logp_pt.detach().numpy(), log_vars_pt, masks)
            entropy = self.pd.entropy(logp_pt.detach().numpy(), log_vars_pt, masks)  
            if kl > self.kl_limit: 
                print(' *** BROKE ***  ',e, kl)
                break 

        t1 = time()
            

        #self.kl_stat = kl
        #self.entropy_stat = entropy
        self.grad_monitor.show()

        if self.verbose:
            print('POLICY ROLLOUT LIST: ',len(self.rollout_list))
            print('POLICY Update: ',t1-t0,observes.shape)
            print('u_adv: ',u_adv)
            print('std_adv: ',std_adv)

        logger.log({'PolicyLoss': loss,
                    'Policy_SD' : np.mean(self.pd.sd(logits_pt, log_vars_pt)), 
                    'Policy_Entropy': entropy,
                    'Policy_KL': kl})


    def calc_loss1(self,logp, advantages,  masks):
        new_masks = masks * (advantages > 0)
        if self.use_padding:
            logp,  advantages = rl_utils.unpad_list([logp,  advantages], new_masks)
        loss = -torch.mean(advantages * logp) 
        return loss

    def calc_loss2(self,logp, advantages,  masks):
        new_masks = masks * (advantages > 0)
        if self.use_padding:
            logp,  advantages = rl_utils.unpad_list([logp,  advantages], new_masks)
        advantages /= torch.sum(advantages)
        loss = -torch.sum(advantages * logp)
        return loss

