"""
    Implements PPO

    PPO: https://arxiv.org/abs/1707.06347
    Modified from policy Written by Patrick Coady (pat-coady.github.io) to implement
    latest version of PPO with pessimistic ratio clipping

    o Has an option to servo both the learning rate and the clip_param to keep KL 
      within  a specified range. This helps on some control tasks
      (i.e., Mujoco Humanid-v2)
 
    o Uses approximate KL 

    o Models distribution of actions as a Gaussian with variance not conditioned on state

 
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
    def __init__(self,   net, pd, adv_func=None,  kl_targ=0.001, mask_neg_adv=False,
                 epochs=20, init_func=rl_utils.default_init,
                 test_mode=False,  shuffle=True,  shuffle_by_chunks=False, servo_kl=False, beta=0.1, max_grad_norm=999, 
                 obs_key='observes', scale_vector_obs=True, scale_image_obs=False, verbose=False, rollout_limit=1):
        """
        Args:
            kl_targ:                target KL divergence between pi_old and pi_new
            epochs:                 number of epochs per update
            test_mode:              boolean, True removes all exploration noise
            shuffle:                boolean, shuffles data each epoch                   
            servo_kl:               boolean:  set to False to not servo beta to KL, which is original PPO implementation
            beta:                   clipping parameter for pessimistic loss ratio
 
        """
        print('Policy with vectorized sample')
        net.apply(init_func)

        self.net = net
        self.pd = pd

        if adv_func is None:
            self.adv_func = advantage_utils.Adv_default()
        else:
            self.adv_func = adv_func
 
        self.servo_kl = servo_kl
        self.test_mode = test_mode
        self.shuffle = shuffle
        self.shuffle_by_chunks = shuffle_by_chunks

        self.mask_neg_adv = mask_neg_adv 
        if self.net.recurrent_steps > 1 and not self.shuffle_by_chunks:
            print('Policy: recurrent steps > 1, disabling shuffle')
            self.shuffle = False
        self.kl_stat = None
        self.entropy_stat = None
        self.kl_targ = kl_targ
        self.epochs = epochs 
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.max_beta = 0.5
        self.min_beta = 0.01 
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.obs_key = obs_key
        self.grad_monitor = rl_utils.Grad_monitor('Policy', net)
        self.vector_scaler = rl_utils.Scaler(net.obs_dim)
        self.image_scaler = rl_utils.Image_scaler(net.obs_dim)
        self.scale_image_obs = scale_image_obs
        self.scale_vector_obs = scale_vector_obs

        self.verbose = verbose 
        self.rollout_limit = rollout_limit
        self.rollout_list = []

        self.calc_loss = self.calc_loss1

        if self.net.recurrent_steps > 1:
            self.use_padding = True
        else:
            self.use_padding = False

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.net.lr)

        print('\tTest Mode:         ',self.test_mode)
        print('\tClip Param:        ',self.beta)
        print('\tShuffle :          ',self.shuffle)
        print('\tShuffle by Chunks: ',self.shuffle_by_chunks)
        print('\tMax Grad Norm:     ',self.max_grad_norm)
        print('\tRecurrent Steps:   ',self.net.recurrent_steps)
        print('\tRollout Limit:     ',self.rollout_limit)
        print('\tAdvantage Func:    ',self.adv_func)
        print('\tAdvantage Norm:    ',self.adv_func.normalizer)
        print('\tPD:                ',self.pd)
        print('\tLoss Function:     ',self.calc_loss)

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
                    image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp   = \
                            rl_utils.shuffle_list_by_chunks([image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp], self.net.recurrent_steps)
                else:
                    image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp,  = \
                            sklearn.utils.shuffle(image_observes, vector_observes, actions, advantages, states, masks, flags, old_logp )

            actions_pt = self.pd.from_numpy(actions)

            self.optimizer.zero_grad()
            logits_pt, log_vars_pt,  _ = self.net.forward(image_observes,  vector_observes, states, masks, flags, unroll=True)
            logp_pt = self.pd.logp(actions_pt, logits_pt, log_vars_pt)
            loss = self.calc_loss(logp_pt, torch.from_numpy(old_logp).float(), torch.from_numpy(advantages).float(), self.beta, masks)
            loss.backward()
            if self.max_grad_norm is not None:
                ng = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            else:
                ng = None
            self.optimizer.step()
            self.grad_monitor.add(ng)

            kl = self.pd.kl(old_logp, logp_pt.detach().numpy(), log_vars_pt, masks)
            entropy = self.pd.entropy(logp_pt.detach().numpy(), log_vars_pt, masks)  
            if kl > 4.0 * self.kl_targ and self.servo_kl:
                print(' *** BROKE ***  ',e, kl)
                break 

        t1 = time()
            
        if self.servo_kl:
            self.adjust_beta(kl)

        for g in self.optimizer.param_groups:
            g['lr'] = self.net.lr * self.lr_multiplier
        self.kl_stat = kl
        self.entropy_stat = entropy
        self.grad_monitor.show()

        if self.verbose:
            print('POLICY ROLLOUT LIST: ',len(self.rollout_list))
            print('POLICY Update: ',t1-t0,observes.shape)
            print('kl = ',kl, ' beta = ',self.beta,' lr_mult = ',self.lr_multiplier)
            print('u_adv: ',u_adv)
            print('std_adv: ',std_adv)

        logger.log({'PolicyLoss': loss,
                    'Policy_SD' : np.mean(self.pd.sd(logits_pt, log_vars_pt)), 
                    'Policy_Entropy': entropy,
                    'Policy_KL': kl,
                    'Policy_Beta': self.beta,
                    'Policy_lr_mult': self.lr_multiplier})

    def adjust_beta(self,kl):
        if  kl < self.kl_targ / 2:
            self.beta = np.minimum(self.max_beta, 1.5 * self.beta)  # max clip beta
            #print('too low')
            if self.beta > (self.max_beta/2) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        elif kl > self.kl_targ * 2:
            #print('too high')
            self.beta = np.maximum(self.min_beta, self.beta / 1.5)  # min clip beta
            if self.beta <= (2*self.min_beta) and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5

    def calc_loss1(self,logp, old_logp, advantages, beta, masks):

        if self.mask_neg_adv:
            new_masks = masks * (advantages > 0)
        else:
            new_masks = masks

        if self.use_padding:
            logp, old_logp, advantages = rl_utils.unpad_list([logp, old_logp, advantages], new_masks)

        ratio = torch.exp(logp - old_logp)
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(ratio, 1.0 - beta, 1.0 + beta)
        
        loss = -torch.mean(torch.min(surr1,surr2)) 
        return loss

    def calc_loss2(self,logp, old_logp, advantages, beta, masks):

        if self.mask_neg_adv:
            new_masks = masks * (advantages > 0)
        else:
            new_masks = masks

        if self.use_padding:
            logp, old_logp, advantages = rl_utils.unpad_list([logp, old_logp, advantages], new_masks)

        advantages /= torch.sum(advantages)
        ratio = torch.exp(logp - old_logp)
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(ratio, 1.0 - beta, 1.0 + beta)

        loss = -torch.sum(torch.min(surr1,surr2))
        return loss

    def calc_loss3(self,logp, old_logp, advantages, beta, masks):

        if self.mask_neg_adv:
            new_masks = masks * (advantages > 0)
        else:
            new_masks = masks

        if self.use_padding:
            logp, old_logp, advantages = rl_utils.unpad_list([logp, old_logp, advantages], new_masks)

        ratio = torch.exp(logp - old_logp)
        surr1 = advantages * ratio

        loss = -torch.mean(surr1)
        return loss


