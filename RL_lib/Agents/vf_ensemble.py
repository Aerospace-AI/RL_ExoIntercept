import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sklearn.utils

class VF_ensemble(object):

    """
        pred, vpred, state, error are model specific
        Looks like we actually don't use rpred and vpred in rollouots ??
        (only for env specific testing where we plot predicted versus actual trajectories)
    """

    def __init__(self, vf_list):
        print('VF Ensemble:')
        self.vf_list = vf_list
        self.m = len(self.vf_list)


    def predict(self, observe, state):
        next_state_list = []
        next_vpred_list = []
        state_list = np.split(state, self.m, axis=1)
        for i in range(self.m):
            vpred, state = self.vf_list[i].predict(observe, state_list[i] )   
            next_vpred_list.append(vpred)
            next_state_list.append(state)
    
        vpred = np.mean(next_vpred_list)
        return vpred, np.hstack(next_state_list)
        
    def fit(self, rollouts, logger):
       
        state_list = np.split(rollouts['vf_states'], self.m, axis=1)
        padded_state_list = np.split(rollouts['padded_vf_states'], self.m, axis=1)

        for i in range(self.m):
            m_rollouts = rollouts.copy() 
            m_rollouts['vf_states'] = state_list[i]
            m_rollouts['padded_vf_states'] = padded_state_list[i]
            if False:
                print(i,' 0: ', m_rollouts['vf_states'][0])
                print(i,' 1: ', m_rollouts['vf_states'][1])

            self.vf_list[i].fit(m_rollouts, logger) 

    def get_initial_state(self):
        state_list = []
        for i in range(self.m):
            state_list.append(self.vf_list[i].net.initial_state)
        return np.hstack(state_list)
    
    def update_scalers(self, rollouts):
        for i in range(self.m):
            self.vf_list[i].update_scalers(rollouts)

 
