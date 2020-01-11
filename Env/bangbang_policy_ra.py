

import numpy as np
import env_utils as envu
import attitude_utils as attu
from time import time

class BangBang_policy(object):
    def __init__(self,act_dim, max_acc_range=(0.,2*9.81), tf=80 ):
        self.act_dim = act_dim
        self.max_acc_range = max_acc_range
        self.tf = tf 

    def reset(self):
        self.duration = np.random.uniform(low=self.tf/8, high=self.tf/2, size=2)  # full bang-bang +/-
        self.t_start = np.random.uniform(low=0, high=self.tf-self.duration, size=2)
        self.t_switch = self.t_start + self.duration / 2
        self.t_stop = self.t_start + self.duration
        #print('debug: ', self.duration, self.t_start, self.t_switch, self.t_stop)
        self.action = np.zeros(self.act_dim)
        self.steps = 0
        if np.random.rand() > 0.5:
            self.sign = 1.0
        else:
            self.sign = -1.0
        self.max_acc = np.random.uniform(low=self.max_acc_range[0], high=self.max_acc_range[1])
        #print(self.max_acc)
 
    def sample(self, obs):
        x = 0.0
        y = 0.0
        if self.steps > self.t_start[0]:
            x = self.sign*self.max_acc
        if self.steps > self.t_start[1]:
            y = self.sign*self.max_acc
        if self.steps > self.t_switch[0]:
            x = -self.sign*self.max_acc 
        if self.steps > self.t_switch[1]:
            y =  -self.sign*self.max_acc 
        if self.steps > self.t_stop[0]:
            x = 0.0 
        if self.steps > self.t_stop[1]:
            y = 0.0 
 
        self.steps += 1
 
        z = 0.0
        #print('****',self.steps, x, y)
        acc_n = np.asarray([x,y,z])
        
        v_t = obs[3:6] 
        v_dvec = v_t / np.linalg.norm(v_t)

        z = np.asarray([0.,0.,1.])
        C = attu.DCM3(z , v_dvec)
        acc = C.dot(acc_n)

        #print('DOT: ', np.dot(acc/np.linalg.norm(acc) , v_dvec), np.linalg.norm(v_t))
        #print('ACC: ', acc, np.linalg.norm(acc))
        #print(self.action, np.linalg.norm(self.action) / np.linalg.norm(acc)) 
        #print(acc , v_t,  np.linalg.norm(v_t))
        self.action = acc
        return acc 


