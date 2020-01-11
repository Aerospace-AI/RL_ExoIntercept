

import numpy as np
import env_utils as envu
import attitude_utils as attu
from time import time

class Spiral_policy(object):
    def __init__(self,act_dim, max_acc=2*9.81, qp_range=(5,50) ):
        self.act_dim = act_dim
        self.max_acc = max_acc
        self.qp_range = qp_range 

    def reset(self):
        self.qp = np.random.randint(low=self.qp_range[0], high=self.qp_range[1]+1)
        self.action = np.zeros(self.act_dim)        
        self.steps_x = 0 
        self.steps_y = self.qp  
        self.dir_x = 1
        self.dir_y = 1 
    def sample(self, obs):
        if self.steps_x % (2 * self.qp) == 0:
            self.dir_x = -self.dir_x
        if self.steps_y % (2 * self.qp) == 0:
            self.dir_y = -self.dir_y
        #print(self.dir_x,self.dir_y, self.qp)     
        x = self.max_acc * self.dir_x 
        y = self.max_acc * self.dir_y
 
        self.steps_x += 1
        self.steps_y += 1
 
        z = 0.0
        #print('****',self.steps_y, y)
        acc_n = np.asarray([x,y,z])
        
        v_t = obs[3:6] 
        v_dvec = v_t / np.linalg.norm(v_t)

        z = np.asarray([0.,0.,1.])
        C = attu.DCM3(z , v_dvec)
        acc = C.dot(acc_n)

        #print('DOT: ', np.dot(acc/np.linalg.norm(acc) , v_dvec), np.linalg.norm(v_t))
        #print('ACC: ', acc)
        #print(self.action, np.linalg.norm(self.action) / np.linalg.norm(acc)) 
        #print(acc , v_t,  np.linalg.norm(v_t))
        self.action = acc
        return acc 


