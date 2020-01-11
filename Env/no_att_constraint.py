import numpy as np
import attitude_utils as attu

class No_att_constraint(object):

    def  __init__(self, attitude_parameterization, terminate_on_violation=True, 
                  attitude_limit=(np.pi/2+np.pi/8, np.pi/2-np.pi/16, np.pi/2-np.pi/16),
                  attitude_margin=(np.pi/8, np.pi/8, np.pi/8),  
                    attitude_coeff=-10.0, attitude_penalty=-100.):
        self.attitude_parameterization = attitude_parameterization
        self.attitude_margin = attitude_margin
        self.attitude_limit = attitude_limit
        self.attitude_coeff = attitude_coeff
        self.attitude_penalty = attitude_penalty
        self.terminate_on_violation = terminate_on_violation
        print('Attitude Constraint')
        self.violation_type = np.zeros(3)
        self.cnt = 0

    def get_margin(self,state,debug=False):
        return 1 

    def get_reward(self,state):
        reward = 0
        return reward 



    def get_term_reward(self,state):
        return 0.0

