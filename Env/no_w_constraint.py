import numpy as np

class No_w_constraint(object):

    def  __init__(self,  terminate_on_violation=True, 
                  w_limit=(2*np.pi, 2*np.pi, 2*np.pi),
                  w_margin=(np.pi/4, np.pi/4, np.pi/4),  
                  w_coeff=-10.0, w_penalty=-100.):
        self.w_margin = w_margin
        self.w_limit = w_limit
        self.w_coeff = w_coeff
        self.w_penalty = w_penalty
        self.terminate_on_violation = terminate_on_violation
        print('Rotational Velocity Constraint')
        self.violation_type = np.zeros(3)
        self.cnt = 0

    def get_margin(self,state,debug=False):
        return 1.0 

    def get_reward(self,state):
        return 0.0



    def get_term_reward(self,state):
        return 0.0


