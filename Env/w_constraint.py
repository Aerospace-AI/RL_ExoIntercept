import numpy as np

class W_constraint(object):

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
        w = state['w'].copy()
        if np.any(np.abs(w) > self.w_limit):
            margin = -1
        else:
            margin = 1
        return margin 

    def get_reward(self,state):
        w = state['w'].copy()
        reward = self.get_r(w[0], self.w_margin[0], self.w_limit[0]) + \
                 self.get_r(w[1], self.w_margin[1], self.w_limit[1]) + \
                 self.get_r(w[2], self.w_margin[2], self.w_limit[2])
        return reward 

    def get_r(self,ac,margin,limit):
        ac = np.abs(ac)
        r = 0.0
        
        tau = margin / 2
        if ac > ( limit - margin):
            err = (limit - margin) - ac
        else:
            err = 0.0 
        #print('err: ',ac, err)
        if err < 0: 
            r = -self.w_coeff * err 
        return r    


    def get_term_reward(self,state):
        w =  state['w']
        vio = w > self.w_limit
        self.violation_type += vio
        if np.any(vio):
            if self.cnt % 100 == 0:
                print('*** W VIO TYPE CNT: ',self.violation_type)
            self.cnt += 1
        margin = self.get_margin(state)
        if margin < 0:
            return self.w_penalty 
        else:
            return 0.0


        
