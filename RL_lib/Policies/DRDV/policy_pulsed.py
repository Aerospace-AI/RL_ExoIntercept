import numpy as np
from rl_utils import Action_converter
import env_utils as envu
import attitude_utils as attu
from time import time

class Policy(object):

    def __init__(self, env, pulsed=True, actions_per_dim=3, nominal_g=-1e-2, nominal_mass=500):
        self.env = env
        self.nominal_g = nominal_g 
        self.net = self.Net()
        #assert not env.scale_agent_action
        env.scale_agent_action = False
        self.actions_per_dim = actions_per_dim
        self.pulsed = pulsed
        self.nominal_mass = nominal_mass
        self.action_converter = Action_converter(1,actions_per_dim)

    def get_thrust(self,state):
        state = np.squeeze(state)
        rg = state[0:3]
        vg = state[3:6]
        gamma = 0.0
        p = [gamma + np.linalg.norm(self.nominal_g)**2/2  ,  0., -2. * np.dot(vg,vg)  , -12. * np.dot(vg,rg) , -18. * np.dot(rg , rg)]

        p_roots = np.roots(p)
        for i in range(len(p_roots)):
            if np.abs(np.imag(p_roots[i])) < 0.0001:
                if p_roots[i] > 0:
                    t_go = np.real(p_roots[i])
        if t_go > 0:
            a_c = -6. * rg/t_go**2 - 4. * vg /t_go #- self.env.dynamics.g
        else:
            a_c = np.zeros(3) 

        thrust = a_c * self.nominal_mass

        if self.pulsed:
            idx = self.action_converter.action2idx(thrust)
            thrust = np.squeeze(self.action_converter.idx2action(idx))
            thrust *= self.env.lander.max_thrust
        thrust = envu.limit_thrust(thrust, self.env.lander.min_thrust, self.env.lander.max_thrust)
        return thrust 

    def sample(self, image_obs, obs, state):
        action = self.get_thrust(obs)
        #action = envu.reverse_thrust(action, self.env.lander.min_thrust, self.env.lander.max_thrust)
        action = np.expand_dims(action,axis=0)
        return action, action.copy(), self.net.initial_state

    def update(self, rollouts, logger): 
        logger.log({'PolicyLoss': 0.0,
                    'PolicyEntropy': 0.0,
                    'KL': 0.0,
                    'Beta': 0.0,
                    'Variance' : 0.0,
                    'lr_multiplier': 0.0})

    def update_scalers(self, rollouts):
        pass

    class Net(object):
        def __init__(self):
            self.initial_state = np.zeros((1,1))
