import numpy as np
import env_utils as envu
import attitude_utils as attu

class Target(object):

    def __init__(self,  policy, actuator_model,  dynamics, name='Target', sensor=None,  debug_done=False, 
                    attitude_parameterization=None, w_constraint=None, att_constraint=None, dry_mass=500):

        self.policy = policy
        self.actuator_model = actuator_model
        self.dynamics = dynamics
        self.att_constraint = None
        self.sensor = sensor
        self.debug_done = debug_done
        self.w_constraint = w_constraint
        self.att_constraint = att_constraint
        self.attitude_parameterization = attitude_parameterization
        self.dry_mass = dry_mass

        self.name = name
 
        self.state_keys = ['position','velocity','thrust','bf_thrust',  'torque', 'attitude', 'attitude_321', 'w',  'mass']
        self.init_mass = 500.
        self.nominal_mass = self.init_mass
        m = self.init_mass
        h = 2
        d = 2
        w = 2
        self.inertia_tensor = 1./12 * m * np.diag([h**2 + d**2 , w**2 + d**2, w**2 + h**2])
        print('Inertia Tensor: ',self.inertia_tensor)
        self.nominal_inertia_tensor = self.inertia_tensor.copy()
    
        self.get_state_agent    = self.get_state_agent_gt


        self.state = {}
        self.prev_state = {}
        print('Target Model: ')
        print(' - foo: ', 0.0)

    def sample_action(self):
        action = self.policy.sample(self.get_state_agent(0.0))
        return action

    def check_for_done(self, steps):
        done = False
        if self.att_constraint is not None and self.att_constraint.get_margin(self.state) < 0.0 and self.att_constraint.terminate_on_violation:
            done = True
            if self.debug_done:
                print('Missile Attitude Constraint: ', self.att_constraint.get_margin(self.state) , steps)
        if self.w_constraint is not None and self.w_constraint.get_margin(self.state) < 0.0 and self.w_constraint.terminate_on_violation:
            done = True
            if self.debug_done:
                print('Missile W Constraint: ', self.att_constraint.get_margin(self.state) , steps)
        return done
 
    def get_state_agent_sensor(self,t):
        image = self.sensor.get_image_state(self.state, object_locations=self.target_position)
        return image

    def get_state_agent_gt(self,t):
        state = np.hstack((self.state['position'], self.state['velocity'],  self.state['w'], self.state['attitude']))
        return state

    def get_state_dynamics(self):
        state = np.hstack((self.state['position'], self.state['velocity'],  self.state['w'], self.state['mass'],  self.state['attitude']))
        return state
 

         
        



 
