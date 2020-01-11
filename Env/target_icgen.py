import numpy as np
import attitude_utils as attu

class Target_icgen(object):

    def __init__(self, attitude_parameterization=None,
                 v_mag=(1000., 1000.),
                 v_theta=(0.0, np.pi/2),
                 v_phi=(-np.pi,np.pi),
                 min_mass=450, max_mass=500, min_init_position=(0.0,  0.0, 50000.), max_init_position=(0.0,  0.0, 50000.)):  

        self.attitude_parameterization=attitude_parameterization

        self.min_init_position = min_init_position
        self.max_init_position = max_init_position

        self.v_mag = v_mag
        self.v_theta = v_theta
        self.v_phi = v_phi
 
        self.min_mass = min_mass
        self.max_mass = max_mass



    def show(self):
        print('Target_icgen:')
 
    def set_ic(self , target):
       
        target.policy.reset()
 
        target.init_mass = np.random.uniform(low=self.min_mass, high=self.max_mass)
       
        pos = np.random.uniform(low=self.min_init_position, high=self.max_init_position,size=3)

        theta = np.random.uniform(low=self.v_theta[0],   high=self.v_theta[1])
        phi   = np.random.uniform(low=self.v_phi[0],     high=self.v_phi[1])
        mag_v     = np.random.uniform(low=self.v_mag[0],       high=self.v_mag[1])

        vx = mag_v * np.sin(theta) * np.cos(phi)
        vy = mag_v * np.sin(theta) * np.sin(phi)
        vz = mag_v * np.cos(theta)

        vel = np.asarray([vx, vy, vz])

        #print('target: ', pos, vel) 
        target.state['position'] = pos 
        target.state['velocity'] = vel 

        target.state['attitude'] = np.asarray([1.,0.,0.,0.]) 
        target.state['attitude_321'] = self.attitude_parameterization.q2Euler321(target.state['attitude'])
        target.state['w'] = np.zeros(3)
 
        target.state['thrust'] = np.zeros(3)
        target.state['mass']   = target.init_mass

        target.inertia_tensor = target.nominal_inertia_tensor 

        target.state['acceleration'] = np.zeros(3)
        target.state['accX_0'] = np.zeros(3)
        target.state['accX_1'] = np.zeros(3)
        target.state['accX_2'] = np.zeros(3)

        #print('target: ', target.state['position'], target.state['velocity'])
 
