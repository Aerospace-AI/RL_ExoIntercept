import attitude_utils as attu
import env_utils as envu 
import numpy as np
from time import time

class Dynamics_model_3dof(object):

    """
        The dynamics model take a agent model object (and later an obstacle object) and modifies  
        the state of the agent.

        The agent object instantiates an engine model, that maps body frame thrust and torque to
        the inertial frame.  Note that each agent can have its own intertial frame which can be 
        centered on the agent's target. 

        Currentlly this model does not model environmental dynamics, will be added later
 
        The agent model maintains a state vector: 
            position                                [0:3]
            velocity                                [3:6]
            body frame rotational velocity (w_bn)   [6:9]
            mass                                    [9]     
            attitude in target frame                [10:]  (size depends on attitude parameterization)
                 

    """

    def __init__(self, h=0.02, noise_u=np.zeros(3), noise_sd=np.zeros(3), M=5.9722e24, convert_body_frame_acc=False):  
        self.h = h 
        self.M = M
        self.convert_body_frame_acc = convert_body_frame_acc
        self.noise_sd = noise_sd
        self.noise_u =  noise_u
        self.max_disturbance = np.zeros(3)
        self.max_norm_disturbance = 0.
        self.cnt = 0 
        self.G = 6.674e-11
        print('3dof dynamics model')

    def next(self, t , agent):

        x = agent.get_state_dynamics()[0:6]
        if self.convert_body_frame_acc:
            acc_body_frame = agent.actuator_model.get_action()
            dcm_NB = agent.attitude_parameterization.get_body_to_inertial_DCM(agent.state['attitude'])
            acc_inertial_frame = dcm_NB.dot(acc_body_frame)
        else:
            acc_inertial_frame = agent.actuator_model.get_action()

        #print(x[1], x[4], acc_inertial_frame[1],np.linalg.norm(x[0:3]))
        noise = (self.noise_u + np.clip(self.noise_sd * np.random.normal(size=3), 0, 3*self.noise_sd)) /  agent.state['mass']
        radial_dist =  np.linalg.norm(agent.state['position'])
        pos_dvec = agent.state['position'] / radial_dist
        g = self.G * self.M * pos_dvec / radial_dist*2
        disturbance = g + noise

        self.max_disturbance = np.maximum(self.max_disturbance,np.abs(disturbance))
        self.max_norm_disturbance = np.maximum(self.max_norm_disturbance,np.linalg.norm(disturbance))
        if self.cnt % 300000 == 0:
            print('Dynamics: Max Disturbance (m/s^2): ',self.max_disturbance, np.linalg.norm(self.max_disturbance))
        self.cnt += 1 
        acc_inertial_frame += disturbance 

        ode = lambda t,x : self.eqom(t, x, acc_inertial_frame)
        x_next = envu.rk4(t, x, ode, self.h )

        agent.state['position'] = x_next[0:3]
        agent.state['velocity'] = x_next[3:6]
        agent.state['thrust'] = acc_inertial_frame

        return x_next

    
     
           
    def eqom(self,t, x, acc):

        r = x[0:3]
        v = x[3:6]

        rdot = v
        vdot = acc

        xdot = np.zeros(6)
        xdot[0:3] = v
        xdot[3:6] = acc

        return xdot
 
         
       
        
        
