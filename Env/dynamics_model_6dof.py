import attitude_utils as attu
import env_utils as envu 
import numpy as np
from time import time

class Dynamics_model_6dof(object):

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

    def __init__(self, h=0.5, noise_u=np.zeros(3), noise_sd=np.zeros(3), M=5.9722e24):  
        self.h = h 
        self.g_o = 9.81
        self.M = M
        self.noise_sd = noise_sd
        self.noise_u =  noise_u
        self.max_disturbance = np.zeros(3)
        self.max_norm_disturbance = 0.
        self.cnt = 0 
        self.G = 6.674e-11
        print('6dof dynamics model')

    def next(self, t , agent):

        J = agent.inertia_tensor * agent.state['mass'] / agent.nominal_mass
        w = agent.state['w']
        x = agent.get_state_dynamics()
            
        old_v = x[3:6].copy()

        #
        # get force and torque in body frame
        #
 
        BT, F,L,mdot = agent.actuator_model.get_action()

        #
        # convert force to acceleration
        #

        acc_body_frame = F / agent.state['mass']

        #
        # Find acceleration to inertial frame
        # Since the attitude is BN (body with respect to inertial) the associated DCM 
        # is BN and maps from inertial to body, so we need to invert it (transpose)
        # to map pfrom body to inertial (I think)
        # 

        noise = (self.noise_u + np.clip(self.noise_sd * np.random.normal(size=3), 0, 3*self.noise_sd)) /  agent.state['mass']
        # centrifugal force requires spaacecraft position in asteroid centered frame
        radial_dist =  np.linalg.norm(agent.state['position'])
        pos_dvec = agent.state['position'] / radial_dist


        dcm_NB = agent.attitude_parameterization.get_body_to_inertial_DCM(agent.state['attitude'])
        acc_inertial_frame = dcm_NB.dot(acc_body_frame)
        #print('acc inertial: ', acc_body_frame, acc_inertial_frame)
        thrust = acc_inertial_frame * agent.state['mass']

        g = self.G * self.M * pos_dvec / radial_dist*2
        #print('g: ', g, self.G * self.M,  pos_dvec / radial_dist*2) 
        disturbance = g + noise

        self.max_disturbance = np.maximum(self.max_disturbance,np.abs(disturbance))
        self.max_norm_disturbance = np.maximum(self.max_norm_disturbance,np.linalg.norm(disturbance))
        if self.cnt % 300000 == 0:
            #print('Dynamics: Max Disturbance (N):     ',agent.state['mass'] * self.max_disturbance, agent.state['mass'] * np.linalg.norm(self.max_disturbance)) 
            print('Dynamics: Max Disturbance (m/s^2): ',self.max_disturbance, np.linalg.norm(self.max_disturbance))
        self.cnt += 1 
        acc_inertial_frame += disturbance 
        #
        # Here we use the Euler rotational equations of motion to find wdot
        #

        Jinv = np.linalg.inv(J)
        w_tilde = attu.skew(w)
        wdot = -Jinv.dot(w_tilde).dot(J).dot(w) + Jinv.dot(L)
        #print('DEBUG: ',L,wdot)
        #
        # differential kinematic equation for derivative of attitude
        #
        # integrate w_bt (body frame agent rotation relative to target frame) to get 
        # agent attitude in target frame
        # w_bn is stored in agent (rotation in inertial frame, which is caused by thruster torque)
        # reward function will try to make  w_bt zero
        #

        w_bt = w
        qdot = agent.attitude_parameterization.qdot(agent.state['attitude'], w_bt)

        #
        # Use 4th order Runge Kutta to integrate equations of motion
        #

        ode = lambda t,x : self.eqom(t, x, acc_inertial_frame, qdot, wdot, mdot)
        x_next = envu.rk4(t, x, ode, self.h )
        attitude = x_next[10:]
        attitude = agent.attitude_parameterization.fix_attitude(attitude) # normalize quaternions
        # integrate w_bt (agent_body to targeta to get agent attitude in target frame)
        # w_bn is stored in agent (rotation in inertial frame, which is caused by thruster torque)

        #print(thrust_command, w, x_next[6:9])
        agent.state['position'] = x_next[0:3]
        agent.state['velocity'] = x_next[3:6]
        agent.state['w']        = x_next[6:9]
        agent.state['mass']     = np.clip(x_next[9],agent.dry_mass,None)
        agent.state['attitude'] = attitude 
        agent.state['attitude_321'] = agent.attitude_parameterization.q2Euler321(attitude) 

        if np.any(agent.state['velocity'] > 6000.) : # or agent.name == 'Missile': 
            print('dynamics: ', agent.name)
            print('\t: old_v',  old_v) 
            print('\t: v',  agent.state['velocity'])
            print('\t: inertial acc', acc_inertial_frame)
            print('\t: body acc',  acc_body_frame)
            print('\t: mass ', agent.state['mass'])
            print('\t: mdot ', mdot)
            print('\t: force ', F)

        #if not  np.all(agent.state['attitude'] < 4):
        #    print(agent.state['attitude'] , agent.state['w'])
        #assert np.all(agent.state['attitude'] < 4)

        agent.state['thrust'] = thrust 
        agent.state['bf_thrust'] = BT 
        agent.state['torque'] = L

        #if agent.name == 'Missile':
        #    print('DEBUG3: ',np.dot(agent.state['velocity'],agent.state['acceleration'])) 
        return x_next

    def eqom(self,t, x, acc, qdot, wdot, mdot):

        r = x[0:3]
        v = x[3:6]
        w = x[6:9]

        rdot = v
        vdot = acc

        xdot = np.zeros(10+qdot.shape[0])
        xdot[0:3] = v
        xdot[3:6] = acc
        xdot[6:9] = wdot
        xdot[9] = mdot
        xdot[10:] = qdot

        return xdot
 
         
       
        
        
