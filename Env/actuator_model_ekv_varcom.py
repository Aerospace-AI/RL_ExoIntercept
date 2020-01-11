import numpy as np

class Actuator_model_ekv(object):

    """
        Thruster model for spacecraft computes force and torque in the body frame and converts
        to inertial frame
        Commanded thrust is clipped to lie between zero and one, and then scaled based off of 
        thrust capability
        NOTE: maximum thrust is achieved with:
            tcmd = np.zeros(12)
            tcmd[0:2]=np.ones(2)
            tcmd[4:6]=np.ones(2)
            tcmd[8:10]=np.ones(2)
        and is norm([2,2,2])=3.46

    """
    def __init__(self,  pulsed=False, max_thrust_trans=2.0, max_thrust_att=0.2, debug_fail=False, fail_idx_override=None, Isp=210.0, 
                 com=np.zeros(3), com_mag_max=0.0,  com_dir=None, com_fuel_scale=25, max_fuel=10):
        #                   dvec                      body position
        config = [
                      [ 0.0, -1.0,    0.0,    0.00,  -0.25,    0.0 ],  # upper -x face, 
                      [ 0.0,  1.0,    0.0,    0.00,   0.25,    0.0 ],  # upper +x face, 

                      [ 0.0,  0.0,    1.0,    0.0,    0.00,   0.25 ],  # left -y face, 
                      [ 0.0,  0.0,   -1.0,    0.0,    0.00,  -0.25 ],  # left +y face, 

                      [ 0.0,  0.0,    1.0,    0.00,  -0.25,    0.0 ],  # rotate around x      4
                      [ 0.0,  0.0,   -1.0,    0.00,   0.25,    0.0 ],  # rotate around x      5

                      [ 0.0, -1.0,    0.0,    0.00,    0.0,   0.25 ],  # rotate around x      6
                      [ 0.0,  1.0,    0.0,    0.00,    0.0,  -0.25 ],  # rotate around x      7

                      [ 0.0,  0.0,   -1.0,    0.50,   0.00,  -0.25 ],  # rotate around y       8
                      [ 0.0,  0.0,    1.0,   -0.50,   0.00,   0.25 ],  # rotate around y      9

                      [ 0.0,  0.0,    1.0,    0.50,   0.00,   0.25 ],  # rotate around y     10
                      [ 0.0,  0.0,   -1.0,   -0.50,   0.00,  -0.25 ],  # rotate around y      11

                      [ 0.0, -1.0,    0.0,    0.5,   -0.25,    0.00 ],  # rotate around z      12
                      [ 0.0,  1.0,    0.0,   -0.5,    0.25,    0.00 ],  # rotate around z     13

                      [ 0.0,  1.0,    0.0,    0.5,    0.25,    0.00 ],  # rotate around z     14
                      [ 0.0, -1.0,    0.0,   -0.5,   -0.25,    0.00 ]   # rotate around z      15
                    ]

        config = np.asarray(config)
        
        self.dvec = config[:,0:3]

        self.position = config[:,3:6]

        self.num_actuators = 10 

        trans_thrust = max_thrust_trans*np.ones(4)
        rot_thrust = max_thrust_att*np.ones(12)
        #rot_thrust[0:4] = rot_thrust[0:4] / 2 
        self.max_thrust1 = np.hstack((trans_thrust,rot_thrust))

        self.reward_scale_att = np.sum(rot_thrust)

        self.com = com
        self.com_dir = com_dir
        self.com_fuel_scale = com_fuel_scale
        self.com_mag_max = com_mag_max
        self.max_fuel = max_fuel
        self.fuel_reward_scale = np.sum(self.max_thrust1)
        self.fuel_reward_scale_att = np.sum(rot_thrust)
        self.pulsed = pulsed 
   
        self.Isp = Isp 
        self.g_o = 9.81
 
        self.eps = 1e-8

        self.mdot = None 

        self.debug_fail = debug_fail

        self.fail_idx_override = fail_idx_override

        self.fail_scale = None 
                             
        self.fail_idx = 0 

        self.fail = False

        print('thruster model x( fixed ): ',max_thrust_trans, max_thrust_att, max_thrust_att / max_thrust_trans)

    def thrust_map(self,commanded_thrust):
        thrust = np.zeros(16)
        thrust[0:4] = commanded_thrust[0:4]
        if thrust[0] > self.eps and thrust[1] > self.eps:
            thrust[0] = thrust[1] = -1
        if thrust[2] > self.eps and thrust[3] > self.eps:
            thrust[2] = thrust[3] = -1

        #print('CT: ', commanded_thrust)
        thrust[4]  = commanded_thrust[4]
        thrust[5]  = commanded_thrust[4]
        thrust[6]  = commanded_thrust[5]
        thrust[7]  = commanded_thrust[5]
        thrust[8] = commanded_thrust[6]
        thrust[9] = commanded_thrust[6]
        thrust[10] = commanded_thrust[7]
        thrust[11] = commanded_thrust[7]
        thrust[12] = commanded_thrust[8]
        thrust[13] = commanded_thrust[8]
        thrust[14] = commanded_thrust[9]
        thrust[15] = commanded_thrust[9]

        return thrust
              
    def set_action(self,commanded_thrust, fuel_used):
        #print('1: ', commanded_thrust)

        assert commanded_thrust.shape[0] == self.num_actuators

        if self.pulsed:
            commanded_thrust = commanded_thrust > self.eps
            commanded_thrust = self.thrust_map(commanded_thrust)
        commanded_thrust = np.clip(commanded_thrust, 0, 1.0) * self.max_thrust1

        if self.fail:
            if self.fail_idx_override is None:
                fail_idx = self.fail_idx
            else:
                fail_idx = self.fail_idx_override
   
            if self.debug_fail:
                orig_commanded_thrust = commanded_thrust.copy()
            commanded_thrust[fail_idx] = commanded_thrust[self.fail_idx] * self.fail_scale
            if self.debug_fail:
                if not np.all(orig_commanded_thrust ==  commanded_thrust):
                    print('orig: ', orig_commanded_thrust)
                    print('mod:  ', commanded_thrust) 

        #print('2: ', commanded_thrust)
 
        force = np.expand_dims(commanded_thrust,axis=1) * self.dvec

        com = self.com * fuel_used / self.max_fuel
        com = np.clip(com, -self.com, self.com)
        #print(self.com, com, fuel_used) 
        torque = np.cross(self.position + com, force)

        force = np.sum(force,axis=0)
        torque = np.sum(torque,axis=0)

        mdot = -np.sum(np.abs(commanded_thrust)) / (self.Isp * self.g_o)
        #print(mdot, np.sum(np.abs(commanded_thrust)), self.g_o)
        self.mdot = mdot # for rewards
        self.force = force
        self.torque = torque
        self.commanded_thrust = commanded_thrust

    def get_action(self):  
        return self.commanded_thrust, self.force, self.torque, self.mdot
