import numpy as np
import env_utils as envu

class Reward(object):

    """
        Minimizes Velocity Field Tracking Error

    """

    def __init__(self, reward_scale=1.0,  fuel_coeff=-0.01, hit_coeff = 10.0, 
                 hit_rlimit=2.0, tracking_sigma=0.05, optflow_sigma=0.005, 
                 tracking_coeff=1.00, att_coeff=-0.50, att_sigma=np.pi/8, tracking_bias=0.00, debug=False,
                 optflow_coeff=0.0, fov_coeff=-50.0, fov_rlim=5.0,  cv_coeff=0.0):

        self.reward_scale =         reward_scale
        self.fuel_coeff =           fuel_coeff

        self.hit_rlimit =       hit_rlimit

        self.fov_rlim =              fov_rlim
        self.fov_coeff =            fov_coeff
        self.optflow_coeff =        optflow_coeff

        self.hit_coeff =        hit_coeff

        self.tracking_coeff =       tracking_coeff
        self.att_coeff =            att_coeff
        self.att_sigma =            att_sigma
        self.tracking_bias =        tracking_bias
        self.tracking_sigma =       tracking_sigma
        self.optflow_sigma =        optflow_sigma
        self.debug =                debug

        print('Reward_terminal')

    def get(self, missile,  action, done, steps):
        r_tm         =  missile.state['r_tm']
        v_tm         =  missile.state['v_tm']
        miss         =  np.linalg.norm(r_tm)


        fov_penalty = 0.0
        pixel_coords = (missile.sensor.cs_coords) # - missile.sensor.offset) / (missile.sensor.eo_model.p_x // 2)


        if not missile.sensor.check_for_vio():
            tracking_error = np.linalg.norm(pixel_coords)
            att_error = missile.attitude_parameterization.distance(missile.att_v_b, missile.state['target_attitude'])
            #print(att_error)
            optflow = np.asarray([missile.sensor.du,missile.sensor.dv])
            optflow_error = np.linalg.norm(optflow)
            tracking_reward =  self.tracking_coeff * np.exp(-optflow_error**2/self.optflow_sigma**2)
            att_reward =  self.att_coeff * att_error #np.exp(-att_error**2/self.att_sigma**2)
            r_tracking = tracking_reward + att_reward 
            #print(tracking_error**2/self.tracking_sigma**2 , optflow_error**2/self.optflow_sigma**2) 
        else:
            r_tracking = 0.0
            tracking_reward = 0.0
            att_reward = 0.0
            tracking_error = 0.0
            optflow_error = 0.0
            att_error = 0.0

        r_att = missile.att_constraint.get_reward(missile.state)
        r_w   = missile.w_constraint.get_reward(missile.state)

        landing_margin = 0.
        att_penalty = 0.0
        w_penalty = 0.0
        r_hit = 0.0
        hit_100cm = 0.0
        hit_50cm = 0.0
        if done:

            if missile.sensor.check_for_vio() and miss > self.fov_rlim:
                fov_penalty = self.fov_coeff


            att_penalty = missile.att_constraint.get_term_reward(missile.state)

            w_penalty = missile.w_constraint.get_term_reward(missile.state)




            att = np.abs(missile.state['attitude_321'][1:3])
            w   = np.abs(missile.state['w'])

            if self.debug or miss < self.hit_rlimit: 
                r_hit = self.hit_coeff
            if miss < 1.0:
                hit_100cm = 1.0
            if miss < 0.5:
                hit_50cm = 1.0

            #print(r_tm, miss, hit_50cm, hit_100cm)

        reward_info = {}
        r_fuel = self.fuel_coeff * np.sum(np.abs(missile.state['bf_thrust'][4:])) / (missile.actuator_model.fuel_reward_scale_att)

        reward_info['fuel'] = r_fuel

        reward1 = (r_w + fov_penalty +  w_penalty + r_att +  att_penalty +  r_tracking +  r_fuel + self.tracking_bias) * self.reward_scale
        reward2 = r_hit * self.reward_scale
        reward = (reward1, reward2)
        missile.trajectory['reward'].append(reward1 + reward2)
        missile.trajectory['hit_reward'].append(r_hit)
        missile.trajectory['hit_100cm'].append(hit_100cm)
        missile.trajectory['hit_50cm'].append(hit_50cm)
        missile.trajectory['fov_penalty'].append(fov_penalty)
        missile.trajectory['att_reward'].append(att_reward)
        missile.trajectory['att_penalty'].append(att_error)
        missile.trajectory['w_reward'].append(r_w)
        missile.trajectory['w_penalty'].append(w_penalty)
        missile.trajectory['tracking_reward'].append(r_tracking)
        missile.trajectory['tracking_error'].append(tracking_error)
        missile.trajectory['optflow_error'].append(optflow_error)
        missile.trajectory['fuel_reward'].append(r_fuel)
        return reward, reward_info


