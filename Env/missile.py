import numpy as np
import env_utils as envu
import attitude_utils as attu

class Missile(object):

    def __init__(self,  target, actuator_model,  dynamics, name='Missile', sensor=None, use_trajectory_list=False, 
                 debug_done=False, debug_cv=False, align_cv=False, perturb_pn_velocity=True, bias_scale=(-0.00, 0.00),
                 attitude_parameterization=None, sensor_dvec=np.asarray([1.,0.,0.]), w_constraint=None, att_constraint=None, dry_mass=10.):

        self.target = target
        self.bias_scale = bias_scale
        self.actuator_model = actuator_model
        self.dynamics = dynamics
        self.att_constraint = None
        self.perturb_pn_velocity = perturb_pn_velocity 
        self.align_cv = align_cv
        self.sensor = sensor
        self.sensor_dvec = sensor_dvec
        self.debug_done = debug_done
        self.debug_cv = debug_cv
        self.use_trajectory_list = use_trajectory_list
        self.w_constraint = w_constraint
        self.att_constraint = att_constraint
        self.attitude_parameterization = attitude_parameterization
        self.dry_mass = dry_mass

        self.name = name

 
        self.state_keys = ['position','velocity','thrust','bf_thrust',  'torque', 'attitude', 'attitude_321', 'w',  'mass', 'r_tm', 'v_tm', 
                                'acceleration', 'accX_0', 'accX_1', 'accX_2']
        self.trajectory_list = []
        self.trajectory = {} 
        self.init_mass = 50.
        self.nominal_mass = self.init_mass
        m = self.init_mass
        h = 1
        r = 0.5
        self.inertia_tensor = m  * np.diag( [ r**2 / 2., (3*r**2+h**2) / 12. , (3*r**2+h**2) / 12.] )
        print('Inertia Tensor: ',self.inertia_tensor)
        self.nominal_inertia_tensor = self.inertia_tensor.copy()

        self.use_trajectory_list = use_trajectory_list        

        self.get_state_agent    = self.get_state_agent_sensor

        self.state = {}
        self.state['bf_thrust'] = np.zeros(4)
        self.prev_state = {}
        print('Missile Model: ')
        print(' - foo: ', 0.0)


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
 
    def clear_trajectory(self):
        for k in self.get_engagement_keys():
            self.trajectory[k] = []
        self.p_bias = np.random.uniform(low=self.bias_scale[0], high=self.bias_scale[1], size=3) 
        self.v_bias = np.random.uniform(low=self.bias_scale[0], high=self.bias_scale[1], size=3)
        self.state['target_attitude'] = self.state['attitude'].copy()

    def update_trajectory(self, done, t, target):

        es = self.get_engagement_state(t, target)
        for k,v in es.items():
            self.trajectory[k].append(v)
        if(done):
            if self.use_trajectory_list:
                self.trajectory_list.append(self.trajectory.copy())
    def set_att_align_cv(self):
        bf_optical_axis = self.sensor.eo_model.get_bf_optical_axis()
        v_dvec = self.state['velocity'] / np.linalg.norm(self.state['velocity'])
        if self.debug_cv:
            nf_optical_axis = self.sensor.eo_model.get_optical_axis_q(self.state['attitude'])
            print(v_dvec, nf_optical_axis)
            print('debug CV (before): ', nf_optical_axis.dot(v_dvec))

        C_nb = attu.DCM3(bf_optical_axis, v_dvec).T
        self.state['attitude'] = self.attitude_parameterization.dcm2q(C_nb)

        if self.debug_cv:
            nf_optical_axis = self.sensor.eo_model.get_optical_axis_q(self.state['attitude'])
            print('debug CV (after): ', nf_optical_axis.dot(v_dvec)) 
        
    def perturb_state(self,state):
        perturbed_state = state.copy()
        perturbed_state['position'] = state['position'] + np.abs(state['position']) * self.p_bias
        perturbed_state['velocity'] = state['velocity'] + np.abs(state['velocity']) * self.v_bias
        return perturbed_state
 
    def get_state_agent_sensor(self,t, update_dudv):
        if self.align_cv:
            self.set_att_align_cv()
        perturbed_state = self.perturb_state(self.target.state)
        state = self.sensor.get_image_state(self.state, object_locations=perturbed_state['position'],update_dudv=update_dudv)
        image = np.zeros((1,2,4,4))  # placeholder
        return image, state

    def get_state_agent_sensor_att_w2(self,t, update_dudv):
        if self.align_cv:
            self.set_att_align_cv()
        perturbed_state = self.perturb_state(self.target.state)
        state1 = self.sensor.get_image_state(self.state, object_locations=perturbed_state['position'],update_dudv=update_dudv)
        att_error =  self.attitude_parameterization.sub(self.state['attitude'], self.state['target_attitude'])
        state = np.hstack((state1, att_error,  self.state['w']))
        image = np.zeros((1,2,4,4))  # placeholder
        return image, state

    def get_state_agent_sensor_att_w_vf(self,t, update_dudv):
        if self.align_cv:
            self.set_att_align_cv()
        perturbed_state = self.perturb_state(self.target.state)
        state1 = self.sensor.get_image_state(self.state, object_locations=perturbed_state['position'],update_dudv=update_dudv)
        v_dvec = self.state['velocity'] / np.linalg.norm(self.state['velocity'])
        C_bn = self.attitude_parameterization.q2dcm(self.state['attitude'])
        v_dvec_b = C_bn.dot(v_dvec)
        v_err = v_dvec_b - self.sensor_dvec
        self.v_err = v_err
        #print(v_err)
        state = np.hstack((state1, v_err,  self.state['w']))
        image = np.zeros((1,2,4,4))  # placeholder
        return image, state

    def get_state_agent_sensor_att_w_vf2(self,t, update_dudv):
        if self.align_cv:
            self.set_att_align_cv()
        perturbed_state = self.perturb_state(self.target.state)
        state1 = self.sensor.get_image_state(self.state, object_locations=perturbed_state['position'],update_dudv=update_dudv)
        v_dvec = self.state['velocity'] / np.linalg.norm(self.state['velocity'])
        C_bn = self.attitude_parameterization.q2dcm(self.state['attitude'])
        v_dvec_b = C_bn.dot(v_dvec)
        C = attu.DCM2(self.sensor_dvec, v_dvec_b)
        att_v_b = self.attitude_parameterization.dcm2q(C)
        self.att_v_b = att_v_b
        #print(self.att_v_b)
        self.state['target_attitude'] = 1.0*np.asarray([1,0,0,0])
        state = np.hstack((state1, att_v_b,  self.state['w']))
        image = np.zeros((1,2,4,4))  # placeholder
        return image, state

    def get_state_agent_PN(self,t):
        if self.align_cv:
            self.set_att_align_cv()
        image = self.sensor.get_image_state(self.state, object_locations=self.target.state['position'])
        p_state = self.perturb_state(self.target.state)
        #print(pos, pos_n, pos / pos_n - 1)
        #print(r_dvec, r_dvec_n , r_dvec / r_dvec_n - 1)
        state = np.hstack((self.state['position'], self.state['velocity'], p_state['position'], p_state['velocity'],  self.target.policy.action ))
        image = np.zeros((1,2,4,4))  # placeholder
        return image, state


    def get_state_agent_PN_att(self,t, update_dudv):
        if self.align_cv:
            self.set_att_align_cv()
        image = self.sensor.get_image_state(self.state, object_locations=self.target.state['position'], update_dudv=update_dudv)
        p_state = self.perturb_state(self.target.state)
        state = np.hstack((self.state['position'], self.state['velocity'], p_state['position'], p_state['velocity'],  self.target.policy.action, self.state['attitude'] ))
        image = np.zeros((1,2,4,4))  # placeholder
        return image, state



    def get_state_dynamics(self):
        state = np.hstack((self.state['position'], self.state['velocity'],  self.state['w'], self.state['mass'],  self.state['attitude']))
        return state
 
    def show_cum_stats(self):
        print('Cumulative Stats (mean,std,max,argmax)')
        stats = {}
        argmax_stats = {}
        keys = ['thrust']
        formats = {'thrust' : '{:6.2f}'} 
        for k in keys:
            stats[k] = []
            argmax_stats[k] = []
        for traj in self.trajectory_list:
            for k in keys:
                v = traj[k]
                v = np.asarray(v)
                if len(v.shape) == 1:
                    v = np.expand_dims(v,axis=1)
                wc = np.max(np.linalg.norm(v,axis=1))
                argmax_stats[k].append(wc)
                stats[k].append(np.linalg.norm(v,axis=1))
                 
        for k in keys:
            f = formats[k]
            v = stats[k]
            v = np.concatenate(v)
            #v = np.asarray(v)
            s = '%-8s' % (k)
            #print('foo: ',k,v,v.shape)
            s += envu.print_vector(' |',np.mean(v),f)
            s += envu.print_vector(' |',np.std(v),f)
            s += envu.print_vector(' |',np.min(v),f)
            s += envu.print_vector(' |',np.max(v),f)
            argmax_v = np.asarray(argmax_stats[k])
            s += ' |%6d' % (np.argmax(argmax_v))
            print(s)

    def show_final_stats(self,type='final'):
        if type == 'final':
            print('Final Stats (mean,std,min,max)')
            idx = -1
        else:
            print('Initial Stats (mean,std,min,max)')
            idx = 0
 
        stats = {}
        keys = ['hit_reward', 'hit_100cm', 'hit_50cm', 'norm_vf', 'norm_rf', 'position', 'velocity', 'fuel', 'attitude_321', 'w' ]
        
        formats = { 'hit_reward' : '{:8.1f}', 'hit_100cm' :  '{:8.2f}', 'hit_50cm' :  '{:8.2f}', 'norm_rf' : '{:8.1f}', 'norm_vf' : '{:8.3f}', 'position' : '{:8.1f}' , 'velocity' : '{:8.3f}', 'fuel' : '{:6.2f}', 'attitude_321' : '{:8.3f}', 'w' : '{:8.3f}'}

        for k in keys:
            stats[k] = []
        for traj in self.trajectory_list:
            for k in keys:
                v = traj[k]
                v = np.asarray(v)
                if len(v.shape) == 1:
                    v = np.expand_dims(v,axis=1)
                stats[k].append(v[idx])

        for k in keys:
            f = formats[k]
            v = stats[k]
            s = '%-8s' % (k)
            s += envu.print_vector(' |',np.mean(v,axis=0),f)
            s += envu.print_vector(' |',np.std(v,axis=0),f)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)

    def update_state(self, target):
        self.state['r_tm'] = target.state['position'] - self.state['position']
        self.state['v_tm'] = target.state['velocity'] - self.state['velocity']
        #if np.any(self.state['v_tm'] > 6000.):
        #    print('DEBUG9: ', self.state['v_tm'], self.state['velocity'], target.state['velocity'])

         
    def get_engagement_state(self,t,target):
 
        engagement_state = {}
        engagement_state['t'] = t

        engagement_state['r_tm'] = self.state['r_tm'] 
        engagement_state['v_tm'] = self.state['v_tm'] 

        engagement_state['position'] = self.state['position'] 
        engagement_state['velocity'] = self.state['velocity'] 
        engagement_state['norm_rf'] = np.linalg.norm(self.state['r_tm'])
        engagement_state['norm_vf'] = np.linalg.norm(self.state['v_tm'])
        engagement_state['attitude'] = self.state['attitude']
        engagement_state['attitude_321'] = self.state['attitude_321']
        engagement_state['w']      =  self.state['w']
        engagement_state['thrust'] = self.state['thrust'] 
        engagement_state['mass']   = self.state['mass']
        engagement_state['fuel']   = self.init_mass-self.state['mass']
        engagement_state['vc'] =  envu.get_vc(engagement_state['r_tm'], engagement_state['v_tm']) 
        engagement_state['range'] = np.linalg.norm(self.state['position']) 
        engagement_state['pixel_coords'] = self.sensor.traj_pixel_coords
        engagement_state['optical_flow'] = np.hstack((self.sensor.du, self.sensor.dv))
        engagement_state['bf_thrust'] = self.state['bf_thrust']
        engagement_state['missile_acc'] = self.state['thrust'] / self.state['mass']
        engagement_state['target_acc'] =  self.target.state['thrust'] # not a typo, using thrust variable to hold acceleration  
         
        C_bn = self.attitude_parameterization.q2dcm(self.state['attitude'])
        opt_axis = self.sensor.eo_model.get_optical_axis(C_bn)
        v_dvec =  self.state['velocity'] / np.linalg.norm(self.state['velocity'])
        theta_cv = np.arccos(np.clip(np.dot(v_dvec,opt_axis), -1, 1))
        engagement_state['theta_cv'] = theta_cv
        return engagement_state

    def get_engagement_keys(self):
        keys = ['t', 'norm_rf', 'norm_vf', 'position', 'velocity', 'attitude', 'attitude_321', 'w', 'thrust', 'bf_thrust', 'torque', 'mass', 'fuel', 'vc', 'range', 'reward','fuel_reward','tracking_reward',  'optflow_error', 'tracking_error', 'att_penalty','att_reward','hit_reward','hit_50cm', 'hit_100cm', 'value','w_reward','w_penalty',  'pixel_coords', 'optical_flow', 'theta_cv', 'r_tm','v_tm','fov_penalty','missile_acc', 'target_acc']
        return keys



        



 
