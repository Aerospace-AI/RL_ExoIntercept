import numpy as np
import attitude_utils as attu
import env_utils as envu

class Missile_icgen(object):

    def __init__(self, attitude_parameterization=None,
                 position_r=(40000., 50000.0),
                 position_theta=(0.0, np.pi/2),
                 position_phi=(-np.pi,np.pi),
                 heading_error=(0,np.pi/16),
                 com_max=(0.0,0.0,0.0),
                 mag_v =(1000,1000),
                 attitude_error=(0,np.pi/16),
                 missile_wll=(0.0,0.0,0.0),
                 missile_wul=(0.0,0.0,0.0),
                 min_mass=50, max_mass=50,
                 p_engine_fail=0.0,
                 engine_fail_scale=(0.5,1.0),
                 debug_fail=False,
                 debug=False,
                 max_loopcnt = 10, 
                 noise_u=np.zeros(3), noise_sd=np.zeros(3),  
                 inertia_uncertainty_diag=0.0, inertia_uncertainty_offdiag=0.0): 
        self.mag_v = mag_v

 
        self.heading_error = heading_error
        self.position_r = position_r
        self.position_theta = position_theta
        self.position_phi = position_phi


        self.attitude_parameterization=attitude_parameterization
        self.attitude_error=attitude_error
        self.missile_wul=missile_wul
        self.missile_wll=missile_wll

        self.debug_fail = debug_fail
        self.p_engine_fail = p_engine_fail
        self.engine_fail_scale = engine_fail_scale
 
        self.min_mass = min_mass
        self.max_mass = max_mass

        self.com_max = com_max
        self.max_loopcnt = max_loopcnt
        self.noise_u = noise_u
        self.noise_sd = noise_sd

        self.infeasible_cnt = 0
        self.cnt = 0

        self.inertia_uncertainty_diag = inertia_uncertainty_diag
        self.inertia_uncertainty_offdiag = inertia_uncertainty_offdiag 

        self.max_pointing_error = 0.0    
        self.debug = debug


    def show(self):
        print('Landing_icgen:')
        print('\tmin_w / max_w:',self.min_w,self.max_w)
 
    def set_ic(self , missile,  target):
 
        # ENGINE FAILURE
        assert missile.actuator_model.fail is not None 
        missile.actuator_model.fail = np.random.rand() < self.p_engine_fail
        missile.actuator_model.fail_idx =  np.random.randint(low=0,high=missile.actuator_model.num_actuators)
        missile.actuator_model.fail_scale = np.random.uniform(low=self.engine_fail_scale[0], high=self.engine_fail_scale[1])
        if  self.debug_fail:
            print('Engine Fail? : ', self.p_engine_fail, missile.actuator_model.fail, missile.actuator_model.fail_idx, missile.actuator_model.fail_scale)
        com_max = 1.0*np.asarray([self.com_max[0], self.com_max[1]/2, self.com_max[2]/2])
        missile.actuator_model.com = np.random.uniform(low=-com_max,high=com_max)
        missile.init_mass = np.random.uniform(low=self.min_mass, high=self.max_mass)

        position_theta  = self.position_theta
        position_phi    = self.position_phi
        position_r      = self.position_r
        heading_error   = self.heading_error

        ######## don't barf on infeasible engagement, just wait till we get one #####

        loopcnt = 0
        while True:
 
            theta = np.random.uniform(low=position_theta[0],   high=position_theta[1])
            phi   = np.random.uniform(low=position_phi[0],     high=position_phi[1])
            r     = np.random.uniform(low=position_r[0],       high=position_r[1])

            rx = r * np.sin(theta) * np.cos(phi)
            ry = r * np.sin(theta) * np.sin(phi)
            rz = r * np.cos(theta)
            rel_pos = np.asarray([rx, ry, rz])
            pos = target.state['position'] + rel_pos 
            r_tm = -rel_pos
            mag_v = np.random.uniform(low=self.mag_v[0], high=self.mag_v[1])

            self.cnt += 1
            vel = attu.get_lead(mag_v, target.state['velocity'], r_tm)
            if vel is None:
                self.infeasible_cnt += 1
                if self.infeasible_cnt % 10 == 0:
                    print('!!!! %d out of %d initial conditions resulted in infeasible intercept  !!!!' % (self.infeasible_cnt , self.cnt))
            else:
                break
            loopcnt += 1
            if loopcnt > self.max_loopcnt:
                print('More than %d consecutive engagements were infeasible' % (self.max_loopcnt)) 
                assert False
        dvec_v = vel / np.linalg.norm(vel)

        #######################################

        missile.sensor.ideal_velocity = vel.copy()
        missile.sensor.ideal_attitude = attu.make_random_attitude_error(self.attitude_parameterization, (0.0, 0.0), dvec_v, missile.sensor_dvec)

        ############

        # now create a random heading error
        dvec_v, theta_debug = attu.make_random_heading_error(heading_error, dvec_v)
        vel = mag_v * dvec_v
        missile.state['position'] = pos 
        missile.state['velocity'] = vel  
        missile.state['attitude'] = attu.make_random_attitude_error(self.attitude_parameterization, self.attitude_error, dvec_v, missile.sensor_dvec)
        missile.state['attitude_321'] = self.attitude_parameterization.q2Euler321(missile.state['attitude'])
        missile.state['w'] = np.random.uniform(low=self.missile_wll, high=self.missile_wul, size=3)
 
        missile.state['thrust'] = np.zeros(3)
        missile.state['mass']   = missile.init_mass

        missile.state['acceleration'] = np.zeros(3)
        missile.state['accX_0'] = np.zeros(3)
        missile.state['accX_1'] = np.zeros(3)
        missile.state['accX_2'] = np.zeros(3)

        #print('missile position:  ', missile.state['position'])
        #print('missile velocity: ', missile.state['velocity'])
        #print('target position: ', target.state['position'])
        #print('target velocity: ', target.state['velocity'])

        opt_axis = missile.sensor.eo_model.get_optical_axis(self.attitude_parameterization.q2dcm(missile.state['attitude']))
        los =  r_tm / np.linalg.norm(r_tm)
        theta_cv = np.arccos(np.clip(np.dot(opt_axis, los),-1,1)) 

        it_noise1 = np.random.uniform(low=-self.inertia_uncertainty_offdiag, 
                                      high=self.inertia_uncertainty_offdiag, 
                                      size=(3,3))
        np.fill_diagonal(it_noise1,0.0)
        it_noise1 = (it_noise1 + it_noise1.T)/2
        it_noise2 = np.diag(np.random.uniform(low=-self.inertia_uncertainty_diag,
                            high=self.inertia_uncertainty_diag,
                            size=3))
        missile.inertia_tensor = missile.nominal_inertia_tensor + it_noise1 + it_noise2
        
    def get_zem(self, r_tm, v_tm):
        #r_tm = r_t - r_m
        #v_tm = v_t - v_m

        vc = envu.get_vc(r_tm, v_tm)
        r = np.linalg.norm(r_tm)

        t_go = r / vc

        zem = (r_tm + v_tm*t_go)
        los = r_tm / r

        zem_par = np.dot(zem , r_tm) * los
        zem_perp = zem - zem_par
        dv = np.linalg.norm(zem) / t_go
        foo = 2 * np.linalg.norm(zem) / (5*9.81*t_go**2) 
        return zem, dv, foo 
