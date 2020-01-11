
import numpy as np
import env_utils as envu
import attitude_utils as attu

class ZEM_policy(object):
    def __init__(self, ap, N=3,  augmented=True, max_acc=5*9.91 ):
        self.N = N
        self.ap = ap
        self.augmented = augmented
        self.max_acc = max_acc
        self.t_y_p = np.asarray([0., 1., 0.])
        self.t_y_n = np.asarray([0.,-1., 0.])
        self.t_z_p = np.asarray([0., 0., 1.])
        self.t_z_n = np.asarray([0., 0., -1.])

        self.net = self.Net()

    def sample(self, image_obs, obs, state):
        obs = np.squeeze(obs)
        #print(obs.shape)
        r_m = obs[0:3]
        v_m = obs[3:6]
        r_t = obs[6:9]
        v_t = obs[9:12]
        a_t = obs[12:15]
        q   = obs[15:19]

        r_tm = r_t - r_m    
        v_tm = v_t - v_m
      
        vc = envu.get_vc(r_tm, v_tm)
        r = np.linalg.norm(r_tm)

        t_go = r / vc
        
        if self.augmented: 
            zem = r_tm + v_tm*t_go + 0.5 * a_t * t_go**2 
        else:
            zem = r_tm + v_tm*t_go
       
        acc = self.N * zem / t_go**2

        acc_bf = self.ap.q2dcm(q).dot(acc) 
      
        acc_mag = np.linalg.norm(acc_bf)
        acc_dir = acc_bf / np.linalg.norm(acc_bf)
        
        proj_t_y_p = np.dot(acc_dir, self.t_y_p)
        proj_t_y_n = np.dot(acc_dir, self.t_y_n)
        proj_t_z_p = np.dot(acc_dir, self.t_z_p)
        proj_t_z_n = np.dot(acc_dir, self.t_z_n)

        acc_0 = proj_t_y_p * acc_mag 
        acc_1 = proj_t_y_n * acc_mag 
        acc_2 = proj_t_z_p * acc_mag 
        acc_3 = proj_t_z_n * acc_mag 

        acc_0 = acc_0 > self.max_acc / 3
        acc_1 = acc_1 > self.max_acc / 3
        acc_2 = acc_2 > self.max_acc / 3
        acc_3 = acc_3 > self.max_acc / 3

        action = np.asarray([acc_1, acc_0, acc_2, acc_3]) 
        action = np.expand_dims(action, axis=0)
 
        #print(acc, los, np.dot(acc,los))
        return action, action, state 

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

