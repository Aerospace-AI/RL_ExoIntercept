
import numpy as np
import env_utils as envu
import attitude_utils as attu

class ZEM_policy(object):
    def __init__(self, ap, N=3,  augmented=True, max_acc=5*9.91 , vc=7000, dt=0.10):
        self.N = N
        self.ap = ap
        self.vc = vc
        self.dt = dt
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
        du = obs[3]
        dv = obs[2]
        dudt = du / self.dt
        dvdt = dv / self.dt

        
       
        acc_y = self.N * dudt * self.vc 
        acc_z = self.N * dvdt * self.vc

        if acc_y > self.max_acc / 3:
            acc_yp = 1.0 
        else:
            acc_yp = 0.0
        if acc_y < -self.max_acc / 3:
            acc_yn = 1.0 
        else:
            acc_yn = 0.0

        if acc_z > self.max_acc / 3:
            acc_zp = 1.0 
        else:
            acc_zp = 0.0
        if acc_z < -self.max_acc / 3:
            acc_zn = 1.0 
        else:
            acc_zn = 0.0

        #print('U: ', dudt, acc_y, acc_yn, acc_yp)
        #print('V: ', dvdt, acc_z, acc_zn, acc_zp)
 
        action = np.asarray([acc_yn, acc_yp, acc_zp, acc_zn]) 
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

