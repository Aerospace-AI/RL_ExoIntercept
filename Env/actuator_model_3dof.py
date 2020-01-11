import numpy as np

class Actuator_model_3dof(object):

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
    def __init__(self,  max_acc=2.0*9.81 , debug_fail=False, fail_idx_override=None ):


        self.debug_fail = debug_fail

        self.fail_idx_override = fail_idx_override

        self.fail_scale = None

        self.fail_idx = 0

        self.fail = False

        self.num_actuators = 3
 
        self.max_acc = max_acc 

        self.fuel_reward_scale = 3 * np.sum(self.max_acc)

        print('3-dof Actuator Model: ', max_acc)

                  
    def set_action(self, acc):

        assert acc.shape[0] == 3 
        #print('before: ', acc, self.max_acc)

        acc = np.clip(acc, -self.max_acc, self.max_acc)
        if self.fail:
            if self.fail_idx_override is None:
                fail_idx = self.fail_idx
            else:
                fail_idx = self.fail_idx_override

            if self.debug_fail:
                orig_acc = acc.copy()
            acc[fail_idx] = acc[self.fail_idx] * self.fail_scale
            if self.debug_fail:
                if not np.all(orig_acc ==  acc):
                    print('orig: ', orig_acc, self.fail_scale)
                    print('mod:  ', acc)
        #print('after: ', acc)
        self.acc = acc

    def get_action(self):  
        return self.acc
