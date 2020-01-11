import numpy as np
import rl_utils

class Arch(object):
    def __init__(self,  pt_policy, sample_p=0.5, vf_traj=None):
        self.vf_traj = vf_traj
        self.pt_policy = pt_policy
        self.sample_p = sample_p

    def run_episode(self, env, policy, val_func, model, recurrent_steps):

        image_obs, vector_obs = env.reset()
        image_observes, vector_observes, actions, vpreds, rewards1, rewards2,   policy_states, vf_states   =    [], [], [], [], [], [], [], []
        traj = {}
        done = False
        step = 0.0
        policy_state = policy.net.initial_state
        vf_state = val_func.get_initial_state()
        flag = 1
        self.pt_policy_state = self.pt_policy.net.initial_state
        while not done:

            vector_obs = vector_obs.astype(np.float64).reshape((1, -1))
   
            policy_states.append(policy_state)
            vf_states.append(vf_state)

            image_observes.append(image_obs)
            vector_observes.append(vector_obs)

            pt_action, pt_env_action, self.pt_policy_state = self.pt_policy.sample(image_obs, vector_obs, self.pt_policy_state)
            action, env_action, policy_state = policy.sample(image_obs, vector_obs, policy_state)

            foo = np.random.rand()
            if foo < self.sample_p:
                action = pt_action
                env_action = pt_env_action

            actions.append(action)

            vpred, vf_state = val_func.predict(vector_obs, vf_state)
            if self.vf_traj is not None:
                self.vf_traj['value'].append(vpred.copy())

            vpreds.append(vpred) 

            image_obs, vector_obs, reward, done, reward_info = env.step(env_action)

            reward1 = reward[0]
            reward2 = reward[1]
            if not isinstance(reward1, float):
                reward1 = np.asscalar(reward1)
            if not isinstance(reward1, float):
                reward2 = np.asscalar(reward2)
            rewards1.append(reward1)
            rewards2.append(reward2)
            step += 1e-3  # increment time step feature
            flag = 0

        if self.vf_traj is not None:
            self.vf_traj['value'].append(vpred.copy())

        traj['image_observes'] = np.concatenate(image_observes)
        traj['vector_observes'] = np.concatenate(vector_observes)
        traj['actions'] = np.concatenate(actions)
        traj['rewards1'] = np.array(rewards1, dtype=np.float64)
        traj['rewards2'] = np.array(rewards2, dtype=np.float64)
        traj['policy_states'] = np.concatenate(policy_states)
        traj['vf_states'] = np.concatenate(vf_states)
        traj['vpreds'] = np.array(vpreds, dtype=np.float64)
        traj['flags'] = np.zeros(len(vector_observes))
        traj['flags'][0] = 1

        traj['masks'] = np.ones_like(traj['rewards1'])

        return traj

    def update_scalers(self, policy, val_func, model, rollouts):
        policy.update_scalers(rollouts)
        val_func.update_scalers(rollouts)
        
    def update(self,policy,val_func,model,rollouts, logger):
        policy.update(rollouts, logger)
        val_func.fit(rollouts, logger) 


