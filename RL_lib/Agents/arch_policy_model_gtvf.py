import numpy as np
import rl_utils

class Arch(object):
    def __init__(self, gt_func=None, spred_func=None, vf_traj=None):
        self.gt_func = gt_func
        self.spred_func = spred_func
        self.vf_traj = vf_traj

    def run_episode(self, env, policy, val_func, model, recurrent_steps):

        image_obs, vector_obs = env.reset()
        gt_obs = self.gt_func(0.0).astype(np.float64).reshape((1, -1))

        image_observes, vector_observes, gt_observes, model_actions, actions, vpreds, rewards1, rewards2,   policy_states, vf_states, model_nobserves, model_states, model_errors, model_targets   =    [], [], [], [], [], [], [], [], [], [], [], [], [], []
        traj = {}
        done = False
        step = 0.0
        policy_state = policy.net.initial_state
        vf_state = val_func.get_initial_state()
        model_state = model.get_initial_state()
        model_error = model.get_initial_error()
        flag = 1
        while not done:


            vector_obs = vector_obs.astype(np.float64).reshape((1, -1))
   
            policy_states.append(policy_state)
            vf_states.append(vf_state)
            model_states.append(model_state)
            model_errors.append(model_error)

            gt_observes.append(gt_obs)

            image_observes.append(image_obs)
            vector_observes.append(vector_obs)

            action, env_action, policy_state = policy.sample(image_obs, vector_obs, policy_state)
            actions.append(action)
            model_actions.append(env_action)

            vpred, vf_state = val_func.predict(gt_obs, vf_state)

            vpreds.append(vpred) 

            image_obs, vector_obs, reward, done, reward_info = env.step(env_action)
            gt_obs = self.gt_func(0.0).astype(np.float64).reshape((1, -1))
            model_target = self.spred_func(0.0).astype(np.float64).reshape((1, -1)) 
            if model.model_list[0].use_image:
                model_nobs = image_obs
            else:
                model_nobs = vector_obs.astype(np.float64).reshape((1, -1))
            model_nobserves.append(model_nobs)
            model_targets.append(model_target)
            model_predict,  model_vpred, model_state, model_error = model.predict(action,  model_nobs, model_state, model_error, np.asarray([1]), np.asarray([flag]))

            if self.vf_traj is not None:
                self.vf_traj['value'].append(vpred.copy())
                self.vf_traj['spred'].append(np.squeeze(model_vpred.copy()))
 
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
            self.vf_traj['spred'].append(np.squeeze(model_vpred.copy()))

        traj['model_states'] = np.concatenate(model_states)
        traj['model_errors'] = np.concatenate(model_errors)
        traj['model_targets'] = np.concatenate(model_targets)
        traj['model_nobserves'] = np.concatenate(model_nobserves)
        traj['gt_observes'] = np.concatenate(gt_observes)
        traj['image_observes'] = np.concatenate(image_observes)
        traj['vector_observes'] = np.concatenate(vector_observes)
        traj['actions'] = np.concatenate(actions)
        traj['model_actions'] = np.concatenate(model_actions)
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
        model.update_scalers(rollouts)
        
    def update(self,policy,val_func,model,rollouts, logger):
        policy.update(rollouts, logger)
        val_func.fit(rollouts, logger) 
        model.fit(rollouts, logger)

