import numpy as np
from time import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import env_utils as envu
import pylab
import matplotlib
import render_traj_sensor
#import render_traj_paper_6dof2

class Env(object):
    def __init__(self, missile, target, missile_icgen, target_icgen,  logger,
                 render_func=render_traj_sensor.render_traj,
                 debug_steps=False,
                 precision_range=1000.,
                 use_offset=True,
                 precision_scale=300.,
                 reward_object=None,
                 debug_done=False,
                 nav_period=10,
                 terminate_on_vc=False,
                 tf_limit=5000.0, allow_plotting=True, print_every=1,):  
        self.nav_period = nav_period
        self.precision_scale = precision_scale
        self.precision_range = precision_range
        self.terminate_on_vc = terminate_on_vc
        self.debug_done = debug_done
        self.debug_steps = debug_steps 
        self.logger = logger
        self.missile = missile
        self.target = target
        self.use_offset = use_offset
        self.missile_icgen = missile_icgen
        self.target_icgen = target_icgen
 
        self.rl_stats = RL_stats(missile,logger,render_func, print_every=print_every,allow_plotting=allow_plotting ) 
        self.tf_limit = tf_limit
        self.display_errors = False
        self.allow_plotting = allow_plotting
        self.episode = 0
        self.reward_object = reward_object
        self.h = missile.dynamics.h
        if allow_plotting:
            plt.clf()
            plt.cla()
        print('Missile env fixed (h=',self.h)
        
    def reset(self):
        self.target_icgen.set_ic(self.target) 
        self.missile_icgen.set_ic(self.missile, self.target)
        self.missile.sensor.reset(self.missile)
        #self.target.sensor.reset()
        self.steps = 0
        self.t = 0.0

        self.missile.clear_trajectory()
        image_state, missile_state = self.missile.get_state_agent(self.t,update_dudv=True)

        # for now, use target initial position in FOV
        #if self.use_offset:
        #    self.missile.sensor.offset = self.missile.sensor.centered_pixel_coords

        self.missile.update_state(self.target)
        self.missile.update_trajectory(False, self.t, self.target)

        return image_state, missile_state

    def check_for_done(self):
        done = False
        if self.missile.check_for_done(self.steps):
            #print('1')
            done = True
        if self.target.check_for_done(self.steps):
            #print('2')
            done = True
        if self.t > self.tf_limit:
            #print('3')
            done = True
            if self.debug_done:
                print('Timeout: ', self.steps)
        if self.terminate_on_vc:
            r_tm = self.target.state['position'] - self.missile.state['position']
            v_tm = self.target.state['velocity'] - self.missile.state['velocity']
            vc = envu.get_vc(r_tm, v_tm)
            if vc < 0:
                done = True
            
        #if done:
        #    print('DONE') 
        return done

    #def render(self):
    #    self.rl_stats.show(self.episode)
    #    self.rl_stats.clear()
    #    if self.allow_plotting:
    #        self.render_func(self.missile.trajectory)


    def step(self,missile_action):
        missile_action = missile_action.copy()
        target_action = self.target.sample_action()
        if len(missile_action.shape) > 1:
            missile_action = missile_action[0]
        if len(target_action.shape) > 1:
            target_action = target_action[0]

        self.missile.prev_state = self.missile.state.copy()

        self.missile.actuator_model.set_action(missile_action)
        self.target.actuator_model.set_action(target_action)
    
        #print(self.target.actuator_model.get_action())

        if np.linalg.norm(self.missile.state['r_tm']) < self.precision_range:
            self.missile.dynamics.h = self.h / self.precision_scale
            self.target.dynamics.h = self.h / self.precision_scale
        else:
            self.missile.dynamics.h = self.h
            self.target.dynamics.h = self.h

        steps_to_sim = int(np.round(self.nav_period / self.missile.dynamics.h))
        #print(steps_to_sim, self.nav_period, self.missile.dynamics.h)
        #print(np.linalg.norm(self.missile.state['r_tm']), self.missile.dynamics.h, self.target.dynamics.h, steps_to_sim)

        ts = 0.0
        for i in range(steps_to_sim):
            self.missile.dynamics.next(self.t, self.missile)
            self.target.dynamics.next(self.t, self.target)

            done = self.check_for_done()
            ts += self.missile.dynamics.h
            update_dudv = ts > (self.h - 1e-10)
            if update_dudv:
                ts = 0.0
            #print(self.t, ts, update_dudv, steps_to_sim)
            if done:
                image_state, missile_state = self.missile.get_state_agent(self.t, update_dudv=update_dudv)
                break
            image_state, missile_state = self.missile.get_state_agent(self.t, update_dudv=update_dudv)

            if self.missile.sensor.check_for_vio():
                done = True
                if self.steps <= 5 and self.debug_steps:
                    print('FEW STEPS: ')
                    print(self.missile.trajectory['position'])
                    print(self.missile.trajectory['velocity'])
                    print(self.missile.trajectory['thrust'])
            if done:
                break
            self.t += self.missile.dynamics.h

        self.steps+=1
        self.missile.update_state(self.target)
        reward,reward_info = self.reward_object.get( self.missile, missile_action, done, self.steps)
        self.missile.update_trajectory(done, self.t, self.target)
        if done:
            self.episode += 1
        return image_state, missile_state, reward, done,reward_info


    def test_policy_batch(self, agent , n, print_every=100,  keys=None, test_mode=True):
        t0 = time()

        print('worked 1')
        agent.policy.test_mode = test_mode 
        self.missile.use_trajectory_list = True
        self.episode = 0
        self.missile.trajectory_list = []
        self.display_errors = True
        for i in range(n):
            agent.run_episode()
            #for k in all_keys:
            #    if not k in keys:
            #        self.missile.trajectory_list[-1][k] = None
            if i % print_every == 0 and i > 0:
                print('i (et): %d  (%16.0f)' % (i,time()-t0 ) )
                t0 = time()
                self.missile.show_cum_stats()
                print(' ')
                self.missile.show_final_stats(type='final')
        print('')
        self.missile.show_cum_stats()
        print('')
        self.missile.show_final_stats(type='final')
        print('')
        self.missile.show_final_stats(type='ic')


 
class RL_stats(object):
    
    def __init__(self, missile, logger, render_func, allow_plotting=True, print_every=1 ):
        self.logger = logger
        self.render_func = render_func
        self.missile = missile
        self.keys = ['r_f',  'v_f', 'r_i', 'v_i', 'norm_rf', 'hit_100cm', 'hit_50cm', 'norm_vf',  'thrust', 'norm_thrust','fuel', 'rewards', 'fuel_rewards', 
                     'norm_af', 'norm_wf',  'att_rewards', 'att_penalty', 'attitude', 'w', 'a_f', 'w_f',
                     'w_rewards', 'w_penalty', 'fov_penalty', 'hit_rewards', 'tracking_rewards', 'tracking_error', 
                     'optflow_error','pixel_icoords','theta_cv', 'steps']

        self.formats = {}
        for k in self.keys:
            self.formats[k] = '{:8.2f}'  # default
        self.formats['tracking_error'] = '{:8.4f}' 
        self.formats['optflow_error'] = '{:8.4f}'
        self.stats = {}
        self.history =  { 'Episode' : [] , 'MeanReward' : [], 'StdReward' : [] , 'MinReward' : [],  'Policy_KL' : [], 'Policy_Beta' : [], 'Variance' : [], 'Policy_Entropy' : [], 'ExplainedVarNew' :  [] , 
                          'Norm_rf' : [], 'Norm_vf' : [], 'SD_rf' : [], 'SD_vf' : [], 'Max_rf' : [], 'Max_vf' : [], 
                          'Model ExpVarOld' : [], 'Model P Loss Old' : [], 
                          'Norm_af' : [], 'Norm_wf' : [], 'SD_af' : [], 'SD_wf' : [], 'Max_af' : [], 'Max_wf' : [], 'MeanSteps' : [], 'MaxSteps' : []} 

        self.plot_learning = self.plot_agent_learning
        self.clear()
        
        self.allow_plotting = allow_plotting 
        self.last_time  = time() 

        self.update_cnt = 0
        self.episode = 0
        self.print_every = print_every 

        
        if allow_plotting:
            plt.clf()
            plt.cla()
            self.fig2 = plt.figure(2,figsize=plt.figaspect(0.5))
            self.fig3 = plt.figure(3,figsize=plt.figaspect(0.5))
            self.fig4 = plt.figure(4,figsize=plt.figaspect(0.5))

    def save_history(self,fname):
        np.save(fname + "_history", self.history)

    def load_history(self,fname):
        self.history = np.load(fname + ".npy")

    def clear(self):
        for k in self.keys:
            self.stats[k] = []
    
    def update_episode(self,sum_rewards,steps):    
        self.stats['rewards'].append(sum_rewards)
        self.stats['fuel_rewards'].append(np.sum(self.missile.trajectory['fuel_reward']))
        self.stats['tracking_rewards'].append(np.sum(self.missile.trajectory['tracking_reward']))
        self.stats['tracking_error'].append(self.missile.trajectory['tracking_error'])
        self.stats['pixel_icoords'].append(self.missile.trajectory['pixel_coords'][0])
        self.stats['theta_cv'].append(self.missile.trajectory['theta_cv'])
        self.stats['optflow_error'].append(self.missile.trajectory['optflow_error'])
        self.stats['att_penalty'].append(np.sum(self.missile.trajectory['att_penalty']))

        self.stats['att_rewards'].append(np.sum(self.missile.trajectory['att_reward']))
        #self.stats['att_rewards'].append(np.asarray(self.missile.trajectory['tracking_reward']))
       
        self.stats['fov_penalty'].append(np.sum(self.missile.trajectory['fov_penalty']))
        self.stats['w_penalty'].append(np.sum(self.missile.trajectory['w_penalty']))
        self.stats['w_rewards'].append(np.sum(self.missile.trajectory['w_reward']))
        self.stats['hit_rewards'].append(np.sum(self.missile.trajectory['hit_reward'])) 
        self.stats['attitude'].append(self.missile.trajectory['attitude_321'])
        self.stats['w'].append(self.missile.trajectory['w'])

        self.stats['r_f'].append(self.missile.trajectory['r_tm'][-1])
        self.stats['v_f'].append(self.missile.trajectory['v_tm'][-1])
        self.stats['w_f'].append(self.missile.trajectory['w'][-1])
        self.stats['a_f'].append(self.missile.trajectory['attitude_321'][-1][1:3])
        self.stats['w_f'].append(self.missile.trajectory['w'][-1])
        self.stats['r_i'].append(self.missile.trajectory['r_tm'][0])
        self.stats['v_i'].append(self.missile.trajectory['v_tm'][0])
        self.stats['hit_100cm'].append(np.sum(self.missile.trajectory['hit_100cm']))
        self.stats['hit_50cm'].append(np.sum(self.missile.trajectory['hit_50cm']))
        self.stats['norm_rf'].append(np.linalg.norm(self.missile.trajectory['r_tm'][-1]))
        self.stats['norm_vf'].append(np.linalg.norm(self.missile.trajectory['v_tm'][-1]))
        self.stats['norm_af'].append(np.linalg.norm(self.missile.trajectory['attitude_321'][-1]))  
        self.stats['norm_wf'].append(np.linalg.norm(self.missile.trajectory['w'][-1]))

        self.stats['norm_thrust'].append(np.linalg.norm(self.missile.trajectory['thrust'],axis=1))
        self.stats['thrust'].append(self.missile.trajectory['thrust'])
        self.stats['fuel'].append(np.linalg.norm(self.missile.trajectory['fuel'][-1]))
        self.stats['steps'].append(steps)
        self.episode += 1

    def check_and_append(self,key):
        if key not in self.logger.log_entry.keys():
            val = 0.0
        else:
            val = self.logger.log_entry[key]
        self.history[key].append(val)
 
    # called by render at policy update 
    def show(self):
 
        self.history['MeanReward'].append(np.mean(self.stats['rewards']))
        self.history['StdReward'].append(np.std(self.stats['rewards']))
        self.history['MinReward'].append(np.min(self.stats['rewards']))

        self.check_and_append('Policy_KL')
        self.check_and_append('Policy_Beta')
        self.check_and_append('Variance')
        self.check_and_append('Policy_Entropy')
        self.check_and_append('ExplainedVarNew')
        self.check_and_append('Model ExpVarOld')
        self.check_and_append('Model P Loss Old')
        self.history['Episode'].append(self.episode)

        self.history['Norm_rf'].append(np.mean(self.stats['norm_rf']))
        self.history['SD_rf'].append(np.mean(self.stats['norm_rf']+np.std(self.stats['norm_rf'])))
        self.history['Max_rf'].append(np.max(self.stats['norm_rf']))

        self.history['Norm_vf'].append(np.mean(self.stats['norm_vf']))
        self.history['SD_vf'].append(np.mean(self.stats['norm_vf']+np.std(self.stats['norm_vf'])))
        self.history['Max_vf'].append(np.max(self.stats['norm_vf']))

        self.history['Norm_af'].append(np.mean(self.stats['norm_af']))
        self.history['SD_af'].append(np.mean(self.stats['norm_af']+np.std(self.stats['norm_af'])))
        self.history['Max_af'].append(np.max(self.stats['norm_af']))

        self.history['Norm_wf'].append(np.mean(self.stats['norm_wf']))
        self.history['SD_wf'].append(np.mean(self.stats['norm_wf']+np.std(self.stats['norm_wf'])))
        self.history['Max_wf'].append(np.max(self.stats['norm_wf']))


        self.history['MeanSteps'].append(np.mean(self.stats['steps']))
        self.history['MaxSteps'].append(np.max(self.stats['steps']))

        if self.allow_plotting:
            self.render_func(self.missile.trajectory)

            self.plot_rewards()
            self.plot_learning()
            self.plot_miss()
        if self.update_cnt % self.print_every == 0:
            self.show_stats()
            self.clear()
        self.update_cnt += 1

    def show_stats(self):
        et = time() - self.last_time
        self.last_time = time()

        r_f = np.linalg.norm(self.stats['r_f'],axis=1)
        v_f = np.linalg.norm(self.stats['v_f'],axis=1)       
 
        f = '{:6.2f}'
        print('Update Cnt = %d    ET = %8.1f   Stats:  Mean, Std, Min, Max' % (self.update_cnt,et))
        for k in self.keys:
            f = self.formats[k]    
            v = self.stats[k]
            if k == 'thrust' or  k=='theta_cv' or k=='tracking_error' or k=='optflow_error' or k=='norm_thrust' or k=='attitude' or k=='w': 
                v = np.concatenate(v)
            v = np.asarray(v)
            if len(v.shape) == 1 :
                v = np.expand_dims(v,axis=1)
            s = '%-8s' % (k)
            #print('foo: ',k,v)
            s += envu.print_vector(' |',np.mean(v,axis=0),f)
            s += envu.print_vector(' |',np.std(v,axis=0),f)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)

        #print('R_F, Mean, SD, Min, Max: ',np.mean(r_f), np.std(r_f))
        #print('V_F, Mean, SD, Min, Max: ',np.mean(v_f), np.mean(v_f))
 
 
    def plot_rewards(self):
        self.fig2.clear()
        plt.figure(self.fig2.number)
        self.fig2.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        
        lns1=ax.plot(ep,self.history['MeanReward'],'r',label='Mean R')
        lns2=ax.plot(ep,np.asarray(self.history['MeanReward'])-np.asarray(self.history['StdReward']),'b',label='SD R')
        lns3=ax.plot(ep,self.history['MinReward'],'g',label='Min R')
        lns4=ax2.plot(ep,self.history['MaxSteps'],'c',linestyle=':',label='Max Steps')
        lns5=ax2.plot(ep,self.history['MeanSteps'],'m',linestyle=':',label='Mean Steps')

        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Episode")

        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        ax = plt.gca()
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig2.canvas.draw()

    def plot_agent_learning(self):
        self.fig3.clear()
        plt.figure(self.fig3.number)
        self.fig3.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        lns1=ax.plot(ep,self.history['Policy_Entropy'],'r',label='Entropy')
        lns2=ax2.plot(ep,self.history['Policy_KL'],'b',label='KL Divergence')
        lns3=ax.plot(ep,self.history['ExplainedVarNew'],'g',label='Explained Variance')
        lns4=ax.plot(ep,self.history['Policy_Beta'],'k',label='Beta')
        foo = 10*np.asarray(self.history['Variance'])
        lns5=ax.plot(ep,foo,'m',label='10X Variance')


        lns = lns1+lns2+lns3+lns4+lns5
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Update")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig3.canvas.draw()

    def plot_model_learning(self):
        self.fig3.clear()
        plt.figure(self.fig3.number)
        self.fig3.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        ax = plt.gca()
        ax2 = ax.twinx()
        lns1=ax.plot(ep,self.history['Model P Loss Old'],'r',label='Model Loss')
        lns2=ax2.plot(ep,self.history['Model ExpVarOld'],'b',label='Model ExpVar')

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("Update")
        ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig3.canvas.draw()

    def plot_miss(self):
        self.fig4.clear()
        plt.figure(self.fig4.number)
        self.fig4.set_size_inches(8, 3, forward=True)
        ep = self.history['Episode']
        
        plt.plot(ep,self.history['Norm_rf'],'r',label='Norm_rf (m)')
        plt.plot(ep,self.history['SD_rf'], 'b',linestyle=':',label='SD_rf (m)')
        plt.plot(ep,self.history['Max_rf'], 'g',linestyle=':',label='Max_rf (m)')
 
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
        ax = plt.gca()
        ax.set_xlabel("Episode")
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().subplots_adjust(top=0.85)
        self.fig4.canvas.draw()

 
