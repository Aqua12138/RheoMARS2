import os
import cv2
import numpy as np
import taichi as ti
import pickle as pkl
from fluidlab.utils.misc import is_on_server
import torch

class Recorder:
    def __init__(self, env):
        self.env = env
        self.target_file = env.target_file
        if self.target_file is not None:
            os.makedirs(os.path.dirname(self.target_file), exist_ok=True)

    def record(self, user_input=False):
        policy = self.env.demo_policy(user_input)
        taichi_env = self.env.taichi_env

        # initialize ...
        taichi_env_state = taichi_env.get_state()

        # start recording
        target = {
            'x'    : [],
            'used' : [],
            'mat'  : None
        }
        taichi_env.set_state(**taichi_env_state)
        action_p = policy.get_actions_p()
        if action_p is not None:
            taichi_env.apply_agent_action_p(action_p)
        
        save = False
        if save:
            os.makedirs(f'tmp/recorder', exist_ok=True)
            
        for i in range(self.env.horizon):
            if i < self.env.horizon_action:
                action = policy.get_action_v(i)
            else:
                action = None
            taichi_env.step(action)

            # get state
            if self.target_file is not None:
                cur_state = taichi_env.get_state()
                if taichi_env.has_particles:
                    target['x'].append(cur_state['state']['x'])
                    target['used'].append(cur_state['state']['used'])

            if save:
                img = taichi_env.render('rgb_array')
                cv2.imwrite(f'tmp/recorder/{i:04d}.png', img[:, :, ::-1])
            else:
                if not is_on_server():
                    taichi_env.render('human')

        if self.target_file is not None:
            target['mat'] = taichi_env.simulator.particles_i.mat.to_numpy()
            if os.path.exists(self.target_file):
                os.remove(self.target_file)
            pkl.dump(target, open(self.target_file, 'wb'))
            print(f'===> New target generated and dumped to {self.target_file}.')

    def replay_target(self):
        taichi_env = self.env.taichi_env
        target = pkl.load(open(self.target_file, 'rb'))

        for i in range(self.env.horizon):
            taichi_env.simulator.set_x(0, target['x'][i])
            taichi_env.simulator.set_used(0, target['used'][i])

            if not is_on_server():
                taichi_env.render('human')

    def replay_policy(self, policy_path):
        taichi_env = self.env.taichi_env

        policy = pkl.load(open(policy_path, 'rb'))

        taichi_env.apply_agent_action_p(policy.get_actions_p())

        save = False
        # save = True
        if save:
            os.makedirs(f'tmp/replay', exist_ok=True)
            
        for i in range(self.env.horizon):
            if i < self.env.horizon_action:
                action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            else:
                action = None
            taichi_env.step(action)

            if save:
                img = taichi_env.render('rgb_array')
                cv2.imwrite(f'tmp/replay/{i:04d}.png', img[:, :, ::-1])
            else:
                if not is_on_server():
                    if hasattr(taichi_env.loss, 'tgt_particles_x'):
                        taichi_env.render('human', taichi_env.loss.tgt_particles_x)
                    else:
                        taichi_env.render('human')


    def record_target_grid(self, user_input=False):
        policy = self.env.demo_policy(user_input)
        taichi_env = self.env.taichi_env

        # initialize ...
        taichi_env_state = taichi_env.get_state()

        # start recording
        target = {
            'x': [],
            'used': [],
            'mat': None
        }
        taichi_env.set_state(**taichi_env_state)
        action_p = policy.get_actions_p()
        if action_p is not None:
            taichi_env.apply_agent_action_p(action_p)

        save = False
        if save:
            os.makedirs(f'tmp/recorder', exist_ok=True)

        for i in range(self.env.horizon):
            if i < self.env.horizon_action:
                action = policy.get_action_v(i)
            else:
                action = None
            taichi_env.step(action)

            # get state
            if self.target_file is not None:
                cur_state = taichi_env.get_state()
                if taichi_env.has_particles:
                    target['x'].append(cur_state['state']['x'])
                    target['used'].append(cur_state['state']['used'])

            if save:
                img = taichi_env.render('rgb_array')
                cv2.imwrite(f'tmp/recorder/{i:04d}.png', img[:, :, ::-1])
            else:
                if not is_on_server():
                    taichi_env.render('human')

        if self.target_file is not None:
            target['mat'] = taichi_env.simulator.particles_i.mat.to_numpy()
            target['last_grid'] = taichi_env.simulator.grid.mass.to_numpy()[-2, ...]
            target['last_pos'] = target['x'][-1]

            if os.path.exists(self.target_file):
                os.remove(self.target_file)
            pkl.dump(target, open(self.target_file, 'wb'))
            print(f'===> New target generated and dumped to {self.target_file}.')

    def record_stochastic_data(self, horizon=20, sample_num=100):
        action = np.zeros(shape=self.env.agent.action_dim)
        taichi_env = self.env.taichi_env
        # start recording
        target = {
            'mat': [],
            'last_grid': [],
            'last_pos': [],
            'state': [],
            'statics': []
        }
        save = False
        if save:
            os.makedirs(f'tmp/recorder', exist_ok=True)
        init_states = []
        for b in range(sample_num):
            self.env.collect_data_reset()
            for i in range(horizon):
                taichi_env.step(action)
                taichi_env.render('human')

            target['mat'].append(taichi_env.simulator.particles_i.mat.to_numpy())
            target['last_grid'].append(taichi_env.simulator.grid.mass.to_numpy()[-2, ...])
            target['last_pos'].append(taichi_env.get_state()['state']['x'])
            target['state'].append(taichi_env.get_state()['state'])
            target['statics'].append(taichi_env.statics.get_states())

        if self.target_file is not None:
            if os.path.exists(self.target_file):
                os.remove(self.target_file)
            pkl.dump(target, open(self.target_file, 'wb'))
            print(f'===> New target generated and dumped to {self.target_file}.')

    def debug(self):
        policy = self.env.demo_policy(user_input=True)
        # action_p = policy.get_actions_p()
        # if action_p is not None:
        #     taichi_env.apply_agent_action_p(action_p)
        for j in range(10):
            self.env.reset()
            total_reward = 0
            for i in range(self.env.horizon):
                if i < self.env.horizon:
                    action = policy.get_action_v(i)
                    # print(action)
                else:
                    action = None
                obs, reward, done, done, info = self.env.step(action)
                cv2.imshow('3d grid sensor', obs["gridsensor3d"].detach().cpu().numpy()[..., 1])
                cv2.waitKey(1)
                total_reward += reward
                print(total_reward)

    def eval(self, model_path):
        policy = torch.load(model_path)[0]
        # policy = torch.load(model_path)["Policy"]
        policy.eval()
        taichi_env = self.env.taichi_env
        for i in range(10):
            obs, _ = self.env.reset()
            for i in range(self.env.horizon):
                if i < self.env.horizon:
                    # action = policy(list(obs.values()))[2][0, :].detach().cpu()
                    action = policy([obs['gridsensor3d'].unsqueeze(0), obs['vector_obs'].unsqueeze(0)])[2][0, :].detach().cpu()
                else:
                    action = None
                obs, _, _, _, _ = self.env.step(action)
                if not is_on_server():
                    if hasattr(taichi_env.loss, 'tgt_particles_x'):
                        taichi_env.render('human', taichi_env.loss.tgt_particles_x)
                    else:
                        taichi_env.render('human')



def record_target(env, path=None, user_input=False):
    env.reset()

    recorder = Recorder(env)
    recorder.record(user_input)

def replay_target(env):
    env.reset()

    recorder = Recorder(env)
    recorder.replay_target()

def replay_policy(env, path=None):
    env.reset()

    recorder = Recorder(env)
    recorder.replay_policy(path)

def record_target_grid(env):
    recorder = Recorder(env)
    recorder.record_stochastic_data(horizon=20, sample_num=100)

def debug(env):
    env.reset()
    recorder = Recorder(env)
    recorder.debug()

def eval(env, model_path=None):
    recorder = Recorder(env)
    recorder.eval(model_path)
