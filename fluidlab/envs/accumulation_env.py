import os
import gymnasium as gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine import losses
import pickle as pkl
import copy

class AccumulationEnv(FluidEnv):
    def __init__(self, loss=True, loss_cfg=None, seed=None, renderer_type='GGUI', perc_type="physics"):
        super().__init__(loss, loss_cfg, seed, renderer_type, perc_type, horizon=640)
        self.action_range = np.array([-0.007, 0.007])
        self.rheo_pos = np.array([0.5, 0.32, 0.5])

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_accumulation.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.0, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
            scale=(0.15, 1, 0.15),
            material=BOTTLE,
            has_dynamics=False,
        )

        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.5, 0.0, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=CONE,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.45, 0.30, 0.45),
            upper=(0.55, 0.34, 0.55),
            material=VISCOUS_DEMO,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        if self.renderer_type == 'GGUI':
            self.taichi_env.setup_renderer(
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )
        elif self.renderer_type == 'GL':
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(3.5, 15.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=20,
            )
        else:
            raise NotImplementedError

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
        return self.taichi_env.render(mode, self.taichi_env.loss.tgt_particles_x)

    def demo_policy(self, user_input=False):
        if not user_input:
            raise NotImplementedError

        init_p = np.array([0.5, 0.5, 0.5])
        return KeyboardPolicy_vxy_wz(init_p, v_lin=1, v_ang=1)

    def step(self, action):
        action *= self.action_range[1]
        action = action.clip(self.action_range[0], self.action_range[1])

        action = np.array([action[0],
                               action[1],
                               action[2],
                               action[3] * 0,
                               action[4] * 3,
                               action[5] * 0])
        self.taichi_env.step(action)

        obs    = self._get_obs()
        reward = self._get_reward()

        assert self.t <= self.horizon
        if self.t == self.horizon:
            done = True
        else:
            done = False

        if np.isnan(reward):
            reward = -1000
            done = True

        info = dict()
        # self.render()
        return obs, reward, done, done, info

    def step_grad(self, action):
        action *= self.action_range[1]
        action = action.clip(self.action_range[0], self.action_range[1])

        action = np.array([action[0],
                               action[1],
                               action[2],
                               action[3] * 0,
                               action[4] * 3,
                               action[5] * 0])
        self.taichi_env.step_grad(action)
    def reset(self):
        # Generate the first random number
        rheo_lower = (0.2, 0.1, 0.6)
        rheo_upper = (0.8, 0.1, 0.8)
        random_pos = np.random.uniform(rheo_lower, rheo_upper)

        rheo_pos = self.rheo_pos
        delta_pos = random_pos - rheo_pos
        self.rheo_pos += delta_pos
        self._init_state['state']['x'] += delta_pos

        agent_lower = (0.2, 0.13, 0.2)
        agent_upper = (0.8, 0.5, 0.8)
        random_pos = np.random.uniform(agent_lower, agent_upper)
        self._init_state['state']['agent'][0][0:3] = random_pos


        target_num = np.random.randint(0, 100)
        target_pos = pkl.load(open(self.target_file, 'rb'))['statics'][target_num][0]
        self.taichi_env.statics.statics[0].set_pos(
            position=np.array(target_pos),
            quatation=np.array([[1, 0, 0, 0]]))

        # set_target
        self.taichi_env.loss.update_target(target_num)

        # random mu
        # mu = np.random.uniform(0, 100)
        # self.taichi_env.simulator.update_mu(mu)

        self.taichi_env.set_state(self._init_state['state'], grad_enabled=self.grad_enabled, t=0, f_global=0)
        self.taichi_env.reset_grad()
        info = {}
        return self._get_obs(), info

    def collect_data_reset(self):
        lower = (0.2, 0.32, 0.25)
        upper = (0.8, 0.32, 0.25)
        random_pos = np.random.uniform(lower, upper)

        rheo_pos = self.rheo_pos
        delta_pos = random_pos - rheo_pos

        self.rheo_pos += delta_pos
        self._init_state['state']['x'] += delta_pos

        self.taichi_env.statics.statics[0].set_pos(
            position=np.array([[random_pos[0], random_pos[1]-0.3, random_pos[2]]]),
            quatation=np.array([[1, 0, 0, 0]]))

        self.taichi_env.set_state(self._init_state['state'], grad_enabled=self.grad_enabled, t=0, f_global=0)







