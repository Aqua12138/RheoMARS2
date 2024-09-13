import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict
from fluidlab.configs.macros import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
import fluidlab.utils.misc as misc_utils
from fluidlab.fluidengine import losses
from fluidlab.utils.misc import *
from fluidlab.utils import misc
import copy

class FluidEnv(gym.Env):
    '''
    Base env class.
    '''    
    def __init__(self, loss=True, loss_cfg=None, seed=None, renderer_type='GGUI', perc_type="physics", horizon=1280):
        if seed is not None:
            self.seed(seed)

        self.horizon               = horizon
        self._n_obs_ptcls_per_body = 200

        self.action_range          = np.array([-1.0, 1.0])
        self.renderer_type         = renderer_type
        self.perc_type             = perc_type

        self.target_file           = get_tgt_path(loss_cfg.get('target_file', None))
        self.loss_type             = loss_cfg.get('loss_type', "default")
        self.Loss                  = getattr(losses, loss_cfg.name, None)
        self.weight = dict(loss_cfg.weight)
        self.loss                  = loss

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=200,
            gravity=(0.0, -9.8, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def seed(self, seed):
        misc_utils.set_random_seed(seed)

    def build_env(self):
        self.setup_agent()
        self.setup_statics()
        self.setup_bodies()
        self.setup_smoke_field()
        self.setup_boundary()
        if not misc_utils.is_on_server():
            self.setup_renderer()
        if self.loss:
            self.setup_loss()
            
        self.taichi_env.build()
        self._init_state = copy.deepcopy(self.taichi_env.get_state())
        self._current_state = self.taichi_env.get_state()
        self.grad_enabled = False
        print(f'===>  {type(self).__name__} built successfully.')

    def setup_agent(self):
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        # add static mesh-based objects in the scene
        pass

    def setup_bodies(self):
        # add fluid/object bodies
        self.taichi_env.add_body(
            type='cube',
            lower=(0.2, 0.2, 0.2),
            upper=(0.4, 0.4, 0.4),
            material=WATER,
        )
        self.taichi_env.add_body(
            type='ball',
            center=(0.6, 0.3, 0.6),
            radius=0.1,
            material=WATER,
        )

    def setup_smoke_field(self):
        pass

    def setup_boundary(self):
        pass

    def setup_renderer(self):
        self.taichi_env.setup_renderer()

    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=self.Loss,
            target_file=self.target_file,
            weights=self.weight,
            loss_type=self.loss_type,
        )

    def gym_misc(self):
        obs = self._get_obs()
        self.set_observation_space(obs)
        self.action_space = Box(DTYPE_NP(self.action_range[0]), DTYPE_NP(self.action_range[1]), (self.taichi_env.agent.action_dim,), dtype=DTYPE_NP)
        # self.observation_space = Box(DTYPE_NP(0), DTYPE_NP(1), (180, 90, 2), dtype=DTYPE_NP)

    def set_observation_space(self, obs):
        """根据返回的obs类型设置观察空间"""
        if isinstance(obs, np.ndarray):
            self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), shape=obs.shape, dtype=DTYPE_NP)
        elif isinstance(obs, torch.Tensor):
            self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), shape=obs.shape, dtype=DTYPE_NP)
        elif isinstance(obs, dict):
            spaces = {}
            for key, value in obs.items():
                spaces[key] = Box(DTYPE_NP(0), DTYPE_NP(1), shape=value.shape, dtype=DTYPE_NP)
            self.observation_space = Dict(spaces)
        else:
            raise TypeError("Unsupported observation type")
    def reset(self):
        self.taichi_env.set_state(self._init_state['state'], grad_enabled=self.grad_enabled, t=0, f_global=0)
        info = {}
        return self._get_obs(), info

    # //------------------ shac -----------------
    def reset_grad(self):
        self._current_state["grad_enabled"] = True
        self.taichi_env.set_state(**self._current_state)
        self.taichi_env.reset_grad()
        return self._get_obs()

    def save_state(self):
        self._current_state = self.taichi_env.get_state()

    def compute_actor_loss(self):
        self.taichi_env.loss.compute_actor_loss()

    def compute_actor_loss_grad(self):
        self.taichi_env.loss.compute_actor_loss_grad()

    def set_next_state_grad(self, grad):
        self.taichi_env.set_next_state_grad(grad)

    def get_action_grad(self, n, s):
        # print(self.agent.get_grad(m, n))
        return self.agent.get_grad(n, s)
    # ------------------ shac -----------------//

    def _get_obs(self):
        if self.perc_type == "physics":
            state = self.taichi_env.get_state_RL()
            obs   = []

            if 'x' in state:
                for body_id in range(self.taichi_env.particles['bodies']['n']):
                    body_n_particles  = self.taichi_env.particles['bodies']['n_particles'][body_id]
                    body_particle_ids = self.taichi_env.particles['bodies']['particle_ids'][body_id]

                    step_size = max(1, body_n_particles // self._n_obs_ptcls_per_body)
                    body_x    = state['x'][body_particle_ids][::step_size]
                    body_v    = state['v'][body_particle_ids][::step_size]
                    body_used = state['used'][body_particle_ids][::step_size]

                    obs.append(body_x.flatten())
                    obs.append(body_v.flatten())
                    obs.append(body_used.flatten())

            if 'agent' in state:
                obs += state['agent']

            if 'smoke_field' in state:
                obs.append(state['smoke_field']['v'][::10, 60:68, ::10].flatten())
                obs.append(state['smoke_field']['q'][::10, 60:68, ::10].flatten())

            obs = np.concatenate(obs)
        elif self.perc_type == "sensor":
            obs = {}
            for sensor in self.agent.sensors:
                obs[sensor.name] = sensor.get_obs()
            state = self.taichi_env.get_state_RL()
            obs['vector_obs'] = torch.tensor(state['agent'])[0]
        else:
            assert False
        return obs

    def _get_reward(self):
        loss_info = self.taichi_env.get_step_loss()
        return loss_info['reward']

    def step(self, action):
        action *= self.action_range[1]
        action = action.clip(self.action_range[0], self.action_range[1])

        # action = np.array([action[0],
        #                        action[1],
        #                        action[2],
        #                        action[3] * 3,
        #                        action[4] * 3,
        #                        action[5] * 3])
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
        self.render()
        return obs, reward, done, done, info

    def step_grad(self, action):
        action *= self.action_range[1]
        action = action.clip(self.action_range[0], self.action_range[1])
        self.taichi_env.step_grad(action)

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
        return self.taichi_env.render(mode)

    @property
    def t(self):
        return self.taichi_env.t
