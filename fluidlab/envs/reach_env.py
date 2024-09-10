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

class ReachEnv(FluidEnv):
    def __init__(self, loss=True, loss_cfg=None, seed=None, renderer_type='GGUI', perc_type="physics"):
        super().__init__(loss, loss_cfg, seed, renderer_type, perc_type, horizon=3840)
        self.action_range = np.array([-0.0007, 0.0007])

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_reach.yaml'))
        agent_cfg["build_sensor"] = False
        if self.target_file is not None and self.perc_type == "sensor":
            for sensor in agent_cfg.sensors:
                sensor["params"]["target_file"] = self.target_file
            agent_cfg["build_sensor"] = True
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='table.obj',
            pos=(0.5, 0.1, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1, 1, 1),
            material=CONE,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.45, 0.45, 0.45),
            upper=(0.55, 0.55, 0.55),
            material=MILK_VIS,
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
                render_particle=True,
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

        return self.taichi_env.render(mode)

    def demo_policy(self, user_input=False):
        if not user_input:
            raise NotImplementedError

        init_p = np.array([0.5, 0.55, 0.2])
        return MousePolicy_vxz(init_p)

    # ----- ADAM Policy -------
    def trainable_policy(self, optim_cfg, init_range):
        return GatheringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon, self.action_range)
    def reset(self):
        lower = (0.2, 0.4, 0.6)
        upper = (0.8, 0.4, 0.8)
        random_pos = np.random.uniform(lower, upper)

        agent_lower = (0.3, 0.7, 0.7)
        agent_upper = (0.7, 0.7, 0.7)
        agent_pos = np.random.uniform(agent_lower, agent_upper)
        init_state = copy.deepcopy(self._init_state)
        init_particle_pos = np.array([0.5, 0.5, 0.5])
        delta_pos = random_pos - init_particle_pos

        init_state['state']['x'] += delta_pos
        init_state['state']['agent'] = [[agent_pos[0], agent_pos[1], agent_pos[2], 1, 0, 0, 0]]

        self.taichi_env.set_state(init_state["state"], grad_enabled=self.grad_enabled, t=0, f_global=0)
        self.taichi_env.loss.update_target(0)
        info = {}
        return self._get_obs(), info

    def collect_data_reset(self):
        lower = (0.5, 0.4, 0.2)
        upper = (0.5, 0.4, 0.2)
        random_pos = np.random.uniform(lower, upper)
        self.new_init_state = self._init_state['state']

        init_particle_pos = np.array([0.5, 0.5, 0.5])
        delta_pos = random_pos - init_particle_pos

        # self._init_state['state']['agent'][0][0:3] += delta_pos
        self.init_state['state']['x'] += delta_pos
        self.taichi_env.set_state(self._init_state['state'], grad_enabled=self.grad_enabled, t=0, f_global=0)







