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

class MixingEnv(FluidEnv):
    def __init__(self, loss=True, loss_cfg=None, seed=None, renderer_type='GGUI', perc_type="physics"):
        super().__init__(loss, loss_cfg, seed, renderer_type, perc_type, horizon=1280)
        self.action_range = np.array([-0.003, 0.003])

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_transporting.yaml'))
        agent_cfg["build_sensor"] = False
        if self.target_file is not None and self.perc_type == "sensor":
            for sensor in agent_cfg.sensors:
                sensor["params"]["target_file"] = self.target_file
            agent_cfg["build_sensor"] = True
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        pass

    def setup_bodies(self):
        # self.taichi_env.add_body(
        #     type='cube',
        #     lower=(0.45, 0.55, 0.15),
        #     upper=(0.55, 0.65, 0.25),
        #     material=WATER,
        # )
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.6, 0.2),
            height=0.1,
            radius=0.1,
            material=WATER,
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
        # Generate the first random number
        target_num = np.random.randint(0, 100)
        # Generate the second random number ensuring it is different from the first
        init_num = target_num
        while init_num == target_num:
            init_num = np.random.randint(0, 100)
        # set_init
        self.init_state = pkl.load(open(self.target_file, 'rb'))['state']
        self._init_state['state'] = self.init_state[init_num]

        # set_target
        self.taichi_env.loss.update_target(target_num)
        for i in range(len(self.agent.sensors)):
            self.agent.sensors[i].update_target(target_num)

        # random mu
        # mu = np.random.uniform(0, 0)
        # self.taichi_env.simulator.update_mu(mu)


        self.taichi_env.set_state(self._init_state['state'], grad_enabled=self.grad_enabled, t=0, f_global=0)
        info = {}
        return self._get_obs(), info

    def collect_data_reset(self):
        lower = (0.2, 0.3, 0.2)
        upper = (0.8, 0.7, 0.8)
        random_pos = np.random.uniform(lower, upper)

        self.new_init_state = self._init_state['state']

        init_agent_pos = self._init_state['state']['agent'][0][0:3]
        delta_pos = random_pos - init_agent_pos

        self._init_state['state']['agent'][0][0:3] += delta_pos
        self._init_state['state']['x'] += delta_pos
        self.taichi_env.set_state(self._init_state['state'], grad_enabled=self.grad_enabled, t=0, f_global=0)







