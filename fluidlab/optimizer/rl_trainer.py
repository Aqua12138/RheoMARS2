from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3 import PPO
from fluidlab.fluidengine.algorithms.shac import SHACPolicy
import yaml
import gym as old_gym
import gymnasium as gym
import random
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb
from torch.distributions import Distribution, Independent, Normal
import fluidlab.envs
device = "cuda" if torch.cuda.is_available() else "cpu"
from fluidlab.optimizer.network.encoder import CustomNet, Critic
import torch.nn.functional as F

class MyDummyVectorEnv(DummyVectorEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)

    def reset_grad(self):
        """Apply gradients to each environment.

        :param gradients: List of gradients, one per environment.
        :return: Tuple containing next_states, rewards, dones, and infos from each environment after applying gradients.
        """
        results = []
        for worker in self.workers:
            if hasattr(worker.env, 'reset_grad'):
                result = worker.env.reset_grad()
            else:
                raise NotImplementedError('Environment does not implement reset_grad method')
            results.append(result)
        return np.array(results)

    def compute_actor_loss(self):
        for worker in self.workers:
            if hasattr(worker.env, 'compute_actor_loss'):
                worker.env.compute_actor_loss()
            else:
                raise NotImplementedError('Environment does not implement compute_actor_loss method')

    def compute_actor_loss_grad(self):
        for worker in self.workers:
            if hasattr(worker.env, 'compute_actor_loss_grad'):
                worker.env.compute_actor_loss_grad()
            else:
                raise NotImplementedError('Environment does not implement compute_actor_loss_grad method')
    def save_state(self):
        for worker in self.workers:
            if hasattr(worker.env, 'save_state'):
                worker.env.save_state()
            else:
                raise NotImplementedError('Environment does not implement save_state method')

    def set_next_state_grad(self, grads):
        for worker, grad in zip(self.workers, grads):
            if hasattr(worker.env, 'set_next_state_grad'):
                worker.env.set_next_state_grad(grad)
            else:
                raise NotImplementedError('Environment does not implement set_next_state_grad method')

    def step_grad(self, actions):
        for worker, action in zip(self.workers, actions):
            if hasattr(worker.env, 'step_grad'):
                worker.env.step_grad(action)
            else:
                raise NotImplementedError('Environment does not implement step_grad method')

    def get_action_grad(self, incides):
        results = []
        for worker in self.workers:
            if hasattr(worker.env, 'get_action_grad'):
                result = worker.env.get_action_grad(incides[0], incides[1])
            else:
                raise NotImplementedError('Environment does not implement get_action_grad method')
            results.append(result)
        return np.array(results)
class PPO_trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        train_envs = DummyVectorEnv([lambda: gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=True, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type, perc_type="sensor") for _ in range(self.cfg.params.config.num_envs)])
        assert isinstance(train_envs.observation_space[0], gym.spaces.Dict)
        assert isinstance(train_envs.action_space[0], gym.spaces.Box)

        state_shape = []
        for key, space in train_envs.observation_space[0].items():
            state_shape.append(space.shape)
        action_shape = train_envs.action_space[0].shape

        self.build_actor_critic(state_shape, action_shape, device=cfg.params.config.device)
        self._init_actor_critic()

        self.policy: BasePolicy
        self.policy = PPOPolicy(
            actor=self.actor,
            critic=self.critic,
            optim=self.optim,
            dist_fn=self.dist,
            action_space=train_envs.action_space[0],
            deterministic_eval=self.cfg.params.config.deterministic_eval,
            action_scaling=self.cfg.params.config.action_scaling,
        )

        self.train_collector = Collector(
            policy=self.policy,
            env=train_envs,
            buffer=VectorReplayBuffer(self.cfg.params.config.ReplayBufferSize, len(train_envs)),
        )

        self.test_collector = Collector(policy=self.policy, env=train_envs)
    def solver(self):
        result = OnpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=self.cfg.params.config.max_epochs,
            step_per_epoch=self.cfg.params.config.step_per_epochs,
            repeat_per_collect=self.cfg.params.config.repeat_per_collect,
            episode_per_test=self.cfg.params.config.episode_per_test,
            batch_size=self.cfg.params.config.batch_size,
            step_per_collect=self.cfg.params.config.step_per_collect
        ).run()

    def _init_actor_critic(self):
        torch.nn.init.constant_(self.actor.sigma_param, -0.5)
        for m in self.actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in self.actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    def build_actor_critic(self, state_shape, action_shape, device):
        net_a = CustomNet(
            state_shape,
            hidden_sizes=[256, 128],
            activation=nn.Tanh,
            device=device,
        )

        net_c = CustomNet(
            state_shape,
            hidden_sizes=[128, 128],
            activation=nn.Tanh,
            device=device,
        )

        self.actor = ActorProb(
            net_a,
            action_shape,
            device=device,
            unbounded=True
        ).to(device)

        self.critic = Critic(net_c, device=device).to(device)
        self.actor_critic = ActorCritic(self.actor, self.critic)

        # optimizer of the actor and the critic
        self.optim = torch.optim.Adam(self.actor_critic.parameters(), lr=self.cfg.params.config.lr)

    def dist(self, loc: torch.Tensor, scale: torch.Tensor) -> Distribution:
        return Independent(Normal(loc, scale), 1)


class SHAC_trainer:
    def __init__(self, cfg, args):
        train_envs = MyDummyVectorEnv([lambda: gym.make(cfg.params.env.name, seed=cfg.params.env.seed+i, loss=True, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type, perc_type="sensor") for i in range(cfg.params.config.num_actors)])

        assert isinstance(train_envs.observation_space[0], gym.spaces.Dict)
        assert isinstance(train_envs.action_space[0], gym.spaces.Box)  # for mypy

        state_shape = []
        for key, space in train_envs.observation_space[0].items():
            state_shape.append(space.shape)
        action_shape = train_envs.action_space[0].shape

        self.build_actor_critic(model_path=args.pre_train_model)
        # self._init_actor_critic()

        self.policy = SHACPolicy(
            cfg=cfg,
            args=args,
            envs=train_envs,
            actor=self.actor,
            critic=self.critic,
            dist_fn=self.dist,
            device=cfg.params.config.device)
    def solver(self):
        self.policy.learn()

    def dist(self, loc: torch.Tensor, scale: torch.Tensor) -> Distribution:
        return Independent(Normal(loc, scale), 1)

    def _init_actor_critic(self):
        """
        Initialize the actor and critic networks.

        This function specifically initializes the weights and biases of the linear layers in the actor and critic networks. For the actor network,
        it uses orthogonal initialization for the weights with a gain of sqrt(2), and sets the bias to 0. For the critic network, it uses the same
        initialization method for the weights and bias. Additionally, it scales the weight of the last policy layer in the actor network by a factor of 0.01
        and initializes the sigma parameter to -0.5.

        The purpose of these specific initializations is to improve the stability and performance of the training process.
        """
        # Initialize the weights and biases of the linear layers in the actor network
        for m in self.actor.modules():
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        # Initialize the weights and biases of the linear layers in the critic network
        for m in self.critic.modules():
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)

        # Scale the weight of the last policy layer in the actor network and initialize the bias to 0
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in self.actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

        # Initialize the sigma parameter of the actor network to -0.5
        torch.nn.init.constant_(self.actor.sigma_param, -0.5)

    # 定义一个初始化函数
    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 将偏置初始化为0

    def build_actor_critic(self, model_path):
        # model_path = "/home/zhx/Project/ml-agents/ml-agents/mlagents/trainers/results/debug/Cup/Cup-2700_policy_model.pth"
        # model_path = "/home/zhx/Project/ml-agents/ml-agents/mlagents/trainers/results/pour_3d/Cup/Cup-1499900_policy_model.pth"
        self.actor = torch.load(model_path)["Policy"]
        # self.initialize_weights(self.actor)
        self.critic = torch.load(model_path)['Optimizer:critic']
        # self.initialize_weights(self.critic)

        # net_a = CustomNet(state_shape, hidden_sizes=[256, 128], activation=nn.Tanh, device=device, input_size=256)
        # net_c = CustomNet(state_shape, hidden_sizes=[256, 128], activation=nn.Tanh, device=device, input_size=256)
        # self.actor = ActorProb(net_a, action_shape, device=device, max_action=1).to(device)
        # self.critic = Critic(net_c, device=device).to(device)


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        # 视觉输入编码器
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_vision = nn.Linear(64 * 12 * 23, 128)  # 根据卷积后尺寸调整

        # 向量输入处理
        self.fc_vector = nn.Linear(3, 128)
        self.norm = nn.LayerNorm(128)

        # 输出层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)  # 假设最终输出为单个值

    def forward(self, visual_input, vector_input):
        # 视觉输入处理
        x = F.relu(self.conv1(visual_input.permute(0, 3, 1, 2)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc_vision(x))

        # 向量输入处理
        y = F.relu(self.fc_vector(vector_input))
        y = self.norm(y)

        # 拼接
        combined = torch.cat((x, y), dim=1)

        # 输出层处理
        z = F.relu(self.fc1(combined))
        output = self.fc2(z)

        return output