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
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb
from tianshou.data import Collector, ReplayBuffer
from torch.distributions import Distribution, Independent, Normal
import fluidlab.envs
device = "cuda" if torch.cuda.is_available() else "cpu"
from fluidlab.optimizer.network.encoder import CustomNet, Critic
import torch.nn.functional as F
import traceback
from tianshou.data import Batch

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

# 自定义Actor网络
class CustomActor(nn.Module):
    def __init__(self, model, action_shape):
        super().__init__()
        self.model = model  # 已有的模型
        self.log_std = nn.Parameter(torch.zeros(action_shape))
    def forward(self, obs, state=None, info={}):
        # 将obs转换为obs_list
        obs_list = [obs.gridsensor2d, obs.gridsensor3d, obs.vector_obs]
        mean = self.model(obs_list)[4] # 获取动作的logits
        log_std = self.log_std.expand_as(mean)  # 扩展 log_std
        return (mean, log_std), state  # 返回均值和对数标准差

# 自定义Critic网络
class CustomCritic(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 已有的模型

    def forward(self, obs, **kwargs):
        obs_list =  [[obs.gridsensor2d, obs.gridsensor3d, obs.vector_obs]]
        value = self.model.critic_pass(obs_list)[0]['extrinsic']
        return value.squeeze(-1)  # 确保输出形状正确


class PPO_trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        train_envs = DummyVectorEnv([lambda: gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=True, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type, perc_type="sensor") for _ in range(self.cfg.params.config.num_envs)])
        assert isinstance(train_envs.observation_space[0], gym.spaces.Dict)
        assert isinstance(train_envs.action_space[0], gym.spaces.Box)

        state_shape = []
        for key, space in train_envs.observation_space[0].items():
            state_shape.append(space.shape)
        action_space = train_envs.action_space[0]
        self.action_shape = action_space.shape
        self.build_actor_critic(model_path=args.pre_train_model, intial_model=args.intial_model)

        self.policy = PPOPolicy(
            self.actor,
            self.critic,
            self.optimizer,
            dist_fn=self.dist_fn,
            action_space=action_space,
            vf_coef=0.5,
            ent_coef=0.01,
            max_grad_norm=0.5,
            eps_clip=0.2,
            value_clip=True,
            gae_lambda=0.95,
            discount_factor=0.99,
            reward_normalization=True,
            action_bound_method='clip',
            advantage_normalization=True,
            recompute_advantage=False
        )

        # 创建缓冲区
        train_buffer = VectorReplayBuffer(
            total_size=self.cfg.params.config.ReplayBufferSize,
            buffer_num=self.cfg.params.config.num_envs
        )

        # 创建收集器
        self.train_collector = Collector(self.policy, train_envs, train_buffer)
        self.test_collector = Collector(self.policy, train_envs)

    def solver(self):
        # 开始训练
        try:
            result = onpolicy_trainer(
                policy=self.policy,
                train_collector=self.train_collector,
                test_collector=self.test_collector,
                max_epoch=self.cfg.params.config.max_epochs,
                step_per_epoch=self.cfg.params.config.step_per_epochs,
                step_per_collect=self.cfg.params.config.step_per_collect,  # 这里是 2000
                repeat_per_collect=self.cfg.params.config.repeat_per_collect,
                episode_per_test=self.cfg.params.config.episode_per_test,
                batch_size=self.cfg.params.config.batch_size,
            )
        except Exception as e:
            print("An exception occurred:")
            traceback.print_exc()

    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 将偏置初始化为0

    def build_actor_critic(self, model_path, intial_model=False):
        actor_model = torch.load(model_path)["Policy"]
        critic_model = torch.load(model_path)['Optimizer:critic']
        if intial_model:
            # 对比实验
            self.initialize_weights(actor_model)
            self.initialize_weights(critic_model)

        self.actor = CustomActor(actor_model, self.action_shape)
        self.critic = CustomCritic(critic_model)

        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.cfg.params.config.lr, betas=self.cfg.params.config.betas)

    # 定义动作分布函数
    def dist_fn(self, mean, log_std):
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

class SHAC_trainer:
    def __init__(self, cfg, args):
        train_envs = MyDummyVectorEnv([lambda: gym.make(cfg.params.env.name, seed=cfg.params.env.seed+i, loss=True, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type, perc_type="sensor") for i in range(cfg.params.config.num_actors)])

        assert isinstance(train_envs.observation_space[0], gym.spaces.Dict)
        assert isinstance(train_envs.action_space[0], gym.spaces.Box)  # for mypy

        state_shape = []
        for key, space in train_envs.observation_space[0].items():
            state_shape.append(space.shape)
        action_shape = train_envs.action_space[0].shape

        self.build_actor_critic(model_path=args.pre_train_model, intial_model=args.intial_model)
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

    def build_actor_critic(self, model_path, intial_model=False):
        self.actor = torch.load(model_path)["Policy"]
        self.critic = torch.load(model_path)['Optimizer:critic']

        if intial_model:
            # 对比实验
            self.initialize_weights(self.actor)
            self.initialize_weights(self.critic)



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