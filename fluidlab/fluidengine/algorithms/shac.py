# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from multiprocessing.sharedctypes import Value
import sys, os

from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time
import numpy as np
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from collections import OrderedDict
import pycuda.autoinit

from fluidlab.fluidengine.algorithms.models import actor
from fluidlab.fluidengine.algorithms.models import critic
from fluidlab.fluidengine.algorithms.utils.common import *
import fluidlab.fluidengine.algorithms.utils.torch_utils as tu
from fluidlab.fluidengine.algorithms.utils.running_mean_std import RunningMeanStd
from fluidlab.fluidengine.algorithms.utils.dataset import CriticDataset
from fluidlab.fluidengine.algorithms.utils.time_report import TimeReport
from fluidlab.fluidengine.algorithms.utils.average_meter import AverageMeter
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import gym

import json

VecEnvTensorObs = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]
VecEnvStepReturn = Tuple[VecEnvTensorObs, torch.Tensor, torch.Tensor, List[Dict]]
VecEnvIndices = Union[None, int, Iterable[int]]


def merge_dict_array(dict_array, device='cuda:0'):
    keys = dict_array[0].keys()
    merged_dict = {}
    for key in keys:
        stacked_tensors = [d[key].to(device) for d in dict_array]
        merged_dict[key] = torch.stack(stacked_tensors, dim=0)
    return merged_dict


class SHACPolicy:
    def __init__(self,
                 envs,
                 cfg,
                 args,
                 actor,
                 critic,
                 dist_fn,
                 device="cuda:0"):

        self.device = device
        self.cfg = cfg
        self.args = args

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # init
        self._setup_hyperparameters()
        self._initialize_environment(envs)
        self._initialize_networks(actor, critic, dist_fn)
        self._setup_optimizers()
        self._initialize_buffers()
        self._setup_logging()
        self._save_init_policy()

        # timer
        self.time_report = TimeReport()

    # ----------------- initialize -----------------
    def _initialize_environment(self, envs):
        self.envs = envs
        self.num_envs = self.envs.env_num
        self.num_obs = self.envs.observation_space[0]
        self.num_actions = self.envs.action_space[0].shape
        self.max_episode_length = self.envs.get_env_attr("horizon")[0]
        self.batch_size = self.num_envs * self.steps_num // self.num_batch

    def _setup_hyperparameters(self):
        # Extract hyperparameters from config and set default values
        self.critic_method = self.cfg.params.config.get('critic_method', 'one-step')
        if self.critic_method == 'td-lambda':
            self.lam = self.cfg.params.config.get('lam', 0.95)

        self.steps_num = self.cfg.params.config.get('steps_num', 32)
        self.max_epochs = self.cfg.params.config.get('max_epochs', 100)
        self.actor_lr = float(self.cfg.params.config.get('actor_learning_rate', 1e-3))
        self.critic_lr = float(self.cfg.params.config.get('critic_learning_rate', 1e-3))
        self.lr_schedule = self.cfg.params.config.get('lr_schedule', "constant")
        self.target_critic_alpha = self.cfg.params.config.get('target_critic_alpha', 0.2)
        self.critic_iterations = self.cfg.params.config.get('critic_iterations', 16)
        self.num_batch = self.cfg.params.config.get('num_batch', 4)
        self.truncate_grad = self.cfg.params.config.get('truncate_grad', True)
        self.grad_norm = float(self.cfg.params.config.get('grad_norm', 1.0))
        self.gamma = float(self.cfg.params.config.get('gamma', 0.99))
        self.betas = self.cfg.params.config.get('betas', [0.7, 0.95])
        self.save_interval = float(self.cfg.params.config.get('save_interval', 500))
        self.name = self.cfg.params.env.get('name', "test")

    def _initialize_networks(self, actor, critic, dist_fn):
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.dist_fn = dist_fn

    def _setup_optimizers(self):
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), betas=self.betas, lr=self.actor_lr, weight_decay=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas=self.betas, lr=self.critic_lr)

    def _initialize_buffers(self):
        # Initialize buffers for observations, rewards, etc.
        # replay buffer
        self.obs_buf_grid3D = torch.zeros((self.steps_num, self.num_envs, *self.num_obs["gridsensor3d"].shape),
                                          dtype=torch.float32, device=self.device)
        self.obs_buf_vector = torch.zeros((self.steps_num, self.num_envs, *self.num_obs["vector_obs"].shape),
                                          dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype=torch.float32, device=self.device)

        # loss variables
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_length = 0
        self.best_policy_reward = -100000
        self.episode_reward_mean = -100000
        self._episode_reward_mean = -100000
        self.lr_patience = 0
        self.value_loss = np.inf

    def _setup_logging(self):
        base_dir = os.path.dirname(os.path.dirname(project_dir))

        self.log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.log_dir, self.args.exp_name + '/log'))

        # Save configuration
        save_cfg = copy.deepcopy(self.cfg)
        with open(os.path.join(self.log_dir, 'cfg.yaml'), 'w') as f:
            yaml.dump(save_cfg, f)

    def _save_init_policy(self):
        # Save initial policy weights
        self.save('init_policy')

    # ----------------- train -----------------

    def compute_actor_loss(self):
        # critic data
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)  # debug & critic train
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype=torch.float32,
                                  device=self.device)  # critic train
        actor_loss = torch.tensor(0., dtype=torch.float32, device=self.device)  # debug
        actions = torch.zeros((self.steps_num, self.num_envs, *self.num_actions), dtype=torch.float32,
                              device=self.device)

        if self.episode_length == self.max_episode_length or self.episode_length == 0:
            # reset enviroment
            self.episode_length = 0
            self.envs.set_env_attr("grad_enabled", True)
            obs, info = self.envs.reset()
        else:
            # reset grad
            obs = self.envs.reset_grad()

        obs = merge_dict_array(obs, device=self.device)

        # collect data for critic training and simulation forward
        for i in range(self.steps_num):
            self.episode_length += 1
            with torch.no_grad():
                self.obs_buf_grid3D[i] = obs['gridsensor3d'].clone()
                self.obs_buf_vector[i] = obs['vector_obs'].clone()

            # simple actions
            # logits, hidden = self.actor(obs)
            # dist = self.dist_fn(*logits)
            # action = dist.rsample() # contain grad_fn
            grid_sensor = obs["gridsensor3d"]
            vector_obs = obs["vector_obs"]
            action = self.actor([grid_sensor, vector_obs])[4] # 2:random 4 detemistict
            actions[i] = action
            if torch.isnan(action).any():
                print("The tensor contains NaN values")
            with torch.no_grad():
                actions_clone = action.clone().detach().cpu().numpy()
            obs, reward, done, done, info = self.envs.step(actions_clone)
            obs = merge_dict_array(obs, device=self.device)
            reward = torch.from_numpy(reward).to(self.device)
            done = torch.from_numpy(done).to(self.device)

            obs['vector_obs'].requires_grad_(True)
            obs['gridsensor3d'].requires_grad_(True)
            obs_list = [[obs['gridsensor3d'], obs['vector_obs']]]
            next_values[i + 1] = self.target_critic.critic_pass(obs_list)[0]['extrinsic']
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError

            if i == self.steps_num - 1:
                actor_loss = actor_loss + (- self.gamma * gamma * next_values[i + 1, :]).sum()
                actor_loss.backward()
                # 传入taichi的obs对应变量中
                state_grad = {}
                state_grad['vector_obs'] = obs['vector_obs'].grad.cpu().numpy()
                state_grad['grid_sensor3d'] = obs['gridsensor3d'].grad.cpu().numpy()
                # 按照环境进行分配
                state_grads = [{key: value[i] for key, value in state_grad.items()} for i in range(self.num_envs)]

            # collect data for critic training
            gamma = gamma * self.gamma
            self.rew_buf[i] = reward
            if i < self.steps_num - 1:
                self.done_mask[i] = done
            else:
                self.done_mask[i, :] = 1.
            with torch.no_grad():
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_reward += reward
                if self.episode_length == self.max_episode_length:
                    self.episode_reward_mean = torch.sum(self.episode_reward) / self.num_envs
                    for i in range(self.num_envs):
                        self.episode_reward[i] = 0.

        self.envs.compute_actor_loss()
        # save the sim state
        self.envs.save_state()

        # backward
        self.envs.set_next_state_grad(state_grads)  # for critic grad
        self.envs.compute_actor_loss_grad()

        for i in range(self.steps_num - 1, -1, -1):
            with torch.no_grad():
                actions_clone = actions[i].clone().detach().cpu().numpy()
            self.envs.step_grad(actions_clone)

        action_grads = self.envs.get_action_grad([self.episode_length, self.episode_length - self.steps_num])[:, 0:-1,
                       :]


        # 定义一个阈值，设为 1
        # max_norm = 0.001
        #
        # # 计算每个梯度向量的 2-norm
        # total_norm = np.linalg.norm(action_grads, axis=-1, keepdims=True)
        #
        # # 计算裁减系数
        # clip_coef = np.where(total_norm > max_norm, max_norm / (total_norm + 1e-6), 1.0)
        #
        # # 裁减梯度
        # clipped_action_grads = action_grads * clip_coef

        if torch.isnan(torch.tensor(action_grads, dtype=torch.float32).to(self.device).permute(1, 0, 2)).any():
            print("The tensor contains NaN values")
            action_grads = np.zeros_like(action_grads)
        return actions, torch.tensor(action_grads, dtype=torch.float32).to(self.device).permute(1, 0, 2)

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                            self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (
                                1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + \
                     self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        obs_list = [[batch_sample['obs']['gridsensor3d'], batch_sample['obs']['vector_obs']]]
        predicted_values = self.critic.critic_pass(obs_list)[0]['extrinsic']
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def actor_closure(self):
        # print("before:", tu.grad_norm(self.actor.parameters()))
        self.actor_optimizer.zero_grad()
        # print("after:", tu.grad_norm(self.actor.parameters()))
        self.time_report.start_timer("compute actor loss")

        self.time_report.start_timer("forward and backward simulation")
        actions, action_grads = self.compute_actor_loss()
        actions.backward(action_grads)
        self.time_report.end_timer("forward and backward simulation")

        with torch.no_grad():
            self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
            if self.truncate_grad:
                clip_grad_norm_(self.actor.parameters(), self.grad_norm)
            self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

            # sanity check
            # if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000000.:
            #     print('NaN gradient')
            #     print(self.grad_norm_before_clip)
            #     raise ValueError

        self.time_report.end_timer("compute actor loss")

    def setup_time_report(self):
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward and backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

    def update_learning_rate(self, lr_schedule, epoch):
        # learning rate schedule
        if self.lr_schedule == "linear":
            actor_lr = (1e-6 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
            self.lr = actor_lr
            critic_lr = (1e-6 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr
        elif self.lr_schedule == "ReduceLROnPlateau":
            if self._episode_reward_mean > self.episode_reward_mean:
                self.lr_patience += 1
                self._episode_reward_mean = self.episode_reward_mean
                if self.lr_schedule >= 5:
                    actor_lr = self.actor_lr * 0.5
                    for param_group in self.actor_optimizer.param_groups:
                        param_group['lr'] = actor_lr
                    self.lr = actor_lr
                    critic_lr = self.critic_lr * 0.5
                    for param_group in self.critic_optimizer.param_groups:
                        param_group['lr'] = critic_lr
                    self.lr_patience = 0
        else:
            self.lr = self.actor_lr

    def train_actor(self):
        self.time_report.start_timer("actor training")
        self.actor_closure()
        self.actor_optimizer.step()
        self.time_report.end_timer("actor training")

    def train_critic(self, dataset):
        self.time_report.start_timer("critic training")
        self.value_loss = 0.
        for j in range(self.critic_iterations):
            total_critic_loss = 0.
            batch_cnt = 0
            for i in range(len(dataset)):
                batch_sample = dataset[i]
                self.critic_optimizer.zero_grad()
                training_critic_loss = self.compute_critic_loss(batch_sample)
                training_critic_loss.backward()
                for params in self.critic.parameters():
                    if params.grad is not None:
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)
                self.grad_norm_before_clip_critic = tu.grad_norm(self.critic.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                self.critic_optimizer.step()

                total_critic_loss += training_critic_loss
                batch_cnt += 1

            self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()

        self.time_report.end_timer("critic training")

    def get_dataset(self):
        self.time_report.start_timer("prepare critic dataset")
        with torch.no_grad():
            self.compute_target_values()
            dataset = CriticDataset(self.batch_size,
                                    {"gridsensor3d": self.obs_buf_grid3D, "vector_obs": self.obs_buf_vector},
                                    self.target_values, drop_last=False)
        self.time_report.end_timer("prepare critic dataset")
        return dataset

    def logging(self, epoch):
        time_elapse = time.time() - self.start_time

        if self.episode_reward_mean > self.best_policy_reward:
            self.save()
            self.best_policy_reward = self.episode_reward_mean

        self.writer.add_scalar('lr/epoch', self.lr, epoch)
        self.writer.add_scalar('rewards/time', self.episode_reward_mean, time_elapse)
        self.writer.add_scalar('rewards/epoch', self.episode_reward_mean, epoch)

        self.writer.flush()

        if self.save_interval > 0 and (epoch % self.save_interval == 0):
            self.save(self.name + "policy_epoch{}_reward{:.3f}".format(epoch, self.episode_reward_mean))

    def update_target_critic(self):
        with torch.no_grad():
            alpha = self.target_critic_alpha
            for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                param_targ.data.mul_(alpha)
                param_targ.data.add_((1. - alpha) * param.data)

    def learn(self):
        self.start_time = time.time()

        # setup timers
        self.setup_time_report()
        self.time_report.start_timer("algorithm")

        # initializations
        self.episode_length = 0
        self.envs.set_env_attr("grad_enabled", True)
        self.envs.reset()  # 每次调用reset的时候确定本次是否需要梯度

        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        # main training process
        with tqdm(total=self.max_epochs, desc='Training Progress', unit='epoch', ncols=75, position=0,
                  file=sys.stdout) as pbar_outer:
            for epoch in range(self.max_epochs):
                epoch_start_time = time.time()
                self.update_learning_rate(self.lr_schedule, epoch)
                with tqdm(total=self.max_episode_length // self.steps_num, leave=False, desc='Epoch Progress', ncols=75,
                          position=1, file=sys.stdout, miniters=1) as pbar_inner:
                    for i in range(self.max_episode_length // self.steps_num):
                        time_start_epoch = time.time()
                        # train actor
                        self.train_actor()
                        # prepare dataset of critic training
                        dataset = self.get_dataset()
                        # train critic
                        self.train_critic(dataset)
                        time_end_epoch = time.time()

                        fps = self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch)
                        pbar_inner.write(
                            f'Batch {i + 1}/{self.max_episode_length // self.steps_num}: FPS={fps:.2f}, Value Loss={self.value_loss:.4f}, Grad Norm Before Clip={self.grad_norm_before_clip:.4f}, Reward={self.episode_reward_mean:.4f}')
                        pbar_inner.update(1)

                        self.writer.add_scalar('value_loss/step', self.value_loss, self.step_count)
                        self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                eta_total = epoch_duration * (self.max_epochs - epoch - 1)
                pbar_outer.write(
                    f'Epoch {epoch + 1}/{self.max_epochs}: Duration={epoch_duration:.2f} sec, ETA={eta_total:.2f} sec, Episode Reward Mean={self.episode_reward_mean:.2f}, Best Policy Reward={self.best_policy_reward:.2f}')
                pbar_outer.update(1)

                self.logging(epoch)


        self.time_report.end_timer("algorithm")
        self.time_report.report()

        self.save('final_policy')
        self.close()

    def save(self, filename=None):
        if filename is None:
            filename = 'best_policy'
        torch.save([self.actor, self.critic, self.target_critic], os.path.join(self.log_dir, self.args.exp_name + '/' + "{}.pt".format(filename)))

    def close(self):
        self.writer.close()



