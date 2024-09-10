# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from fluidlab.fluidengine.algorithms.models import model_utils
from fluidlab.fluidengine.algorithms.models.mlagent_torch import encoders
from fluidlab.fluidengine.algorithms.models.mlagent_torch import layers
from fluidlab.fluidengine.algorithms.models.mlagent_torch import distributions
class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.actor = nn.Sequential(*modules).to(device)
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic = False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]
        self.simple_visual_encoder = encoders.SimpleVisualEncoder(height=90,
                                                                  width=180,
                                                                  initial_channels=2,
                                                                  output_size=256).to(device)  # 视觉输入维度

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))
            
        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mu_net)
        print(self.logstd)
    
    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False):
        cat_obs = torch.cat((self.simple_visual_encoder(obs["gridsensor3"]), obs["vector_obs"]), dim=1)
        mu = self.mu_net(cat_obs)
        # print(mu)
        if deterministic:
            return mu
        else:
            std = self.logstd.exp() # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = Normal(mu, std)
            sample = dist.rsample()
            # sample = mu + torch.randn_like(mu) * std
            return sample
    
    def forward_with_dist(self, obs, deterministic = False):
        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std
        
    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)

class ActorStochasticPPO(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticPPO, self).__init__()
        self.h_size=256
        self.device = device
        self.simple_visual_encoder = encoders.SimpleVisualEncoder(height=obs_dim['gridsensor3'][0], width=obs_dim['gridsensor3'][1], initial_channels=obs_dim['gridsensor3'][2], output_size=256).to(device) # 视觉输入维度
        self.normalizer = encoders.Normalizer(vec_obs_size=obs_dim['vector_obs'][0]).to(device) # vector obs维度
        self.linear_encoder = layers.LinearEncoder(input_size=256+obs_dim['vector_obs'][0], num_layers=3, hidden_size=self.h_size, kernel_gain=(0.125 / self.h_size) ** 0.5).to(device)
        self.gaussian_distribution = distributions.GaussianDistribution(hidden_size=256, num_outputs=action_dim[0]).to(device) # num_outputs输出动作维度 # 返回的是action的概率分布,还需要判断
    def get_logstd(self):
        # 确认一下
        return self.logstd
    def forward(self, obs, deterministic = False):
        hide = torch.cat((self.simple_visual_encoder(obs["gridsensor3"]), self.normalizer(obs["vector_obs"])), dim=1)
        hide = self.linear_encoder(hide)
        dist = self.gaussian_distribution(hide)
        if deterministic:
            return dist.deterministic_sample()
        else:
            return dist.sample()

    def forward_with_dist(self, obs, deterministic = False):
        hide = torch.cat((self.simple_visual_encoder(obs["gridsensor3"]), self.normalizer(obs["vector_obs"])), dim=1)
        hide = self.linear_encoder(hide)
        dist = self.gaussian_distribution(hide)

        mu = dist.deterministic_sample()
        std = dist.std
        if deterministic:
            return mu, mu, std
        else:
            return dist.sample(), mu, std

    def evaluate_actions_log_probs(self, obs, actions):
        hide = torch.cat((self.simple_visual_encoder(obs["gridsensor3"]), self.normalizer(obs["vector_obs"])), dim=1)
        hide = self.linear_encoder(hide)
        dist = self.gaussian_distribution(hide)

        return dist.log_prob(actions)

    def update_normalization(self, vector_obs):
        self.normalizer.update(vector_obs)




