# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import numpy as np

from fluidlab.fluidengine.algorithms.models import model_utils
from fluidlab.fluidengine.algorithms.models.mlagent_torch import encoders
from fluidlab.fluidengine.algorithms.models.mlagent_torch import attention
from fluidlab.fluidengine.algorithms.models.mlagent_torch import layers
from fluidlab.fluidengine.algorithms.models.mlagent_torch import decoders

class CriticMLP(nn.Module):
    def __init__(self, obs_dim, cfg_network, device='cuda:0'):
        super(CriticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['critic_mlp']['units'] + [1]
        self.simple_visual_encoder = encoders.SimpleVisualEncoder(height=90,
                                                                  width=180,
                                                                  initial_channels=2,
                                                                  output_size=256).to(device)  # 视觉输入维度
        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['critic_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)
    
        self.obs_dim = obs_dim

        print(self.critic)

    def forward(self, observations):
        cat_obs = torch.cat((self.simple_visual_encoder(observations["gridsensor3"]), observations["vector_obs"]), dim=1)
        return self.critic(cat_obs)

class CriticStochasticPPO(nn.Module):
    def __init__(self, obs_dim, cfg_network, device='cuda:0'):
        super(CriticStochasticPPO, self).__init__()
        self.device = device
        self.h_size = 256
        self.simple_visual_encoder = encoders.SimpleVisualEncoder(height=obs_dim['gridsensor3'][0],
                                                                  width=obs_dim['gridsensor3'][1],
                                                                  initial_channels=obs_dim['gridsensor3'][2],
                                                                  output_size=256).to(device)  # 视觉输入维度
        self.normalizer = encoders.Normalizer(vec_obs_size=obs_dim['vector_obs'][0]).to(device)  # vector obs维度
        # self.linear_encoder = layers.LinearEncoder(input_size=256, num_layers=3, hidden_size=self.h_size, kernel_gain=(0.125 / self.h_size) ** 0.5).to(device)
        self.linear_encoder = layers.LinearEncoder(input_size=256 + obs_dim['vector_obs'][0], num_layers=3,
                                                   hidden_size=self.h_size, kernel_gain=(0.125 / self.h_size) ** 0.5).to(device)
        self.value_head = decoders.ValueHeads(stream_names=["extrinsic"], input_size=256, output_size=1).to(device)

    def forward(self, observations):
        hide = torch.cat((self.simple_visual_encoder(observations["gridsensor3"]), self.normalizer(observations["vector_obs"])), dim=1)
        hide = self.linear_encoder(hide)
        value = self.value_head(hide)
        return value

    def update_normalization(self, vector_obs):
        self.normalizer.update(vector_obs)
