# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle = False, drop_last = False):
        self.obs = {"gridsensor2d": obs["gridsensor2d"].view(-1, obs["gridsensor2d"].shape[2], obs["gridsensor2d"].shape[3], obs["gridsensor2d"].shape[4]),
                    "gridsensor3d": obs["gridsensor3d"].view(-1, obs["gridsensor3d"].shape[2], obs["gridsensor3d"].shape[3], obs["gridsensor3d"].shape[4]),
                    "vector_obs": obs["vector_obs"].view(-1, obs["vector_obs"].shape[-1])}

        self.target_values = target_values.view(-1)
        self.batch_size = batch_size

        if shuffle:
            self.shuffle()
        
        if drop_last:
            self.length = self.obs["vector_obs"].shape[0] // self.batch_size
        else:
            self.length = ((self.obs["vector_obs"].shape[0] - 1) // self.batch_size) + 1
    
    def shuffle(self):
        index = np.random.permutation(self.obs["vector_obs"].shape[0])
        self.obs = {"gridsensor2d": self.obs["gridsensor2d"][index, :],
                    "gridsensor3d": self.obs["gridsensor3d"][index, :],
                    "vector_obs": self.obs["vector_obs"][index, :]}
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, int(self.obs["vector_obs"].shape[0]))
        tmp_obs = {"gridsensor2d": self.obs["gridsensor2d"][start_idx:end_idx, :],
                   "gridsensor3d": self.obs["gridsensor3d"][start_idx:end_idx, :],
                   "vector_obs": self.obs["vector_obs"][start_idx:end_idx, :]}
        return {'obs': tmp_obs, 'target_values': self.target_values[start_idx:end_idx]}
