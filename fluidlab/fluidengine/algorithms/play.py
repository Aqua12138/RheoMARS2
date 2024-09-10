import torch
import numpy as np
import gym
from fluidlab.fluidengine.algorithms.utils.running_mean_std import RunningMeanStd
class play:
    def __init__(self, cfg):
        self.env = gym.make(cfg["params"]["diff_env"]["name"], seed=123, loss=False, loss_type='diff', play=True)
        self.device = "cpu"
        self.num_obs = self.env.observation_space.shape
        if not self.num_obs:
            self.num_obs = {}
            for key in list(self.env.observation_space.spaces.keys()):  # 使用list来避免在迭代时修改字典
                self.num_obs[key] = self.env.observation_space.spaces[key].shape
        if cfg['params']['config'].get('obs_rms', False):
            if isinstance(self.num_obs, dict):
                self.obs_rms = []
                for value in self.num_obs.values():
                    self.obs_rms.append(RunningMeanStd(shape=value, device = self.device))
            else:
                self.obs_rms = RunningMeanStd(shape=self.num_obs, device = self.device) #
        self.gamma = cfg['params']['config'].get('gamma', 0.99)
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
        self.obs_rms[0] = checkpoint[3][0].to(self.device)
        self.obs_rms[1] = checkpoint[3][1].to(self.device)
        self.ret_rms = checkpoint[4].to(self.device) if checkpoint[4] is not None else checkpoint[4]

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(num_games=num_games,
                                                                                                  deterministic=True)
        self.print_info(
            'mean episode loss = {}, mean discounted loss = {}, mean episode length = {}'.format(mean_policy_loss,
                                                                                                 mean_policy_discounted_loss,
                                                                                                 mean_episode_length))

    def print_info(self, *message):
        print('\033[96m', *message, '\033[0m')

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic=False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        episode_length = torch.zeros(1, dtype=int)
        episode_gamma = torch.ones(1, dtype=torch.float32, device=self.device)
        episode_discounted_loss = torch.zeros(1, dtype=torch.float32, device=self.device)

        obs = self.env.reset()
        obs['gridsensor3'] = obs['gridsensor3'].unsqueeze(0)
        obs['vector_obs'] = obs['vector_obs'].unsqueeze(0)
        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                self.obs_rms[0].update(obs['gridsensor3'])
                self.obs_rms[1].update(obs['vector_obs'])
            obs['gridsensor3'] = self.obs_rms[0].normalize(obs['gridsensor3'])
            obs['vector_obs'] = self.obs_rms[1].normalize(obs['vector_obs'])
            actions = self.actor(obs, deterministic=deterministic)

            obs, rew, done, _ = self.env.step(actions)
            obs['gridsensor3'] = obs['gridsensor3'].unsqueeze(0)
            obs['vector_obs'] = obs['vector_obs'].unsqueeze(0)
            episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print(
                        'loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.
                    episode_discounted_loss[done_env_id] = 0.
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.
                    games_cnt += 1

        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length