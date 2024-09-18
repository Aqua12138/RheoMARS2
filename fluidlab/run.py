import os
import gymnasium as gym
import torch
import random
import argparse
import numpy as np
import yaml
import fluidlab.envs
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import solve_policy
from fluidlab.optimizer.recorder import record_target, replay_policy, replay_target, record_target_grid, debug, eval
from fluidlab.utils.config import load_config
from fluidlab.optimizer.rl_trainer import PPO_trainer, SHAC_trainer
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--env_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=334)
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--user_input", action='store_true')
    parser.add_argument("--replay_policy", action='store_true')
    parser.add_argument("--replay_target", action='store_true')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--renderer_type", type=str, default='GGUI')
    parser.add_argument("--loss_type", type=str, default='default')
    parser.add_argument("--collect_data", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--rl", type=str, default='default')
    parser.add_argument("--perc_type", type=str, default='physics')
    parser.add_argument("--pre_train_model", type=str, default=None)
    parser.add_argument("--intial_model", action='store_true')



    args = parser.parse_args()

    return args

def main():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    else:
        cfg = None

    if args.record:
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=False, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type)
        record_target(env, path=args.path, user_input=args.user_input)

    elif args.replay_target:
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=False, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type)
        replay_target(env)

    elif args.replay_policy:
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=False, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type)
        replay_policy(env, path=args.path)

    elif args.collect_data:
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=False, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type)
        record_target_grid(env)

    elif args.debug:
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=True, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type, perc_type=args.perc_type)
        debug(env)

    elif args.rl == "ppo":
        PPO = PPO_trainer(cfg, args)
        PPO.solver()

    elif args.rl == "shac":
        SHAC = SHAC_trainer(cfg, args)
        SHAC.solver()

    elif args.rl == "eval":
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=True, loss_cfg=cfg.params.loss,
                       renderer_type=args.renderer_type, perc_type=args.perc_type)
        eval(env, model_path=args.pre_train_model)
    else:
        logger = Logger(args.exp_name)
        env = gym.make(cfg.params.env.name, seed=cfg.params.env.seed, loss=True, loss_cfg=cfg.params.loss, renderer_type=args.renderer_type)
        solve_policy(env, logger, cfg.params.config)


if __name__ == '__main__':
    main()
