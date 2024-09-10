import os
import cv2
import numpy as np
import taichi as ti
from fluidlab.utils.misc import is_on_server


class Solver:
    def __init__(self, env, logger=None, cfg=None):
        self.cfg = cfg
        self.env = env
        self.target_file = env.target_file
        self.logger = logger

    def solve(self):
        taichi_env = self.env.taichi_env
        policy = self.env.trainable_policy(self.cfg.optim, self.cfg.init_range)

        taichi_env_state = taichi_env.get_state()

        def forward_backward(sim_state, policy, horizon):

            taichi_env.set_state(sim_state, grad_enabled=True)

            # forward pass
            from time import time
            t1 = time()
            taichi_env.apply_agent_action_p(policy.get_actions_p())
            cur_horizon = taichi_env.loss.temporal_range[1]
            for i in range(cur_horizon):
                if i < horizon:
                    action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
                else:
                    action = None
                taichi_env.step(action)

            loss_info = taichi_env.get_final_loss()
            t2 = time()

            # backward pass
            taichi_env.reset_grad()
            taichi_env.get_final_loss_grad()

            for i in range(cur_horizon-1, policy.freeze_till-1, -1):
                if i < horizon:
                    action = policy.get_action_v(i)
                else:
                    action = None
                taichi_env.step_grad(action)

            taichi_env.apply_agent_action_p_grad(policy.get_actions_p())
            t3 = time()
            print(f'=======> forward: {t2-t1:.2f}s backward: {t3-t2:.2f}s')
            return loss_info, taichi_env.agent.get_grad(horizon)

        def gradient_clipping(grads, clip_value):
            """
            对梯度进行裁减

            参数：
            grads: numpy数组，形状为(1000, 3)
            clip_value: float，裁减阈值

            返回：
            裁减后的梯度
            """
            # 计算每个样本的梯度范数（L2范数）
            grad_norms = np.linalg.norm(grads, axis=1)

            # 如果梯度范数超过阈值，则进行裁减
            clip_coef = np.where(grad_norms > clip_value, clip_value / grad_norms, 1.0)

            # 对每个梯度进行缩放
            clipped_grads = grads * clip_coef[:, np.newaxis]

            return clipped_grads

        for iteration in range(self.cfg.n_iters):
            self.logger.save_policy(policy, iteration)
            if iteration % 50 == 0:
                self.render_policy(taichi_env, taichi_env_state, policy, self.env.horizon, iteration)
            loss_info, grad = forward_backward(taichi_env_state['state'], policy, self.env.horizon)
            clipped_grad = gradient_clipping(grad, 1)
            loss = loss_info['loss']
            loss_info['iteration'] = iteration
            policy.optimize(clipped_grad, loss_info)

            if self.logger is not None:
                loss_info['lr'] = policy.optim.lr
                self.logger.log(iteration, loss_info)


    def render_policy(self, taichi_env, init_state, policy, horizon, iteration):
        if is_on_server():
            return

        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(policy.get_actions_p())

        for i in range(horizon):
            action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            taichi_env.step(action)
            # print(i, taichi_env.get_step_loss())

            save = True
            save = False
            if save:
                img = taichi_env.render('rgb_array')
                self.logger.write_img(img, iteration, i)
            else:
                if hasattr(taichi_env.loss, 'tgt_particles_x'):
                    taichi_env.render('human', taichi_env.loss.tgt_particles_x)
                else:
                    taichi_env.render('human')

def solve_policy(env, logger, cfg):
    env.reset()
    solver = Solver(env, logger, cfg)
    solver.solve()
