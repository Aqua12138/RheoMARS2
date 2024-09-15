import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridsensor import GridSensor
import cv2

from fluidlab.utils.geom import quaternion_to_rotation_matrix

@ti.data_oriented
class GridSensor2D(GridSensor):
    def __init__(self, sensor_name, cell_size, x_range, z_range, target_file=None, **kwargs):
        super(GridSensor2D, self).__init__(**kwargs)
        '''
        CellScale: 网格尺寸
        GridSize: 网格检测范围（cellArc latAngleSouth latAngleNorth LonAngle maxDistance minDistance DistanceNormalization）
        RotateWithAgent: 是否随Agent旋转
        agent: Agent
        AgentID: effetor ID
        DetectableTags: 检测物体body tuple
        MaxColliderBufferSize: 最大检测物数量
        DebugColors: 颜色显示，用于debug
        GizmoZOffset: 沿着Z偏移的尺寸
        DataType: 数据类型 目前支持one-hot

        '''
        # Geometry
        self.sensor_name = sensor_name
        self.m_CellSize = cell_size

        self.m_XRange = x_range
        self.m_ZRange = z_range
        self.M = int(self.m_XRange / self.m_CellSize) # gridsensor m
        self.N = int(self.m_ZRange / self.m_CellSize)  # gridsensor n

    @property
    def name(self):
        return self.sensor_name

    def build(self, sim):
        particle_state = ti.types.struct(
            relative_x=ti.types.vector(sim.dim, DTYPE_TI),
            rotated_x=ti.types.vector(sim.dim, DTYPE_TI),
            latitudes=DTYPE_TI,
            longitudes=DTYPE_TI,
            distance=DTYPE_TI
        )

        node_state = ti.types.struct(
            trigger=ti.i32,
            id=ti.i32,
        )

        grid_state = ti.types.struct(
            one_hot=DTYPE_TI
        )

        super().build(sim, particle_state, node_state, grid_state, seeMyself=True)

    @ti.kernel
    def ont_hot_particles(self, s: ti.i32, f:ti.i32):
        for p in range(self.n_particles):
            x = int(ti.floor(self.particle_x[f, p][0] * self.M))
            z = int(ti.floor(self.particle_x[f, p][2] * self.N))
            # 判断x和z是否在范围内
            if 0 <= x < self.M and 0 <= z < self.N:
                self.grid_sensor[s, self.N-z-1, x, self.sim.particles_i[p].body_id].one_hot = 1

    @ti.kernel
    def one_hot_nodes(self, s: ti.i32, f:ti.i32):
        for n in range(self.n_nodes):
            if self.agent.effectors[0].mesh.is_collide(f, self.nodes_x[n]):
                self.grid_sensor[s, self.N-int(self.nodes_x[n][2] * self.N)-1, int(self.nodes_x[n][0] * self.M), self.n_bodies].one_hot = 1
            for i in ti.static(range(self.n_statics)):
                if self.statics[i].is_collide(self.nodes_x[n]):
                    self.grid_sensor[s, self.N-int(self.nodes_x[n][2] * self.N)-1, int(self.nodes_x[n][0] * self.M), self.n_bodies+1+i].one_hot = 1

    @ti.kernel
    def get_sensor_data_kernel(self, s: ti.i32, grid_sensor: ti.types.ndarray()):
        # 这里假设 output 已经是一个正确维度和类型的 Taichi field
        for i, j, k in ti.ndrange(self.M, self.N, self.n_bodies+1+self.n_statics):
            grid_sensor[i, j, k] = self.grid_sensor[s, i, j, k].one_hot

    def step(self):
        self.ont_hot_particles(self.sim.cur_step_global, self.sim.cur_substep_local)
        if self.n_statics > 0:
            self.one_hot_nodes(self.sim.cur_step_global, self.sim.cur_substep_local)

    def step_grad(self):
        if self.n_statics > 0:
            self.one_hot_nodes.grad(self.sim.cur_step_global, self.sim.cur_substep_local)
        self.ont_hot_particles.grad(self.sim.cur_step_global, self.sim.cur_substep_local)

    def get_obs(self):
        grid_sensor = torch.zeros((self.M, self.N, self.n_bodies+1+self.n_statics), dtype=torch.float32, device=self.device)
        self.get_sensor_data_kernel(self.sim.cur_step_global, grid_sensor)

        return grid_sensor[..., :3]


    def clear_grid_sensor(self):
        self.particles.fill(0)
        self.particles.grad.fill(0)
        self.nodes.fill(0)
        self.nodes.grad.fill(0)
        self.grid_sensor.fill(0)
        self.grid_sensor.grad.fill(0)
        self.step()
    def reset_grad(self):
        super().reset_grad()
        self.particles.grad.fill(0)
        self.nodes.grad.fill(0)
        self.grid_sensor.grad.fill(0)

    @ti.kernel
    def set_next_state_grad(self, s: ti.i32, grad: ti.types.ndarray()):
        for i, j, k in ti.ndrange(self.M,
                                  self.N,
                                  self.n_bodies+1+self.n_statics):
            self.grid_sensor.grad[s, i, j, k].one_hot = grad[i, j, k]