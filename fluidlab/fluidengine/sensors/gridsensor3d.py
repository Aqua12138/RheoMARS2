import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
from .gridsensor import GridSensor
import cv2
from fluidlab.utils.geom import quaternion_to_rotation_matrix

@ti.data_oriented
class GridSensor3D(GridSensor):
    def __init__(self, sesnor_name, cell_arc, lat_angle_north, lat_angle_south, lon_angle,
                 max_distance, min_distance, distance_normalization, **kwargs):
        super(GridSensor3D, self).__init__(**kwargs)
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
        self.sensor_name = sesnor_name
        self.m_CellArc = cell_arc
        self.m_LatAngleNorth = lat_angle_north
        self.m_LatAngleSouth = lat_angle_south
        self.m_LonAngle = lon_angle
        self.m_MaxDistance = max_distance
        self.m_MinDistance = min_distance
        self.m_DistanceNormalization = distance_normalization
        self.M = (self.m_LatAngleNorth + self.m_LatAngleSouth) // self.m_CellArc # gridsensor n
        self.N = (self.m_LonAngle // self.m_CellArc) * 2  # gridsensor m

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
            distance=DTYPE_TI,
            one_hot=DTYPE_TI
        )

        super().build(sim, particle_state, node_state, grid_state)

    @ti.kernel
    def transform_point_particle(self, s: ti.i32, f:ti.i32):
        # 计算point相对agent位置
        for p in range(self.n_particles):
            self.particles[s, p].relative_x[0] = self.particle_x[f, p][0] - self.agent.effectors[0].pos[f][0]
            self.particles[s, p].relative_x[1] = self.particle_x[f, p][1] - self.agent.effectors[0].pos[f][1]
            self.particles[s, p].relative_x[2] = self.particle_x[f, p][2] - self.agent.effectors[0].pos[f][2]

            # 获取四元数数据
            a = self.agent.effectors[0].quat[f][0]
            b = -self.agent.effectors[0].quat[f][1]
            c = -self.agent.effectors[0].quat[f][2]
            d = -self.agent.effectors[0].quat[f][3]
            rotation_matrix = quaternion_to_rotation_matrix(a, b, c, d)
            self.particles[s, p].rotated_x = rotation_matrix @ self.particles[s, p].relative_x

    @ti.kernel
    def transform_point_node(self, s: ti.i32, f: ti.i32):
        # statics
        for n in range(self.n_nodes):
            for i in ti.static(range(self.n_statics)):
                if self.statics[i].is_collide(self.nodes_x[n]):
                    self.nodes[s, n].relative_x[0] = self.nodes_x[n][0] - self.agent.effectors[0].pos[f][0]
                    self.nodes[s, n].relative_x[1] = self.nodes_x[n][1] - self.agent.effectors[0].pos[f][1]
                    self.nodes[s, n].relative_x[2] = self.nodes_x[n][2] - self.agent.effectors[0].pos[f][2]

                    # 获取四元数数据
                    a = self.agent.effectors[0].quat[f][0]
                    b = -self.agent.effectors[0].quat[f][1]
                    c = -self.agent.effectors[0].quat[f][2]
                    d = -self.agent.effectors[0].quat[f][3]
                    rotation_matrix = quaternion_to_rotation_matrix(a, b, c, d)
                    self.nodes[s, n].rotated_x = rotation_matrix @ self.nodes[s, n].relative_x
                    self.nodes_i[s, n].trigger = 1
                    self.nodes_i[s, n].id = i

        # dynamics 3d grid sensor do not see itself

    @ti.kernel
    def compute_lat_lon_particle(self, s: ti.i32):
        for i in range(self.n_particles):
            # 提取局部坐标系中的坐标
            x = self.particles[s, i].rotated_x[0]
            y = self.particles[s, i].rotated_x[1]
            z = self.particles[s, i].rotated_x[2]

            # 计算纬度和经度
            # 计算纬度
            self.particles[s, i].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = ti.max(ti.min(y / self.particles[s, i].distance, 1.0), -1.0)
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)

            self.particles[s, i].latitudes = 180 - lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.particles[s, i].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def compute_lat_lon_node(self, s: ti.i32):
        for n in range(self.n_nodes):
            # if self.nodes_i[s, n].trigger:
            # 提取局部坐标系中的坐标
            x = self.nodes[s, n].rotated_x[0]
            y = self.nodes[s, n].rotated_x[1]
            z = self.nodes[s, n].rotated_x[2]
            # 计算纬度和经度
            # 计算纬度
            self.nodes[s, n].distance = ti.sqrt(x * x + y * y + z * z)
            cos_lat_rad = ti.max(ti.min(y / self.nodes[s, n].distance, 1.0), -1.0)
            lat_rad = ti.acos(cos_lat_rad)
            lon_rad = ti.atan2(x, -z)

            self.nodes[s, n].latitudes = 180 - lat_rad * (
                    180.0 / ti.acos(-1.0))  # acos(-1) is a way to get π in Taichi
            self.nodes[s, n].longitudes = lon_rad * (180.0 / ti.acos(-1.0))

    @ti.kernel
    def normal_distance_particle(self, s: ti.i32):
        # 1. 判断距离是否在球体内
        for p in range(self.n_particles):
            if self.particle_used[s, p] != 0:
                if self.particles[s, p].distance < self.m_MaxDistance and self.particles[s, p].distance > self.m_MinDistance:
                    # 2. 判断经度范围和纬度范围
                    if (90 - self.particles[s, p].latitudes < self.m_LatAngleSouth and 90 - self.particles[s, p].latitudes >= 0) or \
                            (ti.abs(self.particles[s, p].latitudes - 90) < self.m_LatAngleNorth and ti.abs(self.particles[s, p].latitudes - 90) >= 0):
                        if ti.abs(self.particles[s, p].longitudes) < self.m_LonAngle:
                            # 计算加权距离
                            d = (self.particles[s, p].distance - self.m_MinDistance) / (self.m_MaxDistance - self.m_MinDistance)
                            normal_d = 0.0
                            if self.m_DistanceNormalization == 1:
                                normal_d = 1 - d
                            else:
                                normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                            self.m_DistanceNormalization + 1)
                            # 计算经纬度索引
                            longitude_index = ti.cast(
                                ti.floor((self.particles[s, p].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                            latitude_index = ti.cast(
                                ti.floor(
                                     (self.particles[s, p].latitudes - (90 - self.m_LatAngleSouth)) / self.m_CellArc),
                                ti.i32)

                            # 使用 atomic_max 更新 normal_distance 的值
                            ti.atomic_max(self.grid_sensor[s, latitude_index, longitude_index, self.sim.particles_i[p].body_id].distance, normal_d)

    @ti.kernel
    def normal_distance_node(self, s: ti.i32):
        # 1. 判断距离是否在球体内
        for n in range(self.n_nodes):
            if self.nodes[s, n].distance < self.m_MaxDistance and self.nodes[s, n].distance > self.m_MinDistance:
                # 2. 判断经度范围和纬度范围
                if (90 - self.nodes[s, n].latitudes < self.m_LatAngleSouth and 90 - self.nodes[s, n].latitudes >= 0) or \
                        (ti.abs(self.nodes[s, n].latitudes - 90) < self.m_LatAngleNorth and ti.abs(self.nodes[s, n].latitudes - 90) >= 0):
                    if ti.abs(self.nodes[s, n].longitudes) < self.m_LonAngle:
                        # 计算加权距离
                        d = (self.nodes[s, n].distance - self.m_MinDistance) / (
                                self.m_MaxDistance - self.m_MinDistance)
                        normal_d = 0.0
                        if self.m_DistanceNormalization == 1:
                            normal_d = 1 - d
                        else:
                            normal_d = 1 - d / (self.m_DistanceNormalization + ti.abs(d)) * (
                                    self.m_DistanceNormalization + 1)
                        # 计算经纬度索引
                        longitude_index = ti.cast(
                            ti.floor((self.nodes[s, n].longitudes + self.m_LonAngle) / self.m_CellArc), ti.i32)
                        latitude_index = ti.cast(
                            ti.floor(
                                (self.nodes[s, n].latitudes - (90 - self.m_LatAngleSouth)) / self.m_CellArc),
                            ti.i32)

                        # 使用 atomic_max 更新 normal_distance 的值
                        ti.atomic_max(self.grid_sensor[s, latitude_index, longitude_index, self.n_bodies+self.nodes_i[s, n].id].distance, normal_d)

    @ti.kernel
    def get_sensor_data_kernel(self, s: ti.i32, grid_sensor: ti.types.ndarray()):
        # 这里假设 output 已经是一个正确维度和类型的 Taichi field
        for i, j, k in ti.ndrange(self.M, self.N, self.n_bodies+self.n_statics):
            grid_sensor[i, j, k] = self.grid_sensor[s, i, j, k].distance

    def step(self):
        self.transform_point_particle(self.sim.cur_step_global, self.sim.cur_substep_local)
        self.compute_lat_lon_particle(self.sim.cur_step_global)
        self.normal_distance_particle(self.sim.cur_step_global)

        if self.n_statics > 0:
            self.transform_point_node(self.sim.cur_step_global, self.sim.cur_substep_local)
            self.compute_lat_lon_node(self.sim.cur_step_global)
            self.normal_distance_node(self.sim.cur_step_global)

    def step_grad(self):
        if self.n_statics > 0:
            self.normal_distance_node.grad(self.sim.cur_step_global)
            self.compute_lat_lon_node.grad(self.sim.cur_step_global)
            self.transform_point_node.grad(self.sim.cur_step_global, self.sim.cur_substep_local)

        self.normal_distance_particle.grad(self.sim.cur_step_global)
        self.compute_lat_lon_particle.grad(self.sim.cur_step_global)
        self.transform_point_particle.grad(self.sim.cur_step_global, self.sim.cur_substep_local)

    def get_obs(self):
        grid_sensor = torch.zeros((self.M, self.N, self.n_bodies+self.n_statics), dtype=torch.float32, device=self.device)
        self.get_sensor_data_kernel(self.sim.cur_step_global, grid_sensor)

        # np.save('/home/zhx/PycharmProjects/draw/image/gridsensor3d.npy', grid_sensor[..., 0:2].detach().cpu().numpy())
        # cv2.imshow('3d grid sensor', grid_sensor.detach().cpu().numpy())
        # cv2.waitKey(1)
        return grid_sensor[..., :2]

    def clear_grid_sensor(self):
        self.particles.fill(0)
        self.particles.grad.fill(0)
        self.nodes.fill(0)
        self.nodes.grad.fill(0)
        self.nodes_i.fill(0)
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
                                  self.n_bodies+self.n_statics):
            self.grid_sensor.grad[s, i, j, k].distance = grad[i, j, k]