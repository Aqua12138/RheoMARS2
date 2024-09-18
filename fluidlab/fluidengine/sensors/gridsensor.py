import numpy as np
import taichi as ti
import torch
from fluidlab.configs.macros import *
from pyquaternion import Quaternion
import pickle as pkl

@ti.data_oriented
class GridSensor:
    def __init__(self, horizon):
        self.horizon = horizon+1

    def reset(self):
        self.clear_grid_sensor()

    def reset_grad(self):
        pass

    def build(self, sim, particle_state, node_state, grid_state, seeMyself=False):
        self.sim = sim
        self.device = 'cuda:0'
        self.dim = sim.dim

        if self.sim.agent is not None:
            self.agent = sim.agent

        if self.sim.particles is not None:
            self.particle_x = sim.particles.x
            self.particle_mat = sim.particles_i.mat
            self.particle_used = sim.particles_ng.used
            self.n_particles = sim.n_particles
            self.n_bodies = sim.n_bodies # the types of rheological

        self.particles = particle_state.field(shape=(self.horizon, self.n_particles,), needs_grad=True, layout=ti.Layout.SOA)

        # mesh_state
        self.resolution = (32, 32, 32)
        self.dx = 1 / 32
        self.n_nodes = self.resolution[0] * self.resolution[1] * self.resolution[2]

        # init_nodes_x
        self.init_nodes_x()
        self.n_statics = sim.n_statics # the types of statics
        self.statics = sim.statics

        self.nodes = particle_state.field(shape=(self.horizon, self.n_nodes), needs_grad=True, layout=ti.Layout.SOA)
        self.nodes_i = node_state.field(shape=(self.horizon, self.n_nodes), needs_grad=False, layout=ti.Layout.SOA)
        self.n_agents = 1
        if seeMyself:
            self.grid_sensor = grid_state.field(shape=(self.horizon, self.M, self.N, self.n_bodies + self.n_statics+self.n_agents), needs_grad=True,
                                                layout=ti.Layout.SOA)
        else:
            self.grid_sensor = grid_state.field(shape=(self.horizon, self.M, self.N, self.n_bodies + self.n_statics+self.n_agents-1), needs_grad=True,
                                                layout=ti.Layout.SOA)

    def init_nodes_x(self):
        self.nodes_x = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_nodes), needs_grad=False)
        self.initialize_nodes()

    @ti.kernel
    def initialize_nodes(self):
        for I in ti.grouped(ti.ndrange(*self.resolution)):
            id = I[2] * (self.resolution[0] * self.resolution[1]) + I[1] * self.resolution[0] + I[0]
            self.nodes_x[id][0] = I[0] * self.dx
            self.nodes_x[id][1] = I[1] * self.dx
            self.nodes_x[id][2] = I[2] * self.dx

    def clear_grid_sensor(self):
        pass