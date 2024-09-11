import pickle as pkl
from fluidlab.utils.misc import *
from fluidlab.fluidengine.losses.Adam.loss import Loss


@ti.data_oriented
class SHACLoss(Loss):
    def __init__(
            self,
            **kwargs,
    ):
        super(SHACLoss, self).__init__(**kwargs)
        self._gamma = 0.99
        self.gamma = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.rew = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps+1,), needs_grad=True)
        self.rew_acc = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps+1,), needs_grad=True)
        self.actor_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

    def build(self, sim):
        self.density_weight = self.weights['density']
        self.density_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps+1,), needs_grad=True)

        self.sdf_loss = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps+1,), needs_grad=True)
        self.sdf_weight = self.weights['sdf']

        self.contact_loss = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps+1,), needs_grad=True)
        self.min_dist = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps+1,), needs_grad=True)
        self.dist_norm = ti.field(dtype=DTYPE_TI_64, shape=(self.max_loss_steps+1,), needs_grad=True)
        self.contact_weight = self.weights['contact']
        self.soft_contact_loss = self.weights['is_soft_contact']

        self.target_density = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        self.target_sdf = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        self.nearest_point = ti.Vector.field(sim.dim, dtype=DTYPE_TI_64, shape=sim.res)
        self.target_sdf_copy = ti.field(dtype=DTYPE_TI_64, shape=sim.res)
        self.nearest_point_copy = ti.Vector.field(sim.dim, dtype=DTYPE_TI_64, shape=sim.res)

        self.inf = 1000
        super(SHACLoss, self).build(sim)

        self.primitive = self.agent.effectors[0].mesh
        self.particle_mass = self.sim.particles_i.mass
        self.grid_mass = ti.field(dtype=DTYPE_TI_64, shape=(*self.res,), needs_grad=True)
        self.tgt_particles_x = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.n_particles)
        self.iou()
        self._target_iou = self._iou

    @ti.func
    def isnan(self, x):
        return not (x <= 0 or x >= 0)

    def iou(self):
        self._iou = self.iou_kernel()

    @ti.kernel
    def iou_kernel(self) -> ti.float64:
        ma = ti.cast(0., DTYPE_TI_64)
        mb = ti.cast(0., DTYPE_TI_64)
        I = ti.cast(0., DTYPE_TI_64)
        Ua = ti.cast(0., DTYPE_TI_64)
        Ub = ti.cast(0., DTYPE_TI_64)
        for i in ti.grouped(ti.ndrange(*self.res)):
            ti.atomic_max(ma, self.grid_mass[i])
            ti.atomic_max(mb, self.target_density[i])
            I += self.grid_mass[i] * self.target_density[i]
            Ua += self.grid_mass[i]
            Ub += self.target_density[i]
        I = I / ma / mb
        U = Ua / ma + Ub / mb

        return I / (U - I)

    def reset_grad(self):
        super(SHACLoss, self).reset_grad()
        self.density_loss.grad.fill(0)
        self.sdf_loss.grad.fill(0)
        self.contact_loss.grad.fill(0)
        self.min_dist.grad.fill(0)
        self.dist_norm.grad.fill(0)
        self.rew.grad.fill(0)
        self.rew_acc.grad.fill(0)
        self.actor_loss.grad.fill(1)

    def load_target(self, path):
        self.targets = pkl.load(open(path, 'rb'))
        print(f'===>  Target loaded from {path}.')

    def update_target(self, i):
        self.tgt_particles_x.from_numpy(self.targets['last_pos'][i])
        self.grids = self.targets['last_grid'][i]
        self.target_density.from_numpy(self.grids)
        self.grid_mass.from_numpy(self.grids)
        self._update_target()

    # -----------------------------------------------------------
    # preprocess target to calculate sdf
    # -----------------------------------------------------------

    def _update_target(self):
        self.target_sdf_copy.fill(self.inf)
        for i in range(self.n_grid * 2):
            self.update_target_sdf()

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def update_target_sdf(self):
        for I in ti.grouped(self.target_sdf):
            self.target_sdf[I] = self.inf
            grid_pos = ti.cast(I * self.dx, DTYPE_TI_64)
            if self.target_density[I] > 1e-4:  # TODO: make it configurable
                self.target_sdf[I] = 0.
                self.nearest_point[I] = grid_pos
            else:
                for offset in ti.grouped(ti.ndrange(*(((-3, 3),) * self.dim))):
                    v = I + offset
                    if v.min() >= 0 and v.max() < self.n_grid and ti.abs(offset).sum() != 0:
                        if self.target_sdf_copy[v] < self.inf:
                            nearest_point = self.nearest_point_copy[v]
                            dist = self.norm(grid_pos - nearest_point)
                            if dist < self.target_sdf[I]:
                                self.nearest_point[I] = nearest_point
                                self.target_sdf[I] = dist
        for I in ti.grouped(self.target_sdf):
            self.target_sdf_copy[I] = self.target_sdf[I]
            self.nearest_point_copy[I] = self.nearest_point[I]

    # other
    @ti.kernel
    def clear_losses(self):
        self.density_loss.fill(0)
        self.density_loss.grad.fill(0)

        self.sdf_loss.fill(0)
        self.sdf_loss.grad.fill(0)

        self.contact_loss.fill(0)
        self.contact_loss.grad.fill(0)

        if not self.soft_contact_loss:
            self.min_dist.fill(self.inf)
        else:
            self.min_dist.fill(0)
        self.min_dist.grad.fill(0)
        self.dist_norm.fill(0)
        self.dist_norm.grad.fill(0)

        self.rew.fill(0)
        self.rew.grad.fill(0)

        self.gamma.fill(1)

        self.rew_acc.fill(0)
        self.rew_acc.grad.fill(0)

        self.actor_loss.fill(0)
        self.actor_loss.grad.fill(1)

    def init_step_loss(self, s, f):
        self.compute_grid_mass(f)
        self.iou()
        self.compute_density_loss_kernel(s)
        self.compute_sdf_loss_kernel(s)
        self.compute_contact_loss(s, f)

        self.sum_up_loss_kernel(s)

    def compute_step_loss(self, s, f):
        self.compute_grid_mass(f)
        self.iou()
        self.compute_density_loss_kernel(s)
        self.compute_sdf_loss_kernel(s)
        self.compute_contact_loss(s, f)

        self.sum_up_loss_kernel(s)

        self.compute_reward_kernel(s)
        self.compute_gamma_reward_kernel(s)

    def compute_step_loss_grad(self, s, f):
        # print(self.rew_acc.grad[s])
        self.compute_gamma_reward_kernel.grad(s)
        # print(self.rew.grad[None])
        self.compute_reward_kernel.grad(s)
        # print(self.step_loss.grad[s-1])
        # print(self.step_loss.grad[s])
        self.sum_up_loss_kernel.grad(s)
        # print(self.density_loss.grad[s])
        # print(self.sdf_loss.grad[s])
        # print(self.contact_loss.grad[s])
        self.compute_contact_loss_grad(s, f)
        self.compute_sdf_loss_kernel.grad(s)
        # print(self.grid_mass.grad)
        self.compute_density_loss_kernel.grad(s)
        self.compute_grid_mass_grad(f)

    def compute_grid_mass(self, f):
        self.grid_mass.fill(0)
        self.grid_mass.grad.fill(0)
        self.compute_grid_mass_kernel(f)

    def compute_grid_mass_grad(self, f):
        self.compute_grid_mass_kernel.grad(f)

    def compute_contact_loss(self, s, f):
        if self.soft_contact_loss:
            self.compute_contact_distance_normalize_kernel(s, f)
            self.compute_soft_contact_distance_kernel(s, f)
        else:
            self.compute_contact_distance_kernel(s, f)
        self.compute_contact_loss_kernel(s)

    def compute_contact_loss_grad(self, s, f):
        self.compute_contact_loss_kernel.grad(s)
        if self.soft_contact_loss:
            self.compute_soft_contact_distance_kernel.grad(s, f)
            self.compute_contact_distance_normalize_kernel.grad(s, f)
        else:
            self.compute_contact_distance_kernel.grad(s, f)

    @ti.kernel
    def compute_grid_mass_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][1]) and self.isnan(
                    self.particle_x[f, p][2])):
                base = (self.particle_x[f, p] * self.sim.inv_dx - 0.5).cast(int)
                fx = self.particle_x[f, p] * self.sim.inv_dx - base.cast(DTYPE_TI)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                for offset in ti.static(ti.grouped(self.sim.stencil_range())):
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    self.grid_mass[base + offset] += weight * self.particle_mass[p]

    @ti.kernel
    def compute_density_loss_kernel(self, s: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.density_loss[s] += ti.abs(self.grid_mass[I] - self.target_density[I])

    @ti.kernel
    def compute_sdf_loss_kernel(self, s: ti.i32):
        for I in ti.grouped(self.grid_mass):
            self.sdf_loss[s] += self.target_sdf[I] * self.grid_mass[I]

    @ti.kernel
    def compute_contact_distance_normalize_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][0]) \
                    and self.isnan(self.particle_x[f, p][0])):
                d_ij = ti.max(self.primitive.sdf(f, self.particle_x[f, p]), 0.)
                ti.atomic_add(self.dist_norm[s], self.soft_weight(d_ij))

    @ti.kernel
    def compute_soft_contact_distance_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][0]) \
                    and self.isnan(self.particle_x[f, p][0])):
                d_ij = ti.max(self.primitive.sdf(f, self.particle_x[f, p]), 0.)
                ti.atomic_add(self.min_dist[s], d_ij * self.soft_weight(d_ij) / self.dist_norm[s])

    @ti.kernel
    def compute_contact_distance_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if not (self.isnan(self.particle_x[f, p][0]) and self.isnan(self.particle_x[f, p][0]) \
                    and self.isnan(self.particle_x[f, p][0])):
                d_ij = ti.max(self.primitive.sdf(f, self.particle_x[f, p]), 0.)
                ti.atomic_min(self.min_dist[s], ti.max(d_ij, 0.))

    @ti.kernel
    def compute_contact_loss_kernel(self, s: ti.i32):
        self.contact_loss[s] += self.min_dist[s] ** 2

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.density_loss[s] * self.density_weight
        self.step_loss[s] += self.sdf_loss[s] * self.sdf_weight
        self.step_loss[s] += self.contact_loss[s] * self.contact_weight

    @ti.kernel
    def compute_reward_kernel(self, s: ti.i32):
        self.rew[s] = ti.min(ti.max((self.step_loss[s - 1] - self.step_loss[s]), 1e-5), )
        self.rew[s] = ti.select(ti.abs(self.rew[s]) < 1e-5, 0, self.rew[s])



    @ti.kernel
    def compute_gamma_reward_kernel(self, s: ti.i32):
        self.rew_acc[s] = self.rew_acc[s - 1] + self.gamma[None] * self.rew[s]
        self.gamma[None] *= self._gamma

    @ti.kernel
    def compute_actor_loss_kernel(self, s: ti.i32):
        self.actor_loss[None] = -self.rew_acc[s]

    def expand_temporal_range(self):
        if self.temporal_range_type == 'expand':
            loss_improved = (self.best_loss - self.total_loss[None])  # loss 上升值
            loss_improved_rate = loss_improved / self.best_loss  # loss 上升速度

            # 判断停滞情况
            if loss_improved_rate < self.plateau_thresh[0] or loss_improved < self.plateau_thresh[1]:
                self.plateau_count += 1
                print('Plateaued!!!', self.plateau_count)
            else:
                self.plateau_count = 0

            if self.best_loss > self.total_loss[None]:
                self.best_loss = self.total_loss[None]

            if self.plateau_count >= self.plateau_count_limit:
                self.plateau_count = 0
                self.best_loss = self.inf

                self.temporal_range[1] = min(self.max_loss_steps, self.temporal_range[1] + self.temporal_expand_speed)
                print(f'temporal range expanded to {self.temporal_range}')

    def compute_actor_loss(self):
        self.compute_actor_loss_kernel(self.sim.cur_step_global)

    def compute_actor_loss_grad(self):
        self.compute_actor_loss_kernel.grad(self.sim.cur_step_global)

    def get_step_loss(self):
        loss = self.step_loss[self.sim.cur_step_global]
        reward = self.rew[self.sim.cur_step_global]
        self._last_loss = loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info
