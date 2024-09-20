import pickle as pkl
from fluidlab.utils.misc import *
from fluidlab.fluidengine.losses.Adam.loss import Loss
import ot


class PPOWLoss(Loss):
    def __init__(
            self,
            **kwargs,
        ):
        super(PPOWLoss, self).__init__(**kwargs)

    def build(self, sim):
        super().build(sim)
        # self.particle_mass = self.sim.particles_i.mass
        # self.grid_mass = ti.field(dtype=DTYPE_TI_64, shape=(*self.res,), needs_grad=True)
        self.tgt_particles_x = None
        res = (10, 10, 10)

        self.grid = None
        self.target_grid = None
        self.step_loss = []


    def clear_loss(self):
        self.step_loss = []

    def reset_grad(self):
        ...
    def load_target(self, path):
        self.targets = pkl.load(open(path, 'rb'))
        print(f'===>  Target loaded from {path}.')

    # -----------------------------------------------------------
    # preprocess target to calculate sdf
    # -----------------------------------------------------------

    def update_target(self, i):
        self.tgt_particles_x = (self.targets['last_pos'][i])
        self._update_target()
        self.M = self.create_cost_matrix() # 求解最优传输矩阵

    def create_cost_matrix(self):
        # range_param = np.array([[-0.6, 0.6], [0.4, 0.6], [-0.6, 0.6]])
        coords = self.compute_coords([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], [10, 10, 10])

        # Calculate the cost matrix
        M = ot.dist(coords, coords)

        return M

    def compute_coords(self, Area, pixel):
        if not isinstance(pixel, np.ndarray):
            pixel = np.array(pixel)

        # Calculate the step size in each dimension
        step_sizes = [(r[1] - r[0]) / p for r, p in zip(Area, pixel)]

        # Calculate the center of each bin in each dimension
        centers = [np.linspace(r[0] + step / 2, r[1] - step / 2, p) for r, step, p in
                   zip(Area, step_sizes, pixel)]

        # Create a grid of center coordinates
        grid = np.meshgrid(*centers, indexing='ij')

        # Flatten and stack the coordinates
        coords = np.column_stack([g.flatten() for g in grid])

        return coords

    def _update_target(self):
        self.targetDistribution, _ = np.histogramdd(self.tgt_particles_x, bins=[10, 10, 10],
                                                    range=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], density=True)
        self.targetDistribution /= np.sum(self.targetDistribution)

    def compute_step_loss(self, s, f):
        cost = self.compute_wasserstein_loss(s, f)
        self.sum_up_loss_kernel(cost)

    def compute_wasserstein_loss(self, s, f):
        particle_pos = self.particle_x.to_numpy()[f, ...]
        grid, _ = np.histogramdd(particle_pos, bins=[10, 10, 10], range=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], density=True)
        grid /= np.sum(grid)

        cost = ot.emd2(grid.flatten(), self.targetDistribution.flatten(), self.M)
        return cost

    def sum_up_loss_kernel(self, cost):
        self.step_loss.append(cost)

    def get_step_loss(self):
        loss = self.step_loss[self.sim.cur_step_global]
        reward = self.step_loss[self.sim.cur_step_global-1] - self.step_loss[self.sim.cur_step_global]

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info
            