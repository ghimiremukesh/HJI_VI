import torch
from torch.utils.data import Dataset

class IntersectionBRAT(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = 4

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.goal = 40
        self.beta = 5.5  # collision ratio, math.sqrt(3.75**2 + 3.75**2)

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # observe value curve from ground truth(BVP solution): step function happens within [0.8s, 1.3s].
        # So I consider time window [0, 1.5s] and cut the ground truth data using this time window.
        # Under [0, 1.5s], range of d = [15m, 60m] and range of v = [15m/s, 32m/s]

        # uniformly sample domain and include coordinates for both agents
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords_1 = torch.cat((time, coords_1), dim=1)
            coords_2 = torch.cat((time, coords_2), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                        self.counter / self.full_count))

            coords_1 = torch.cat((time, coords_1), dim=1)
            coords_2 = torch.cat((time, coords_2), dim=1)

            # make sure we always have training samples at the initial time
            coords_1[-self.N_src_samples:, 0] = start_time
            coords_2[-self.N_src_samples:, 0] = start_time

        # set up the goal: l(x) = max(goal-d1, goal-d2) <= 0
        target_loss_1 = torch.max(self.goal - ((coords_1[:, 1:2] + 1) * (60 - 15) / 2 + 15),
                                  self.goal - ((coords_1[:, 3:4] + 1) * (60 - 15) / 2 + 15))
        target_loss_2 = torch.max(self.goal - ((coords_2[:, 1:2] + 1) * (60 - 15) / 2 + 15),
                                  self.goal - ((coords_2[:, 3:4] + 1) * (60 - 15) / 2 + 15))

        target_loss = torch.cat((target_loss_1, target_loss_2), dim=0)

        # set up the collision area: g(x) = beta-||(d1, d2)|| > 0
        coords_1_g = torch.cat((((coords_1[:, 1:2] + 1) * (60 - 15) / 2 + 15),
                                ((coords_1[:, 3:4] + 1) * (60 - 15) / 2 + 15)), dim=1)
        coords_2_g = torch.cat((((coords_2[:, 1:2] + 1) * (60 - 15) / 2 + 15),
                                ((coords_2[:, 3:4] + 1) * (60 - 15) / 2 + 15)), dim=1)

        # consider heading distance of two vehicle in 2D frame, sqrt((d1-35)^2+(d2-35)^2)
        boundary_values_g_1 = self.beta - torch.norm(coords_1_g - 35, dim=1, keepdim=True)
        boundary_values_g_2 = self.beta - torch.norm(coords_2_g - 35, dim=1, keepdim=True)

        boundary_values_g = torch.cat((boundary_values_g_1, boundary_values_g_2), dim=0)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords_1.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords_1[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords},  {'target_loss': target_loss, 'collision_loss': boundary_values_g,
                                     'dirichlet_mask': dirichlet_mask}

class IntersectionHJI(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=3, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, num_src_samples=1000, seed=0):

        super().__init__()
        torch.manual_seed(0)

        ## TODO: add belief state

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.num_states = 4 # 4 (physical states) + 2 (belief states)
        self.belief_dim = 1 # add to the init after
        self.prior = 0.5
        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end
        self.alpha = 1e-6

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # observe value curve from ground truth(BVP solution): step function happens within [0.8s, 1.3s].
        # So I consider time window [0, 1.5s] and cut the ground truth data using this time window.
        # Under [0, 1.5s], range of d = [15m, 60m] and range of v = [15m/s, 32m/s]

        # uniformly sample domain and include coordinates for both agents (and beliefs)
        coords_1 = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)
        coords_2 = torch.cat((coords_1[:, 2:], coords_1[:, :2]), dim=1)
        belief_coords_1 = torch.zeros(self.numpoints, self.belief_dim).uniform_(0, 1)  # belief state sample uniformly all priors - agent 1
        belief_coords_2 = torch.zeros(self.numpoints, self.belief_dim).uniform_(0, 1)  # belief state sample uniformly all priors - agent 2
        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            # beta_1 = torch.ones(self.numpoints, 1) * self.prior  # agent 1's belief of agent 2
            # beta_2 = torch.ones(self.numpoints, 1) * self.prior  # agent 2's belief of agent 1
            # coords_1 = torch.cat((time, coords_1, beta_1), dim=1)
            # coords_2 = torch.cat((time, coords_2, beta_2), dim=1)
            coords_1 = torch.cat((time, coords_1, belief_coords_1, belief_coords_2), dim=1)
            coords_2 = torch.cat((time, coords_2, belief_coords_2, belief_coords_1), dim=1)

        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (
                    self.counter / self.full_count))

            coords_1 = torch.cat((time, coords_1, belief_coords_1, belief_coords_2), dim=1)
            coords_2 = torch.cat((time, coords_2, belief_coords_2, belief_coords_1), dim=1)

            # make sure we always have training samples at the initial time
            coords_1[-self.N_src_samples:, 0] = start_time
            coords_2[-self.N_src_samples:, 0] = start_time

        # set up boundary condition: V(T) = alpha*d(T) - (v(T) - v(0))^2
        boundary_values_1 = self.alpha * ((coords_1[:, 1:2] + 1) * (60 - 15) / 2 + 15) - \
                            ((coords_1[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values_2 = self.alpha * ((coords_2[:, 1:2] + 1) * (60 - 15) / 2 + 15) - \
                            ((coords_2[:, 2:3] + 1) * (32 - 15) / 2 + 15 - 18) ** 2
        boundary_values = torch.cat((boundary_values_1, boundary_values_2), dim=0)

        if self.pretrain:
            dirichlet_mask = torch.ones(coords_1.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords_1[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        coords = torch.cat((coords_1, coords_2), dim=0)
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}