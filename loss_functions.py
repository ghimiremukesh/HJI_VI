import torch
import diff_operators



def initialize_soccer_hji(dataset):
    def soccer_hji(model_output, gt):

        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']


        y = model_output['model_out']  # (meta_batch_size, num_points, 1); value
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)


        # partial gradient of V w.r.t. time and state
        dvdt = jac[..., 0, 0].squeeze()
        dvdx = jac[..., 0, 1:].squeeze()


        # unnormalize the costate for agent 1
        lam_1 = dvdx[:, :1]
        lam_2 = dvdx[:, 1:2]
        lam_4 = dvdx[:, 2:3]
        lam_5 = dvdx[:, 3:4]
        lam_6 = dvdx[:, 4:]




        # H = lambda^T * (-f) + L because we invert the time
        u = dataset.uMax * torch.sign(lam_2)
        d = dataset.dMax * torch.sign(lam_5)

        v1 = x[:, :, 2]
        v2 = x[:, :, 4]


        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham = -lam_1.squeeze() * v1.squeeze() - lam_2.squeeze() * u.squeeze()- \
                lam_4.squeeze() * v2.squeeze() - lam_5.squeeze() * d.squeeze() - lam_6 * torch.sign(u.squeeze())


        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dvdt + torch.minimum(torch.tensor([0]), ham)

        # boundary condition check
        dirichlet = y[dirichlet_mask] -  source_boundary_values[dirichlet_mask]


        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum(),  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return soccer_hji

