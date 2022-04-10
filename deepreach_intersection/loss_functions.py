import torch
import diff_operators

def initialize_intersection_BRAT(dataset, minWith):
    def intersection_hji(model_output, gt):
        source_target_value = gt['target_loss']
        source_collision_value = gt['collision_loss']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0]
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((60 - 15) / 2)   # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)   # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0]
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)   # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((60 - 15) / 2)   # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # H = lambda^T * (-f) because we invert the time
        # Agent 1's action, detach and show the action as the number
        u1 = -lam11_2.detach()

        # Agent 2's action, detach and show the action as the number
        u2 = -lam22_2.detach()

        max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
        min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

        u1[torch.where(u1 > 0)] = max_acc
        u1[torch.where(u1 < 0)] = min_acc
        u2[torch.where(u2 > 0)] = max_acc
        u2[torch.where(u2 < 0)] = min_acc

        u1.requires_grad = False
        u2.requires_grad = False

        # unnormalize the state for agent 1
        d1 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (60 - 15) / 2 + 15
        v1 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d2 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (60 - 15) / 2 + 15
        v2 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate hamiltonian, H = lambda^T * (-f) because we invert the time
        ham_1 = -lam11_1.squeeze() * v1.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v2.squeeze() - lam12_2.squeeze() * u2.squeeze()
        ham_2 = -lam21_1.squeeze() * v1.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v2.squeeze() - lam22_2.squeeze() * u2.squeeze()

        ham_1 = ham_1.reshape(1, -1)
        ham_2 = ham_2.reshape(1, -1)

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            diff_constraint_hom_1 = dvdt_1 + ham_1
            diff_constraint_hom_2 = dvdt_2 + ham_2
            diff_constraint_hom_1 = torch.max(torch.min(diff_constraint_hom_1[:, :, None],
                                              y1 - source_target_value[:, :y1.shape[1]]),
                                              y1 - source_collision_value[:, :y1.shape[1]])
            diff_constraint_hom_2 = torch.max(torch.min(diff_constraint_hom_2[:, :, None],
                                              y2 - source_target_value[:, y2.shape[1]:]),
                                              y2 - source_collision_value[:, y2.shape[1]:])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # boundary condition check
        dirichlet_1 = y1[dirichlet_mask] - torch.max(source_target_value[:, :y1.shape[1]][dirichlet_mask],
                                                     source_collision_value[:, :y1.shape[1]][dirichlet_mask])
        dirichlet_2 = y2[dirichlet_mask] - torch.max(source_target_value[:, y2.shape[1]:][dirichlet_mask],
                                                     source_collision_value[:, y2.shape[1]:][dirichlet_mask])
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (25e4,100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 25e4,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()/100}
    return intersection_hji

def initialize_intersection_HJI(dataset, minWith):
    def intersection_hji(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]   # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]   # (meta_batch_size, num_points, 1); agent 2's value
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((60 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((60 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).cuda()   # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).cuda()   # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).cuda()   # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).cuda()   # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).cuda()    # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).cuda()    # car length for agent 1
        theta1 = torch.tensor([1.], dtype=torch.float32).cuda()   # behavior for agent 1
        theta2 = torch.tensor([1.], dtype=torch.float32).cuda()   # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).cuda()  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action, detach and let the action as the number, not trainable variable
        # TODO: we consider H = (dV/dt)^T * f - V*L when V = exp(U)
        # TODO: H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        u1 = (0.5 * lam11_2 / y1).detach()

        # Agent 2's action, detach and let the action as the number, not trainable variable
        u2 = (0.5 * lam22_2 / y2).detach()

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
        min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # detach and let the action as the number, not trainable variable
        u1.requires_grad = False
        u2.requires_grad = False

        # unnormalize the state for agent 1
        d1 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (60 - 15) / 2 + 15
        v1 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d2 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (60 - 15) / 2 + 15
        v2 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x1_in = ((d1 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x1_out = (-(d1 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).cuda()
        x2_in = ((d2 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x2_out = (-(d2 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).cuda()

        sigmoid1 = torch.sigmoid(x1_in) * torch.sigmoid(x1_out)
        sigmoid2 = torch.sigmoid(x2_in) * torch.sigmoid(x2_out)
        loss_instant = beta * sigmoid1 * sigmoid2

        # calculate instantaneous loss
        loss_fun_1 = y1 * (u1 ** 2 + loss_instant).cuda()
        loss_fun_2 = y2 * (u2 ** 2 + loss_instant).cuda()

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v1.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v2.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v1.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v2.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            diff_constraint_hom_1 = dvdt_1 + ham_1
            diff_constraint_hom_2 = dvdt_2 + ham_2
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # boundary condition check
        dirichlet_1 = y1[dirichlet_mask] - source_boundary_values[:, :y1.shape[1]][dirichlet_mask]
        dirichlet_2 = y2[dirichlet_mask] - source_boundary_values[:, y2.shape[1]:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (2e4, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 2e5,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 100}
    return intersection_hji
