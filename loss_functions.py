import torch
import diff_operators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        lam11_1 = dvdx_1[:, :1] / ((60 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0]
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((60 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # H = lambda^T * (-f) because we invert the time
        # Agent 1's action, detach and show the action as the number
        u1 = -lam11_2.detach()

        # Agent 2's action, detach and show the action as the number
        u2 = -lam22_2.detach()

        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

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
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 100}

    return intersection_hji


def initialize_intersection_HJI_selfsupervised(dataset, gamma):
    def intersection_hji(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
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
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        theta1 = torch.tensor([1.], dtype=torch.float32).to(device)  # behavior for agent 1
        theta2 = torch.tensor([1.], dtype=torch.float32).to(device)  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action, detach and let the action as the number, not trainable variable
        # TODO: we consider H = (dV/dt)^T * f - V*L when V = exp(U)
        # TODO: H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        # u1 = (0.5 * lam11_2 / y1).detach()
        u1 = 0.5 * lam11_2 * 10

        # Agent 2's action, detach and let the action as the number, not trainable variable
        # u2 = (0.5 * lam22_2 / y2).detach()
        u2 = 0.5 * lam22_2 * 10

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # detach and let the action as the number, not trainable variable
        # u1.requires_grad = False
        # u2.requires_grad = False

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in = ((d11 - R1 / 2 + theta1 * W2 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x11_out = (-(d11 - R1 / 2 - W2 / 2 - L1) * gamma).squeeze().reshape(-1, 1).to(device)
        x12_in = ((d12 - R2 / 2 + theta2 * W1 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x12_out = (-(d12 - R2 / 2 - W1 / 2 - L2) * gamma).squeeze().reshape(-1, 1).to(device)

        sigmoid11 = torch.sigmoid(x11_in) * torch.sigmoid(x11_out)
        sigmoid12 = torch.sigmoid(x12_in) * torch.sigmoid(x12_out)
        loss_instant1 = beta * sigmoid11 * sigmoid12

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds

        x21_in = ((d21 - R1 / 2 + theta1 * W2 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x21_out = (-(d21 - R1 / 2 - W2 / 2 - L1) * gamma).squeeze().reshape(-1, 1).to(device)
        x22_in = ((d22 - R2 / 2 + theta2 * W1 / 2) * gamma).squeeze().reshape(-1, 1).to(device)
        x22_out = (-(d22 - R2 / 2 - W1 / 2 - L2) * gamma).squeeze().reshape(-1, 1).to(device)

        sigmoid21 = torch.sigmoid(x21_in) * torch.sigmoid(x21_out)
        sigmoid22 = torch.sigmoid(x22_in) * torch.sigmoid(x22_out)
        loss_instant2 =  beta * sigmoid21 * sigmoid22

        # calculate instantaneous loss
        loss_fun_1 = 0.1 * (u1 ** 2 + loss_instant1)
        loss_fun_2 = 0.1 * (u2 ** 2 + loss_instant2)

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

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
        dirichlet_1 = y1[dirichlet_mask] - 0.1 * source_boundary_values[:, :y1.shape[1]][dirichlet_mask]
        dirichlet_2 = y2[dirichlet_mask] - 0.1 * source_boundary_values[:, y2.shape[1]:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() / 10,  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 200}

    return intersection_hji


def initialize_intersection_HJI_supervised(dataset, minWith):
    def intersection_hji(model_output, gt):
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        theta1 = torch.tensor([1.], dtype=torch.float32).to(device)  # behavior for agent 1
        theta2 = torch.tensor([1.], dtype=torch.float32).to(device)  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action, detach and let the action as the number, not trainable variable
        # TODO: we consider H = (dV/dt)^T * f - V*L when V = exp(U)
        # TODO: H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        # u1 = (0.5 * lam11_2 / y1).detach()
        u1 = 0.5 * lam11_2

        # Agent 2's action, detach and let the action as the number, not trainable variable
        # u2 = (0.5 * lam22_2 / y2).detach()
        u2 = 0.5 * lam22_2

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # detach and let the action as the number, not trainable variable
        # u1.requires_grad = False
        # u2.requires_grad = False

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in = ((d11 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out = (-(d11 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in = ((d12 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out = (-(d12 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11 = torch.sigmoid(x11_in) * torch.sigmoid(x11_out)
        sigmoid12 = torch.sigmoid(x12_in) * torch.sigmoid(x12_out)
        loss_instant1 = beta * sigmoid11 * sigmoid12

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x21_in = ((d21 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out = (-(d21 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in = ((d22 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out = (-(d22 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21 = torch.sigmoid(x21_in) * torch.sigmoid(x21_out)
        sigmoid22 = torch.sigmoid(x22_in) * torch.sigmoid(x22_out)
        loss_instant2 = beta * sigmoid21 * sigmoid22

        # calculate instantaneous loss
        loss_fun_1 = u1 ** 2 + loss_instant1
        loss_fun_2 = u2 ** 2 + loss_instant2

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        diff_constraint_hom_1 = dvdt_1 + ham_1
        diff_constraint_hom_2 = dvdt_2 + ham_2
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # supervised learning for values
        value1_difference = y1 - groundtruth_values[:, :y1.shape[1]]
        value2_difference = y2 - groundtruth_values[:, y2.shape[1]:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1, lam11_2, lam12_1, lam12_2), dim=1)
        costate2_prediction = torch.cat((lam21_1, lam21_2, lam22_1, lam22_2), dim=1)
        costate1_difference = costate1_prediction - groundtruth_costates[:, :y1.shape[1]].squeeze()
        costate2_difference = costate2_prediction - groundtruth_costates[:, y2.shape[1]:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / 40,  # 40
                'costates_difference': torch.abs(costates_difference).sum() / 120}

    return intersection_hji


def initialize_intersection_HJI_hyrid(dataset, minWith):
    def intersection_hji(model_output, gt):
        groundtruth_values = gt['groundtruth_values']
        groundtruth_costates = gt['groundtruth_costates']
        source_boundary_values = gt['source_boundary_values']
        dirichlet_mask = gt['dirichlet_mask']
        x = model_output['model_in']
        y = model_output['model_out']
        cut_index = x.shape[1] // 2
        supervised_index = groundtruth_values.shape[1] // 2
        hji_index = source_boundary_values.shape[1] // 2

        y1 = model_output['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2 = model_output['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv_1 = jac[:, :cut_index, :]
        dv_2 = jac[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1 = dv_1[..., 0, 0].squeeze()
        dvdx_1 = dv_1[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1 = dvdx_1[:, :1] / ((105 - 15) / 2)  # lambda_11
        lam11_2 = dvdx_1[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1 = dvdx_1[:, 2:3] / ((105 - 15) / 2)  # lambda_12
        lam12_2 = dvdx_1[:, 3:] / ((32 - 15) / 2)  # lambda_12

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2 = dv_2[..., 0, 0].squeeze()
        dvdx_2 = dv_2[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1 = dvdx_2[:, 2:3] / ((105 - 15) / 2)  # lambda_21
        lam21_2 = dvdx_2[:, 3:] / ((32 - 15) / 2)  # lambda_21
        lam22_1 = dvdx_2[:, :1] / ((105 - 15) / 2)  # lambda_22
        lam22_2 = dvdx_2[:, 1:2] / ((32 - 15) / 2)  # lambda_22

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).to(device)  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).to(device)  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).to(device)  # car length for agent 1
        theta1 = torch.tensor([1.], dtype=torch.float32).to(device)  # behavior for agent 1
        theta2 = torch.tensor([1.], dtype=torch.float32).to(device)  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).to(device)  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action, detach and let the action as the number, not trainable variable
        # TODO: we consider H = (dV/dt)^T * f - V*L when V = exp(U)
        # TODO: H = (dV/dt)^T * (-f) + V*L when inverting the time, optimal action u = 1/2 * B^T * lambda / V
        # u1 = (0.5 * lam11_2 / y1).detach()
        u1 = 0.5 * lam11_2

        # Agent 2's action, detach and let the action as the number, not trainable variable
        # u2 = (0.5 * lam22_2 / y2).detach()
        u2 = 0.5 * lam22_2

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).to(device)
        min_acc = torch.tensor([-5.], dtype=torch.float32).to(device)

        u1[torch.where(u1 > max_acc)] = max_acc
        u1[torch.where(u1 < min_acc)] = min_acc
        u2[torch.where(u2 > max_acc)] = max_acc
        u2[torch.where(u2 < min_acc)] = min_acc

        # detach and let the action as the number, not trainable variable
        # u1.requires_grad = False
        # u2.requires_grad = False

        # unnormalize the state for agent 1
        d11 = (model_output['model_in'][:, :cut_index, 1:2] + 1) * (105 - 15) / 2 + 15
        v11 = (model_output['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d12 = (model_output['model_in'][:, :cut_index, 3:4] + 1) * (105 - 15) / 2 + 15
        v12 = (model_output['model_in'][:, :cut_index, 4:] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x11_in = ((d11 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x11_out = (-(d11 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x12_in = ((d12 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x12_out = (-(d12 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid11 = torch.sigmoid(x11_in) * torch.sigmoid(x11_out)
        sigmoid12 = torch.sigmoid(x12_in) * torch.sigmoid(x12_out)
        loss_instant1 = beta * sigmoid11 * sigmoid12

        # unnormalize the state for agent 1
        d21 = (model_output['model_in'][:, cut_index:, 3:4] + 1) * (105 - 15) / 2 + 15
        v21 = (model_output['model_in'][:, cut_index:, 4:] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d22 = (model_output['model_in'][:, cut_index:, 1:2] + 1) * (105 - 15) / 2 + 15
        v22 = (model_output['model_in'][:, cut_index:, 2:3] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds
        x21_in = ((d21 - R1 / 2 + theta1 * W2 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x21_out = (-(d21 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).to(device)
        x22_in = ((d22 - R2 / 2 + theta2 * W1 / 2) * 5).squeeze().reshape(-1, 1).to(device)
        x22_out = (-(d22 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).to(device)

        sigmoid21 = torch.sigmoid(x21_in) * torch.sigmoid(x21_out)
        sigmoid22 = torch.sigmoid(x22_in) * torch.sigmoid(x22_out)
        loss_instant2 = beta * sigmoid21 * sigmoid22

        # calculate instantaneous loss
        loss_fun_1 = u1 ** 2 + loss_instant1
        loss_fun_2 = u2 ** 2 + loss_instant2

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        ham_1 = -lam11_1.squeeze() * v11.squeeze() - lam11_2.squeeze() * u1.squeeze() - \
                lam12_1.squeeze() * v12.squeeze() - lam12_2.squeeze() * u2.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1.squeeze() * v21.squeeze() - lam21_2.squeeze() * u1.squeeze() - \
                lam22_1.squeeze() * v22.squeeze() - lam22_2.squeeze() * u2.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        diff_constraint_hom_1 = dvdt_1 + ham_1
        diff_constraint_hom_2 = dvdt_2 + ham_2
        diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # supervised learning for values
        value1_difference = y1[:, :supervised_index] - groundtruth_values[:, :supervised_index]
        value2_difference = y2[:, :supervised_index] - groundtruth_values[:, supervised_index:]
        values_difference = torch.cat((value1_difference, value2_difference), dim=0)

        # supervised learning for costates
        costate1_prediction = torch.cat((lam11_1[:supervised_index, :],
                                         lam11_2[:supervised_index, :],
                                         lam12_1[:supervised_index, :],
                                         lam12_2[:supervised_index, :]), dim=1)
        costate2_prediction = torch.cat((lam21_1[:supervised_index, :],
                                         lam21_2[:supervised_index, :],
                                         lam22_1[:supervised_index, :],
                                         lam22_2[:supervised_index, :]), dim=1)
        costate1_difference = costate1_prediction - groundtruth_costates[:, :supervised_index].squeeze()
        costate2_difference = costate2_prediction - groundtruth_costates[:, supervised_index:].squeeze()
        costates_difference = torch.cat((costate1_difference, costate2_difference), dim=0)

        # boundary condition check
        dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, :hji_index][dirichlet_mask]
        dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, hji_index:][dirichlet_mask]
        # dirichlet_1 = y1[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, :hji_index][dirichlet_mask]
        # dirichlet_2 = y2[:, supervised_index:][dirichlet_mask] - source_boundary_values[:, hji_index:][dirichlet_mask]
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'values_difference': torch.abs(values_difference).sum() / 40,  # 1e4
                'costates_difference': torch.abs(costates_difference).sum() / 120,
                'dirichlet': torch.norm(torch.abs(dirichlet)),  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 200}

    return intersection_hji


def SSL_1D(dataset):
    import math
    def toy(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']
        # delta = gt['delta']

        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv = jac[:, :, :]

        # agent 1: partial gradient of V w.r.t. time and state
        # dvdt = dv[..., 0, 0].squeeze()
        dvdx = dv[..., 0, 0].squeeze()

        alpha = torch.tensor(1)  # for delta function approx. lim_{\alpha}\goesto zero
        delta = (1 / math.pi) * (alpha / (torch.square(x) + torch.square(alpha)))
        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check

        diff_constraint_hom = dvdx.reshape(-1, 1) - delta
        diff_constraint_hom[:, 0, :] = 0
        # if torch.all(dirichlet_mask):
        #     diff_constraint_hom = torch.Tensor([0])
        # else:
        #     diff_constraint_hom = dvdx - delta

        # boundary condition check
        dirichlet = y[dirichlet_mask] - source_boundary_values[:][dirichlet_mask]

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum(),  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return toy


def Sup_1D(dataset):
    def toy_s(model_output, gt):
        gt_value = gt['gt_values']
        v_pred = model_output['model_out']
        x = model_output['model_in']
        dvdx, _ = diff_operators.jacobian(v_pred, x)

        return {'values_difference': torch.abs(gt_value - v_pred).sum(),
                'costates_difference': torch.abs(dvdx).sum()}

    return toy_s

def Hybrid_1D(dataset):
    import math
    def toy_h(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']
        y = model_output['model_out']

        gt_values = gt['gt_values']
        sup_index = 2 # not equal anymore
        # delta = gt['delta']

        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac, _ = diff_operators.jacobian(y, x)
        dv = jac[:, :, :]

        # agent 1: partial gradient of V w.r.t. time and state
        # dvdt = dv[..., 0, 0].squeeze()
        dvdx = dv[..., 0, 0].squeeze()

        alpha = torch.tensor(1e-7)  # for delta function approx. lim_{\alpha}\goesto zero
        delta = (1 / math.pi) * (alpha / (torch.square(x) + torch.square(alpha)))
        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        # idx = torch.where(dirichlet_mask)[0].detach().cpu().numpy()[0] # will always be at 0

        diff_constraint_hom = dvdx.reshape(-1, 1) - delta
        diff_constraint_hom[:, 0, :] = 0
        # if torch.all(dirichlet_mask):
        #     diff_constraint_hom = torch.Tensor([0])
        # else:
        #     diff_constraint_hom = dvdx - delta

        # boundary condition check
        dirichlet = y[:, :sup_index][dirichlet_mask] - source_boundary_values[:][dirichlet_mask]

        # supervised part
        value_difference = y[:, sup_index:] - gt_values
        costate_difference = dvdx[sup_index:].reshape(-1, 1)


        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum(),  # 1e4
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum(),
                'values_difference': torch.abs(value_difference).sum(),
                'costate_difference': torch.abs(costate_difference).sum()}

    return toy_h