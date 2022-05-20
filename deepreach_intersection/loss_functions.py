import torch
import diff_operators


def initialize_intersection_HJI_a(dataset, minWith):
    def intersection_hji(model_output_a, model_output_na, gt):
        # Dynamics w/belief (p)
        # \dot x = v
        # \dot v = u
        # xxxx[\dot p = sign(2p - 1), p = belief  ---> old formulation]xxx
        # \dot p_i = sign(H_{-i}(a) - H_{-i}(na)) * \alpha   # belief dynamics is a function of other agent's
        #                                                                                           hamiltonian

        source_boundary_values = gt['source_boundary_values']
        x = model_output_a['model_in']
        x_na = model_output_na['model_in']  # to remove autograd issue
        y_a = model_output_a['model_out']
        y_na = model_output_na['model_out']
        cut_index = x.shape[1] // 2  # V1 and V2
        p1 = x[:, :cut_index, 5]  # P2's belief about P1
        p2 = x[:, :cut_index, 6]  # P1's belief about P2

        # Get Aggressive First and Then Non-Aggressive
        y1_a = model_output_a['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2_a = model_output_a['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        y1_na = model_output_na['model_out'][:, :cut_index]
        y2_na = model_output_na['model_out'][:, cut_index:]
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac_a, _ = diff_operators.jacobian(y_a, x)
        dv_1_a = jac_a[:, :cut_index, :]
        dv_2_a = jac_a[:, cut_index:, :]

        # non-agg
        jac_na, _ = diff_operators.jacobian(y_na, x_na)
        dv_1_na = jac_na[:, :cut_index, :]
        dv_2_na = jac_na[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1_a = dv_1_a[..., 0, 0].squeeze()
        dvdx_1_a = dv_1_a[..., 0, 1:].squeeze()

        # non-agg
        dvdt_1_na = dv_1_na[..., 0, 0].squeeze()
        dvdx_1_na = dv_1_na[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1_a = dvdx_1_a[:, :1] / ((60 - 15) / 2)  # lambda_11
        lam11_2_a = dvdx_1_a[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1_a = dvdx_1_a[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2_a = dvdx_1_a[:, 3:4] / ((32 - 15) / 2)  # lambda_12
        lam11_3_a = dvdx_1_a[:, 4:5]  # lambda_11_p1 (costate corr. to belief)
        lam12_3_a = dvdx_1_a[:, 5:6]  # lambda_11_p2 (costate corr. to belief)

        # non-agg
        lam11_1_na = dvdx_1_na[:, :1] / ((60 - 15) / 2)  # lambda_11
        lam11_2_na = dvdx_1_na[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1_na = dvdx_1_na[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2_na = dvdx_1_na[:, 3:4] / ((32 - 15) / 2)  # lambda_12
        lam11_3_na = dvdx_1_na[:, 4:5]  # lambda_11_p1 (costate corr. to belief)
        lam12_3_na = dvdx_1_na[:, 5:6]  # lambda_11_p2 (costate corr. to belief)

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2_a = dv_2_a[..., 0, 0].squeeze()
        dvdx_2_a = dv_2_a[..., 0, 1:].squeeze()

        # non-agg
        dvdt_2_na = dv_2_na[..., 0, 0].squeeze()
        dvdx_2_na = dv_2_na[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1_a = dvdx_2_a[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2_a = dvdx_2_a[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1_a = dvdx_2_a[:, :1] / ((60 - 15) / 2)  # lambda_22
        lam22_2_a = dvdx_2_a[:, 1:2] / ((32 - 15) / 2)  # lambda_22
        lam21_3_a = dvdx_2_a[:, 5:6]  # lambda_21 (belief costate)
        lam22_3_a = dvdx_2_a[:, 4:5]

        # non-agg
        lam21_1_na = dvdx_2_na[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2_na = dvdx_2_na[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1_na = dvdx_2_na[:, :1] / ((60 - 15) / 2)  # lambda_22
        lam22_2_na = dvdx_2_na[:, 1:2] / ((32 - 15) / 2)  # lambda_22
        lam21_3_na = dvdx_2_na[:, 5:6]  # lambda_21 (belief costate)
        lam22_3_na = dvdx_2_na[:, 4:5]

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
        theta_a = torch.tensor([1.], dtype=torch.float32).cuda()  # behavior for agent 1
        theta_na = torch.tensor([5.], dtype=torch.float32).cuda()  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).cuda()  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action, detach and let the action as the number, not trainable variable

        u1_a = (0.5 * lam11_2_a).detach()
        u1_na = (0.5 * lam11_2_na).detach()

        # Agent 2's action, detach and let the action as the number, not trainable variable
        u2_a = (0.5 * lam22_2_a).detach()
        u2_na = (0.5 * lam22_2_na).detach()

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
        min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

        u1_a[torch.where(u1_a > max_acc)] = max_acc
        u1_a[torch.where(u1_a < min_acc)] = min_acc
        u2_a[torch.where(u2_a > max_acc)] = max_acc
        u2_a[torch.where(u2_a < min_acc)] = min_acc

        # non-agg
        u1_na[torch.where(u1_na > max_acc)] = max_acc
        u1_na[torch.where(u1_na < min_acc)] = min_acc
        u2_na[torch.where(u2_na > max_acc)] = max_acc
        u2_na[torch.where(u2_na < min_acc)] = min_acc

        # detach and let the action as the number, not trainable variable
        u1_a.requires_grad = False
        u2_a.requires_grad = False

        # non-agg
        u1_na.requires_grad = False
        u2_na.requires_grad = False

        # unnormalize the state for agent 1
        d1 = (model_output_a['model_in'][:, :cut_index, 1:2] + 1) * (60 - 15) / 2 + 15
        v1 = (model_output_a['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d2 = (model_output_a['model_in'][:, :cut_index, 3:4] + 1) * (60 - 15) / 2 + 15
        v2 = (model_output_a['model_in'][:, :cut_index, 4:5] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds (aggressive)
        x1_in = ((d1 - R1 / 2 + theta_a * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x1_out = (-(d1 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).cuda()
        x2_in = ((d2 - R2 / 2 + theta_a * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x2_out = (-(d2 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).cuda()

        sigmoid1 = torch.sigmoid(x1_in) * torch.sigmoid(x1_out)
        sigmoid2 = torch.sigmoid(x2_in) * torch.sigmoid(x2_out)
        loss_instant = beta * sigmoid1 * sigmoid2

        # calculate the collision area lower and upper bounds (non-aggressive)
        x1_in_na = ((d1 - R1 / 2 + theta_na * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x2_in_na = ((d2 - R2 / 2 + theta_na * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()

        sigmoid1_na = torch.sigmoid(x1_in_na) * torch.sigmoid(x1_out)
        sigmoid2_na = torch.sigmoid(x2_in_na) * torch.sigmoid(x2_out)
        loss_instant_na = beta * sigmoid1_na * sigmoid2_na

        # calculate instantaneous loss
        loss_fun_1 = p1.T * (u1_a ** 2 + loss_instant.cuda()) + (1 - p1).T * (
                y1_a * (u1_a ** 2 + loss_instant_na.cuda()))
        loss_fun_2 = p2.T * (u2_a ** 2 + loss_instant.cuda()) + (1 - p2).T * (
                y2_a * (u2_a ** 2 + loss_instant_na.cuda()))

        # calculate ham for agents for belief dynamics:
        # H = \lambda^T * (-f) + L

        H_1_a = -lam11_1_a.squeeze() * v1.squeeze() - lam11_2_a.squeeze() * u1_a.squeeze() - \
                lam12_1_a.squeeze() * v2.squeeze() - lam12_2_a.squeeze() * u2_a.squeeze() + \
                (u1_a ** 2 + loss_instant.cuda())
        H_1_na = -lam11_1_na.squeeze() * v1.squeeze() - lam11_2_na.squeeze() * u1_na.squeeze() - \
                lam12_1_na.squeeze() * v2.squeeze() - lam12_2_na.squeeze() * u2_na.squeeze() + \
                (u1_na ** 2 + loss_instant.cuda())

        H_2_a = -lam21_1_a.squeeze() * v1.squeeze() - lam21_2_a.squeeze() * u1_a.squeeze() - \
                lam22_1_a.squeeze() * v2.squeeze() - lam22_2_a.squeeze() * u2_a.squeeze() + \
                (u2_a ** 2 + loss_instant.cuda())
        H_2_na = -lam21_1_na.squeeze() * v1.squeeze() - lam21_2_na.squeeze() * u1_na.squeeze() - \
                lam22_1_na.squeeze() * v2.squeeze() - lam22_2_na.squeeze() * u2_na.squeeze() + \
                (u2_na ** 2 + loss_instant_na.cuda())



        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        # set alpha = const (learning rate) = 0.01 (for eg)
        alpha = 0.01
        ham_1 = -lam11_1_a.squeeze() * v1.squeeze() - lam11_2_a.squeeze() * u1_a.squeeze() - lam11_3_a.squeeze() * torch.sign(
            H_2_a - H_2_na) * alpha - \
                lam12_1_a.squeeze() * v2.squeeze() - lam12_2_a.squeeze() * u2_a.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1_a.squeeze() * v1.squeeze() - lam21_2_a.squeeze() * u1_a.squeeze() - lam21_3_a.squeeze() * torch.sign(
            H_1_a - H_1_na) * alpha - \
                lam22_1_a.squeeze() * v2.squeeze() - lam22_2_a.squeeze() * u2_a.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            diff_constraint_hom_1 = dvdt_1_a + ham_1
            diff_constraint_hom_2 = dvdt_2_a + ham_2
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # boundary condition check
        dirichlet_1 = y1_a[dirichlet_mask] - (source_boundary_values[:, :y1_a.shape[1]][dirichlet_mask])
        dirichlet_2 = y2_a[dirichlet_mask] - (source_boundary_values[:, y2_a.shape[1]:][dirichlet_mask])
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 2e7,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 100}

    return intersection_hji


def initialize_intersection_HJI_na(dataset, minWith):
    def intersection_hji(model_output_a, model_output_na, gt):
        # Dynamics w/belief (p)
        # \dot x = v
        # \dot v = u
        # xxxx[\dot p = sign(2p - 1), p = belief  ---> old formulation]xxx
        # \dot p_i = sign(H_{-i}(a) - H_{-i}(na)) * \alpha   # belief dynamics is a function of other agent's
        #                                                                                           hamiltonian

        source_boundary_values = gt['source_boundary_values']
        x = model_output_a['model_in']
        x_na = model_output_na['model_in']
        y_a = model_output_a['model_out']
        y_na = model_output_na['model_out']
        cut_index = x.shape[1] // 2  # V1 and V2
        p1 = x[:, :cut_index, 5]  # P2's belief about P1
        p2 = x[:, :cut_index, 6]  # P1's belief about P2

        # Get Aggressive First and Then Non-Aggressive
        y1_a = model_output_a['model_out'][:, :cut_index]  # (meta_batch_size, num_points, 1); agent 1's value
        y2_a = model_output_a['model_out'][:, cut_index:]  # (meta_batch_size, num_points, 1); agent 2's value
        y1_na = model_output_na['model_out'][:, :cut_index]
        y2_na = model_output_na['model_out'][:, cut_index:]
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        # calculate the partial gradient of V w.r.t. time and state
        jac_a, _ = diff_operators.jacobian(y_a, x)
        dv_1_a = jac_a[:, :cut_index, :]
        dv_2_a = jac_a[:, cut_index:, :]

        # non-agg
        jac_na, _ = diff_operators.jacobian(y_na, x_na)
        dv_1_na = jac_na[:, :cut_index, :]
        dv_2_na = jac_na[:, cut_index:, :]

        # agent 1: partial gradient of V w.r.t. time and state
        dvdt_1_a = dv_1_a[..., 0, 0].squeeze()
        dvdx_1_a = dv_1_a[..., 0, 1:].squeeze()

        # non-agg
        dvdt_1_na = dv_1_na[..., 0, 0].squeeze()
        dvdx_1_na = dv_1_na[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 1
        lam11_1_a = dvdx_1_a[:, :1] / ((60 - 15) / 2)  # lambda_11
        lam11_2_a = dvdx_1_a[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1_a = dvdx_1_a[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2_a = dvdx_1_a[:, 3:4] / ((32 - 15) / 2)  # lambda_12
        lam11_3_a = dvdx_1_a[:, 4:5]  # lambda_11_p1 (costate corr. to belief)
        lam12_3_a = dvdx_1_a[:, 5:6]  # lambda_11_p2 (costate corr. to belief)

        # non-agg
        lam11_1_na = dvdx_1_na[:, :1] / ((60 - 15) / 2)  # lambda_11
        lam11_2_na = dvdx_1_na[:, 1:2] / ((32 - 15) / 2)  # lambda_11
        lam12_1_na = dvdx_1_na[:, 2:3] / ((60 - 15) / 2)  # lambda_12
        lam12_2_na = dvdx_1_na[:, 3:4] / ((32 - 15) / 2)  # lambda_12
        lam11_3_na = dvdx_1_na[:, 4:5]  # lambda_11_p1 (costate corr. to belief)
        lam12_3_na = dvdx_1_na[:, 5:6]  # lambda_11_p2 (costate corr. to belief)

        # agent 2: partial gradient of V w.r.t. time and state
        dvdt_2_a = dv_2_a[..., 0, 0].squeeze()
        dvdx_2_a = dv_2_a[..., 0, 1:].squeeze()

        # non-agg
        dvdt_2_na = dv_2_na[..., 0, 0].squeeze()
        dvdx_2_na = dv_2_na[..., 0, 1:].squeeze()

        # unnormalize the costate for agent 2
        lam21_1_a = dvdx_2_a[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2_a = dvdx_2_a[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1_a = dvdx_2_a[:, :1] / ((60 - 15) / 2)  # lambda_22
        lam22_2_a = dvdx_2_a[:, 1:2] / ((32 - 15) / 2)  # lambda_22
        lam21_3_a = dvdx_2_a[:, 5:6]  # lambda_21 (belief costate)
        lam22_3_a = dvdx_2_a[:, 4:5]

        # non-agg
        lam21_1_na = dvdx_2_na[:, 2:3] / ((60 - 15) / 2)  # lambda_21
        lam21_2_na = dvdx_2_na[:, 3:4] / ((32 - 15) / 2)  # lambda_21
        lam22_1_na = dvdx_2_na[:, :1] / ((60 - 15) / 2)  # lambda_22
        lam22_2_na = dvdx_2_na[:, 1:2] / ((32 - 15) / 2)  # lambda_22
        lam21_3_na = dvdx_2_na[:, 5:6]  # lambda_21 (belief costate)
        lam22_3_na = dvdx_2_na[:, 4:5]

        # calculate the collision area for aggressive-aggressive case
        R1 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 1
        R2 = torch.tensor([70.], dtype=torch.float32).cuda()  # road length for agent 2
        W1 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
        W2 = torch.tensor([1.5], dtype=torch.float32).cuda()  # car width for agent 1
        L1 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
        L2 = torch.tensor([3.], dtype=torch.float32).cuda()  # car length for agent 1
        theta_a = torch.tensor([1.], dtype=torch.float32).cuda()  # behavior for agent 1
        theta_na = torch.tensor([5.], dtype=torch.float32).cuda()  # behavior for agent 2
        beta = torch.tensor([10000.], dtype=torch.float32).cuda()  # collision ratio

        # H = lambda^T * (-f) + L because we invert the time
        # Agent 1's action, detach and let the action as the number, not trainable variable

        u1_a = (0.5 * lam11_2_a).detach()
        u1_na = (0.5 * lam11_2_na).detach()

        # Agent 2's action, detach and let the action as the number, not trainable variable
        u2_a = (0.5 * lam22_2_a).detach()
        u2_na = (0.5 * lam22_2_na).detach()

        # set up bounds for u1 and u2
        max_acc = torch.tensor([10.], dtype=torch.float32).cuda()
        min_acc = torch.tensor([-5.], dtype=torch.float32).cuda()

        u1_a[torch.where(u1_a > max_acc)] = max_acc
        u1_a[torch.where(u1_a < min_acc)] = min_acc
        u2_a[torch.where(u2_a > max_acc)] = max_acc
        u2_a[torch.where(u2_a < min_acc)] = min_acc

        # non-agg
        u1_na[torch.where(u1_na > max_acc)] = max_acc
        u1_na[torch.where(u1_na < min_acc)] = min_acc
        u2_na[torch.where(u2_na > max_acc)] = max_acc
        u2_na[torch.where(u2_na < min_acc)] = min_acc

        # detach and let the action as the number, not trainable variable
        u1_a.requires_grad = False
        u2_a.requires_grad = False

        # non-agg
        u1_na.requires_grad = False
        u2_na.requires_grad = False

        # unnormalize the state for agent 1
        d1 = (model_output_a['model_in'][:, :cut_index, 1:2] + 1) * (60 - 15) / 2 + 15
        v1 = (model_output_a['model_in'][:, :cut_index, 2:3] + 1) * (32 - 15) / 2 + 15

        # unnormalize the state for agent 2
        d2 = (model_output_a['model_in'][:, :cut_index, 3:4] + 1) * (60 - 15) / 2 + 15
        v2 = (model_output_a['model_in'][:, :cut_index, 4:5] + 1) * (32 - 15) / 2 + 15

        # calculate the collision area lower and upper bounds (aggressive)
        x1_in = ((d1 - R1 / 2 + theta_a * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x1_out = (-(d1 - R1 / 2 - W2 / 2 - L1) * 5).squeeze().reshape(-1, 1).cuda()
        x2_in = ((d2 - R2 / 2 + theta_a * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x2_out = (-(d2 - R2 / 2 - W1 / 2 - L2) * 5).squeeze().reshape(-1, 1).cuda()

        sigmoid1 = torch.sigmoid(x1_in) * torch.sigmoid(x1_out)
        sigmoid2 = torch.sigmoid(x2_in) * torch.sigmoid(x2_out)
        loss_instant = beta * sigmoid1 * sigmoid2

        # calculate the collision area lower and upper bounds (non-aggressive)
        x1_in_na = ((d1 - R1 / 2 + theta_na * W2 / 2) * 5).squeeze().reshape(-1, 1).cuda()
        x2_in_na = ((d2 - R2 / 2 + theta_na * W1 / 2) * 5).squeeze().reshape(-1, 1).cuda()

        sigmoid1_na = torch.sigmoid(x1_in_na) * torch.sigmoid(x1_out)
        sigmoid2_na = torch.sigmoid(x2_in_na) * torch.sigmoid(x2_out)
        loss_instant_na = beta * sigmoid1_na * sigmoid2_na

        # calculate instantaneous loss
        loss_fun_1 = p1.T * (u1_a ** 2 + loss_instant.cuda()) + (1 - p1).T * (
                y1_a * (u1_a ** 2 + loss_instant_na.cuda()))
        loss_fun_2 = p2.T * (u2_a ** 2 + loss_instant.cuda()) + (1 - p2).T * (
                y2_a * (u2_a ** 2 + loss_instant_na.cuda()))

        # calculate ham for agents for belief dynamics:
        # H = \lambda^T * (-f) + L

        H_1_a = -lam11_1_a.squeeze() * v1.squeeze() - lam11_2_a.squeeze() * u1_a.squeeze() - \
                lam12_1_a.squeeze() * v2.squeeze() - lam12_2_a.squeeze() * u2_a.squeeze() + \
                (u1_a ** 2 + loss_instant.cuda())
        H_1_na = -lam11_1_na.squeeze() * v1.squeeze() - lam11_2_na.squeeze() * u1_na.squeeze() - \
                 lam12_1_na.squeeze() * v2.squeeze() - lam12_2_na.squeeze() * u2_na.squeeze() + \
                 (u1_na ** 2 + loss_instant.cuda())

        H_2_a = -lam21_1_a.squeeze() * v1.squeeze() - lam21_2_a.squeeze() * u1_a.squeeze() - \
                lam22_1_a.squeeze() * v2.squeeze() - lam22_2_a.squeeze() * u2_a.squeeze() + \
                (u2_a ** 2 + loss_instant.cuda())
        H_2_na = -lam21_1_na.squeeze() * v1.squeeze() - lam21_2_na.squeeze() * u1_na.squeeze() - \
                 lam22_1_na.squeeze() * v2.squeeze() - lam22_2_na.squeeze() * u2_na.squeeze() + \
                 (u2_na ** 2 + loss_instant_na.cuda())

        # calculate hamiltonian, H = lambda^T * (-f) + L because we invert the time
        # set alpha = const (learning rate) = 0.01 (for eg)
        alpha = 0.01
        ham_1 = -lam11_1_na.squeeze() * v1.squeeze() - lam11_2_na.squeeze() * u1_na.squeeze() - lam11_3_na.squeeze() * torch.sign(
            H_2_a - H_2_na) * alpha - \
                lam12_1_na.squeeze() * v2.squeeze() - lam12_2_na.squeeze() * u2_na.squeeze() + loss_fun_1.squeeze()
        ham_2 = -lam21_1_na.squeeze() * v1.squeeze() - lam21_2_na.squeeze() * u1_na.squeeze() - lam21_3_na.squeeze() * torch.sign(
            H_1_a - H_1_na) * alpha - \
                lam22_1_na.squeeze() * v2.squeeze() - lam22_2_na.squeeze() * u2_na.squeeze() + loss_fun_2.squeeze()

        # dirichlet_mask is the bool array. It evaluates whether y[dirichlet_mask] is boundary condition or not
        # HJI check
        if torch.all(dirichlet_mask):
            diff_constraint_hom_1 = torch.Tensor([0])
            diff_constraint_hom_2 = torch.Tensor([0])
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)
        else:
            diff_constraint_hom_1 = dvdt_1_na + ham_1
            diff_constraint_hom_2 = dvdt_2_na + ham_2
            diff_constraint_hom = torch.cat((diff_constraint_hom_1, diff_constraint_hom_2), dim=0)

        # boundary condition check
        dirichlet_1 = y1_na[dirichlet_mask] - (source_boundary_values[:, :y1_na.shape[1]][dirichlet_mask])
        dirichlet_2 = y2_na[dirichlet_mask] - (source_boundary_values[:, y2_na.shape[1]:][dirichlet_mask])
        dirichlet = torch.cat((dirichlet_1, dirichlet_2), dim=0)

        # A factor of (2e5, 100) to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 2e7,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() / 100}

    return intersection_hji
