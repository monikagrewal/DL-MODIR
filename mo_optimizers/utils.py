import numpy as np
import torch
from torch.autograd import Variable


def compute_grads_and_losses(n_mo_obj,cur_net,optimizer,obj_func,input_data_batch,label_batch):
    """
    computes gradients for each objective
    """
    grads = dict()
    for i_mo_obj in range(0,n_mo_obj):
        optimizer.zero_grad()
        Y_hat = cur_net(input_data_batch)
        loss_per_sample = obj_func(Y_hat,label_batch)
        loss_per_sample = torch.stack(loss_per_sample, dim=0)
        losses = loss_per_sample.mean(dim=1).view(-1)
        losses[i_mo_obj].backward()

        # compute gradients but keep computational graph because we will need it to backprop over the final loss function
        # task_loss[i_mo_obj].backward(retain_graph = True)
        
        grads[i_mo_obj] = []
        # can use scalable method proposed in the MOO-MTL paper for large scale problem
        # but we keep use the gradient of all parameters in this experiment
        if hasattr(cur_net,"style_layers"):
            param = cur_net.params
            if param.grad is not None:
                grads[i_mo_obj].append(param.grad.data.clone().flatten())
        else:
            for param in cur_net.params:
                if param.grad is not None:
                    grads[i_mo_obj].append(param.grad.data.clone().flatten())

    grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
    grads = torch.stack(grads_list)

    # check that losses are not nan and [0,infty)
    for entry in losses:
        assert (not torch.any(torch.isnan(entry)))
        assert torch.all(entry >= 0)
        assert torch.all(torch.isfinite(entry))

    # check that losses are not nan and (-infty,infty)
    assert (not torch.any(torch.isnan(grads)))
    assert torch.all(torch.isfinite(grads))

    return(grads,losses)


def generate_k_preferences(K, min_angle=None, max_angle=None):
    """
    generate evenly distributed preference vector
    original code from "circle_points_epo" in (probably) EPO repo
    """
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y].astype(np.float32)


def generate_fixed_weights(n_obj, n_mo_sol):
    """
    inputs:
    n_obj, n_mo_sol

    outputs:
    weights: n_obj * n_mo_sol, tensor
    """
    if n_obj==1:
        weights = torch.ones(n_mo_sol).view(1, -1)
    elif n_obj==2:
        if n_mo_sol==1:
            weights = torch.ones(n_obj, n_mo_sol)
        else:
            weights = torch.zeros(n_obj, n_mo_sol)
            for i_mo_sol in range(0, n_mo_sol):
                weights[0, i_mo_sol] = i_mo_sol/(n_mo_sol-1)
                weights[1, i_mo_sol] = 1 - weights[0, i_mo_sol]
    else:
        raise ValueError('generating fixed weights is not yet generalized to more than 2 objectives')

    return weights