import numpy as np
import torch
import logging

from config import config
from utilities.logging import log_iteration_metrics, save_checkpoint
from utilities.functions_plotting import save_os_visualization
from functions.functions_evaluation import compute_hv_in_higher_dimensions


def sanity_check(val, min_val=0, max_val=np.infty):
    assert (not np.any(np.isnan(val)))
    # assert np.all(val >= min_val)
    assert np.all(val < max_val)    
    return None


def dynamic_weight_optimization_mean(data, mo_optimizer, net_ensemble, criterion):
    """
    maximize the average (for each batch) HV over all solutions
    """
    opt_name = mo_optimizer.name
    inputs = data['X']
    labels = data['Y']

    # dynamic weight calculation in epo and pareto MTL requires gradients, so needs extra forward & backward propagation
    if opt_name in ['pareto_mtl', 'epo']:
        dynamic_weights = mo_optimizer.compute_weights(net_ensemble.net_list,
                                                    net_ensemble.optimizer_list,
                                                    criterion,
                                                    inputs,
                                                    labels)

    # forward propagation
    _, mo_obj_val_per_sample, mo_obj_val_torch, mo_obj_val_mean, so_obj_val = forward_propagation(inputs, labels, net_ensemble, criterion)

    # compute dynamic weights
    if opt_name in ['uhv', 'higamo_hv', 'linear_scalarization']:
        dynamic_weights = mo_optimizer.compute_weights(mo_obj_val_mean)

    dynamic_weights = dynamic_weights.to(config.DEVICE)
    # backward propagation and optimizer step
    if net_ensemble.__class__.__name__=="DeepEnsemble":
        for i_mo_sol in range(0, net_ensemble.n_mo_sol):
            weighted_objective = torch.sum(dynamic_weights[:, i_mo_sol] * mo_obj_val_torch[i_mo_sol])
            weighted_objective.backward()
            net_ensemble.optimizer_list[i_mo_sol].step()
    elif net_ensemble.__class__.__name__=="KHeadEnsemble":
        weighted_objective = 0.
        for i_mo_sol in range(0, net_ensemble.n_mo_sol):
            # adding torch.mean, does it have an effect on the step size? with mean, the magnitude of weighted_objective for both mo_modes is the same
            weighted_objective += (1./net_ensemble.n_mo_sol) * torch.sum(dynamic_weights[:, i_mo_sol] * mo_obj_val_torch[i_mo_sol])
        
        weighted_objective += so_obj_val
        
        weighted_objective.backward()
        net_ensemble.optimizer.step()


    dynamic_weights_cpu = dynamic_weights.cpu().numpy()
    metrics = {"dynamic_weights": dynamic_weights_cpu,
                "loss": mo_obj_val_per_sample}

    return net_ensemble, metrics


def dynamic_weight_optimization_per_sample(data, mo_optimizer, net_ensemble, criterion):
    """
    maximize the HV of all samples separately
    """
    opt_name = mo_optimizer.name
    inputs = data['X']
    labels = data['Y']
    if isinstance(inputs, list):
        n_samples = inputs[0].shape[0]
    else:
        n_samples = inputs.shape[0]
    
    # dynamic weight calculation in epo and pareto MTL
    if opt_name in ['pareto_mtl', 'epo']:
        raise ValueError("dynamic weight optimization per sample does not make sense for ParetoMTL and EPO (imo)")

    # forward propagation
    mo_obj_val_torch_per_sample, mo_obj_val_per_sample, _, _ = forward_propagation(inputs, labels, net_ensemble, criterion)

    # compute dynamic weights per sample
    dynamic_weights_per_sample = torch.ones(net_ensemble.n_mo_sol, net_ensemble.n_mo_obj, n_samples)
    for i_sample in range(0, n_samples):
        if opt_name in ['uhv', 'higamo_hv', 'linear_scalarization']:
            weights = mo_optimizer.compute_weights(mo_obj_val_per_sample[i_sample,:,:])
        dynamic_weights_per_sample[:, :, i_sample] = weights.permute(1,0)

    dynamic_weights_per_sample = dynamic_weights_per_sample.to(config.DEVICE)
    dynamic_weights = torch.mean(dynamic_weights_per_sample, dim=2)
    dynamic_weights = dynamic_weights.permute(1, 0)

    # backward propagation
    if net_ensemble.__class__.__name__=="DeepEnsemble":
        for i_mo_sol in range(0, net_ensemble.n_mo_sol):
            # adding torch.mean, does it have an effect on the step size? with mean, the magnitude of weighted_objective for both mo_modes is the same
            weighted_objective = torch.sum(torch.mean(dynamic_weights_per_sample[i_mo_sol, :, :] * mo_obj_val_torch_per_sample[i_mo_sol], dim=1))
            weighted_objective.backward()
            net_ensemble.optimizer_list[i_mo_sol].step()
    elif net_ensemble.__class__.__name__=="KHeadEnsemble":
        weighted_objective = 0.
        for i_mo_sol in range(0, net_ensemble.n_mo_sol):
            # adding torch.mean, does it have an effect on the step size? with mean, the magnitude of weighted_objective for both mo_modes is the same
            weighted_objective += (1./net_ensemble.n_mo_sol) * torch.sum(torch.mean(dynamic_weights_per_sample[i_mo_sol, :, :] * mo_obj_val_torch_per_sample[i_mo_sol], dim=1))
        
        weighted_objective.backward()
        net_ensemble.optimizer.step()


    dynamic_weights_cpu = dynamic_weights.cpu().numpy()
    metrics = {"dynamic_weights": dynamic_weights_cpu,
                "loss": mo_obj_val_per_sample}   

    return net_ensemble, metrics


def forward_propagation(inputs, labels, net_ensemble, criterion):
    """
    compute mo_obj_val per sample and the mo_obj_val_mean (mo_obj_val averaged over all samples)
    """
    if isinstance(inputs, list):
        n_samples = inputs[0].shape[0]
    else:
        n_samples = inputs.shape[0]
    mo_obj_val_torch_per_sample = list()
    mo_obj_val_torch = list()
    mo_obj_val_per_sample = np.zeros((n_samples, net_ensemble.n_mo_obj, net_ensemble.n_mo_sol))
    mo_obj_val_mean = np.zeros((net_ensemble.n_mo_obj, net_ensemble.n_mo_sol))

    if net_ensemble.__class__.__name__=="DeepEnsemble":
        for i_mo_sol in range(0, net_ensemble.n_mo_sol):
            net_ensemble.optimizer_list[i_mo_sol].zero_grad()
            
            Y_hat = net_ensemble.net_list[i_mo_sol](inputs)
            loss_per_sample = criterion(Y_hat, labels)
            loss_per_sample = torch.stack(loss_per_sample, dim=0)
            loss_mean = loss_per_sample.mean(dim=1).view(-1)

            mo_obj_val_torch_per_sample.append(loss_per_sample)
            mo_obj_val_torch.append(loss_mean)
            
            mo_obj_val_per_sample[:, :, i_mo_sol] = loss_per_sample.cpu().detach().numpy().T
            mo_obj_val_mean[:, i_mo_sol] = loss_mean.cpu().detach().numpy()
    elif net_ensemble.__class__.__name__=="KHeadEnsemble":
        net_ensemble.optimizer.zero_grad()
        Y_hat_list = net_ensemble.model(inputs)
        for i_mo_sol in range(0, net_ensemble.n_mo_sol):
            Y_hat = Y_hat_list[i_mo_sol]
            loss_per_sample = criterion(Y_hat, labels)
            loss_per_sample = torch.stack(loss_per_sample, dim=0)
            loss_mean = loss_per_sample.mean(dim=1).view(-1)

            mo_obj_val_torch_per_sample.append(loss_per_sample)
            mo_obj_val_torch.append(loss_mean)
            
            mo_obj_val_per_sample[:, :, i_mo_sol] = loss_per_sample.cpu().detach().numpy().T
            mo_obj_val_mean[:, i_mo_sol] = loss_mean.cpu().detach().numpy()

    # check validity of mo_obj_val_mean & mo_obj_val_per_sample
    sanity_check(mo_obj_val_mean)
    sanity_check(mo_obj_val_per_sample)
    if mo_obj_val_mean.shape[0]>3:
        mo_obj_val_mean = mo_obj_val_mean[:-1, :]
        so_obj_val = mo_obj_val_torch[0][-1]
        mo_obj_val_torch = [item[:-1] for item in mo_obj_val_torch]
    else:
        so_obj_val = 0

    outputs = (mo_obj_val_torch_per_sample, mo_obj_val_per_sample, mo_obj_val_torch, mo_obj_val_mean, so_obj_val)
    return outputs


def train(mo_optimizer, net_ensemble, criterion, validation_fn, scaler, dataloaders, cache, writer):
    # training
    cache.iter = 0
    cache.epoch = 0
    while cache.iter < config.LEARNING_ITERATIONS:
        for i, data in enumerate(dataloaders["train"]):
            logging.debug(f"Iteration: {cache.iter}")
            if config.MO_MODE=="loss_per_sample":
                net_ensemble, metrics = dynamic_weight_optimization_per_sample(data, mo_optimizer, net_ensemble, criterion)
            elif config.MO_MODE=="mean_loss_over_samples":
                net_ensemble, metrics = dynamic_weight_optimization_mean(data, mo_optimizer, net_ensemble, criterion)
            else:
                raise ValueError(f"unknown MO_MODE: {config.MO_MODE}")

           
            # logging: train
            log_iteration_metrics(metrics, cache.iter, writer, data="train", loss_functions=config.LOSS_FUNCTIONS)
            # validation
            if ( (cache.iter+1)%config.VALIDATION_FREQUENCY )==0 or ( (cache.iter+1)==config.LEARNING_ITERATIONS ):
                if config.VISUALIZE_OUTPUT in ["val", "all"]:
                    visualize = True
                else:
                    visualize = False
                metrics = validation_fn(net_ensemble, dataloaders["val"], criterion, cache, visualize=visualize)

                # compute hv: mean HV over samples
                loss = metrics["loss"]
                n_samples = loss.shape[0]
                hv_per_sample = np.zeros(n_samples)
                for i_sample in range(0,n_samples):
                    hv_per_sample[i_sample] = compute_hv_in_higher_dimensions(loss[i_sample,:,:], config.REF_POINT)
                metrics["hv"] = hv_per_sample

                # logging: val
                log_iteration_metrics(metrics, cache.iter, writer, data="val", loss_functions=config.LOSS_FUNCTIONS)
                # visualizing: val pareto front
                save_os_visualization(metrics["loss"], cache.out_dir_val, config.LOSS_FUNCTIONS)
                # saving
                mean_hv = float(np.mean(hv_per_sample))
                cache.hv = mean_hv
                if cache.hv >= cache.best_hv:
                    cache.best_hv = cache.hv
                    cache.best_iter = cache.iter
                    if config.SAVE_MODEL == "best":
                        save_checkpoint(net_ensemble, cache, filename=f"checkpoint_{config.SAVE_MODEL}")
                
                if config.SAVE_MODEL == "final":
                    save_checkpoint(net_ensemble, cache, filename=f"checkpoint_{config.SAVE_MODEL}")
            
            cache.iter += 1
            if (cache.iter)==config.LEARNING_ITERATIONS:
                break

        # update learning rate
        if net_ensemble.__class__.__name__=="DeepEnsemble":
            for scheduler in net_ensemble.lr_scheduler_list:
                scheduler.step()
        elif net_ensemble.__class__.__name__=="KHeadEnsemble":
            net_ensemble.lr_scheduler.step()

        cache.epoch += 1
        logging.info(f"Epoch: {cache.epoch}")
    # TODO: add Pareto front to summarywriter