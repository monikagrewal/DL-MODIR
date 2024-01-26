import numpy as np
import torch
import logging

from config import config
from utilities.logging import log_iteration_metrics, save_checkpoint
from utilities.functions_plotting import save_os_visualization
from functions.functions_evaluation import compute_hv_in_higher_dimensions

def main(test_fn, net_ensemble, dataloader, criterion, cache, writer):
    if config.VISUALIZE_OUTPUT in ["test", "all"]:
        visualize = True
    else:
        visualize = False
    metrics = test_fn(net_ensemble, dataloader, criterion, cache, visualize=visualize)

    # compute hv: mean HV over samples
    loss = metrics["loss"]
    n_samples = loss.shape[0]
    hv_per_sample = np.zeros(n_samples)
    for i_sample in range(0,n_samples):
        hv_per_sample[i_sample] = compute_hv_in_higher_dimensions(loss[i_sample,:,:], config.REF_POINT)
    metrics["hv"] = hv_per_sample

    # logging: test
    # log_iteration_metrics(metrics, cache.iter, writer, data="test", loss_functions=config.LOSS_FUNCTIONS)
    # visualizing: test pareto front
    save_os_visualization(metrics["loss"], cache.out_dir_test, 0, config.LOSS_FUNCTIONS)
    # saving
    mean_hv = float(np.mean(hv_per_sample))