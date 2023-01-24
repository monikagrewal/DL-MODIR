import numpy as np
import torch
import os
import pdb
import random
import copy
import csv
import pickle
import json

from importlib import import_module
from datetime import datetime

from class_net_ensemble import NetEnsemble
from class_statistics_writer import StatisticsWriter
from functions_plotting import plot_selected_iterations, plot_loss_weights, plot_training_process, plot_convergence
# from functions_optimization_benchmarks import load_benchmark
import time
from tqdm import tqdm


def do_dynamic_weight_optimization(cfg, net_ensemble, statistics_writer_instance, output_folder):
    mo_mode = cfg["mo_mode"]
    ntraining_sample = len(net_ensemble.training_data)
    n_learning_iterations = cfg["n_learning_iterations"]
    n_epochs = int(np.ceil(( n_learning_iterations*cfg["batch_size"] ) / ntraining_sample))
    nbatches = len(net_ensemble.training_data) // cfg["batch_size"]
    niter = 0
    for epoch in range(n_epochs):
        desc = "Epoch {}".format(epoch)
        for i, data in tqdm(enumerate(net_ensemble.training_dataloader), desc=desc, total=nbatches, unit="batches"):
            if mo_mode=="loss_per_sample":
                net_ensemble.dynamic_weight_optimization_per_sample(data)
            elif mo_mode == "mean_loss_over_samples":
                net_ensemble.dynamic_weight_optimization_mean(data)
            else:
                raise ValueError("unknown mo mode")

            niter += 1
            if not statistics_writer_instance.grid_search_memory_saving:
                cur_iter_is_best_iter = statistics_writer_instance.update_statistics(net_ensemble, epoch)          
                if (niter%1000)==0 or (niter==n_learning_iterations):
                    net_ensemble.validation_fn(net_ensemble, statistics_writer_instance, epoch, output_folder)
                    statistics_writer_instance.save(net_ensemble, output_folder)

            # update learning rate
            for scheduler in net_ensemble.meta_optimizer_scheduler_list:
                scheduler.step()

            if niter==n_learning_iterations:
                break

        if niter==n_learning_iterations:
            break

    if statistics_writer_instance.grid_search_memory_saving:
        cur_iter_is_best_iter = statistics_writer_instance.update_statistics(net_ensemble, epoch)
        net_ensemble.validation_fn(net_ensemble, statistics_writer_instance, epoch, output_folder)
        statistics_writer_instance.save(net_ensemble, output_folder)

    return(net_ensemble)


def run_instance(output_folder_name, rand_seed, target_device, cfg, grid_search_memory_saving):
    # set seeds
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_default_dtype(torch.float32)

    problem_name = cfg["problem_name"]
    mo_optimizer = cfg["mo_optimizer"]
    n_mo_sol = cfg["n_mo_sol"]
    n_learning_iterations = cfg["n_learning_iterations"]
    ref_point = np.array(cfg["ref_point"])
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    use_simple_gradient_weighing_in_optimizer = False # REMOVE IN FUTURE VERSIONS
    # whether or not dUHV/df gets normalized
    use_obj_space_normalization = True
    
    # I removed the optimizer and moved the statistics writing functionality to another object. It remains commmented in case we revive the external optimizer functionality
    # uhv_opt = optimizer(n_mo_sol,net_ensemble.n_parameters,step_size,use_simple_gradient_weighing_in_optimizer,use_obj_space_normalization,n_learning_iterations,output_file_name,output_folder_name,ref_point,net_ensemble.weights)
    # instantiate StatisticsWriter
    output_file_name = os.path.join(output_folder_name, 'statistics.dat')
    statistics_writer_instance = StatisticsWriter(cfg["mo_mode"], output_file_name, ref_point, n_mo_sol, n_learning_iterations, grid_search_memory_saving)
    
    # create an ensemble of networks
    t0 = time.time()
    net_ensemble = NetEnsemble(target_device, cfg)
    net_ensemble = do_dynamic_weight_optimization(cfg, net_ensemble, statistics_writer_instance, output_folder_name)
    print("Total time: {} seconds".format(time.time() - t0))
    if not grid_search_memory_saving:
        # plot objective space (only works for 2 or 3 losses)
        if (statistics_writer_instance.n_mo_obj == 2) or (statistics_writer_instance.n_mo_obj == 3):
            extensive_plotting = False # plot many iterations in objective space
            plot_selected_iterations(statistics_writer_instance,cfg,output_folder_name,extensive_plotting)
        # plot loss weights
        plot_loss_weights(statistics_writer_instance, cfg, output_folder_name)

        # plot convergence
        plot_convergence(statistics_writer_instance, cfg, output_folder_name)
        # plot training HV & losses
        plot_training_process(statistics_writer_instance, cfg, output_folder_name)

        # problem-dependent post-processing
        if problem_name == 'vincent_van_jarjar':
            net_ensemble.postprocess_fn(net_ensemble, output_folder_name)
        elif problem_name == 'sin_cos':
            net_ensemble.postprocess_fn(net_ensemble, statistics_writer_instance, output_folder_name,cfg,target_device)
        elif problem_name == 'pruning':
            net_ensemble.postprocess_fn(net_ensemble, output_folder_name)
        else:
            pass


def run_experiment(cfg_list, grid_search_memory_saving, experiment_name="tmp", base_seed=20200625, target_device="cuda:0", number_of_runs=1):
    # ---- set up folder structure ----
    base_path = os.path.dirname(os.path.abspath(__file__))
    suffix = datetime.strftime(datetime.now(), "_%d-%m-%Y_%H%M%S")
    experiment_path = os.path.join(base_path, "output_files", experiment_name + suffix)
    os.makedirs(experiment_path, exist_ok=True)

    config_list_filepath = os.path.join(experiment_path, 'config_list.json')
    json.dump(cfg_list, open(config_list_filepath, "w"))

    #  ---- iterate over config file ----
    logfilepath = os.path.join(experiment_path, "errorlog.txt")
    logfile = open(logfilepath, "w")
    logfile.close()
    for idx, cfg in enumerate(cfg_list):
        idx_path = os.path.join(experiment_path, str(idx))
        os.makedirs(idx_path, exist_ok=True)
        config_filepath = os.path.join(idx_path, 'cfg.json')
        json.dump(cfg, open(config_filepath, "w"))

        for i_run in range(0, number_of_runs):
            rand_seed = base_seed + i_run
            run_path = os.path.join(idx_path, "run_{}".format(rand_seed))
            os.makedirs(run_path, exist_ok=True)
            # try:
            run_instance(run_path, rand_seed, target_device, cfg, grid_search_memory_saving)
            # except Exception as e:
                # print(e)
                # logfile = open(logfilepath, "a")
                # logfile.write("Error in config idx = {}, run_path = {}\n".format(idx, run_path))
                # logfile.write("{}\n".format(e))
                # logfile.close()




if __name__ == '__main__':
    pass
