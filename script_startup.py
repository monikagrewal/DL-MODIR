import os
import pdb
import json
from functions_run_instance import run_experiment
from utils import get_config_list, generate_config_list

"""
Usage: 
Step 1: define an experiment_name, it will decide the main output folder, e.g., "grid_search"

Step 2, option 1: call get_config_list with parameter "config_name"
		config_name = either path of existing config_list or name of the problem to load default values

Step 2, option 2: call generate_config_list with parameters:
		config_name = name of the problem e.g. "mtl_as_moo"
		params_dict = dictionary of parameters that need to be changed. See full description in function docs

"""
base_seed = 12345
target_device = "cuda:3"
number_of_runs = 1
grid_search_memory_saving = False # do not plot, do not store dynamic_weights_list, hv_per_sample_list, mo_obj_val_per_sample_list in class StatisticsWriter

config_name = "modir_amc"
experiment_name = "preliminary_experiments/modir_amc_per_sample"
# config_list = get_config_list(config_name)

params_dict = {"mo_mode": ["loss_per_sample"]}
# params_dict = {"mo_mode": ["mean_loss_over_samples"]}
config_list = generate_config_list(config_name, params_dict)
print(config_list)
print("number of experiments: ", len(config_list))
run_experiment(config_list, grid_search_memory_saving, experiment_name=experiment_name,
				base_seed=base_seed, target_device=target_device, number_of_runs=number_of_runs)
