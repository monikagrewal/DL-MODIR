import os
import numpy as np
import torch
import itertools
import json
import glob
import _pickle as pickle
from cfg_default import default_config
import re
import copy

def train_and_val_split(training_data, validation_data, train_ratio=0.8):
	nsamples = len(training_data)
	print("total data: ", nsamples)
	indices = np.arange(nsamples)
	np.random.shuffle(indices)

	train_indices = indices[:int(nsamples*train_ratio)]
	val_indices = indices[int(nsamples*train_ratio):]

	training_data = training_data.partition(train_indices)
	validation_data = validation_data.partition(val_indices)
	print("training data: {}, validation data: {}".format(
			len(training_data), len(validation_data)))
	return training_data, validation_data


def generate_config_list(config_name, params_dict):
	"""
	Inputs: 
	config_name: Examples: "modir", "sin_cos", "linear_regression", 
	"fashion_mnist", "vincent_van_jarjar", "mtl_as_moo"
	
	params_dict: Example: {"batch_size": [8, 16, 32],
							"n_mo_sol": [4, 5, 6],
							("meta_optimizer_params", "lr"): [0.01, 0.001]}

	Note: 1) the keys should be exactly the keys in the config file
	2) For a parameter in a nested level of dictionary, the key should be a tuple
	Assuming the config file has only two levels of nesting

	"""
	# ---- get template_config from default_config.py ----
	default_problems = list(default_config.keys())
	if config_name in default_problems:
		template_config = default_config[config_name]
	elif type(config_name) == dict:
		template_config = config_name
	else:
		raise ValueError("Default config for the specified name could not be found.")

	#---- convert params_dict to grid ----
	keys = params_dict.keys()
	values = (params_dict[key] for key in keys)
	params_grid = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
	
	#---- make new config for each entry in params_grid ----
	config_list = []
	for params in params_grid:
		new_config = template_config.copy()
		for name, value in params.items():
			if not isinstance(name, str):
				new_config[name[0]][name[1]] = value
			else:
				new_config[name] = value
		x = copy.deepcopy(new_config)
		config_list.append(x)

	return config_list


def get_config_list(config_name):
	default_problems = list(default_config.keys())
	if os.path.splitext(config_name)[-1]==".json" and os.path.isfile(config_name):
		config_list = [json.load(open(config_name))]
	elif config_name in default_problems:
		config_list = [default_config[config_name].copy()]
	else:
		raise ValueError("Can't make sense of the given config name: {}".format(config_name))

	return config_list


def find_best_config(experiment_path):
	def print_nicely(arr):
		"""
		first entry is confusing, so don't print that
		"""
		new_arr = [item[1:] for item in arr]
		for item in new_arr:
			print("{}".format(item))

	config_paths = glob.glob(experiment_path + "/*/cfg.json")
	if len(config_paths)==0:
		raise RuntimeError("No config files were found in the experiment_path: {}".format(experiment_path))
	config_paths.sort()
	all_metrics = []
	for idx, pathname in enumerate(config_paths):
		parent, config_name = os.path.split(pathname)
		config_folder_name = parent.split("/")[-1]
		run_paths = glob.glob(parent + "/run*/")
		best_epochs_list = []
		metric_list = []
		for run_path in run_paths:
			checkpoint_path = os.path.join(run_path, "checkpoint.pth")
			if not os.path.exists(checkpoint_path):
				print("No saved checkpoint found in path: {}".format(checkpoint_path))
				best_epoch = 0
				metric = 0
			else:
				checkpoint = torch.load(checkpoint_path)
				stats = checkpoint["stats"]
				best_epoch = stats.epoch
				metric = stats.validation_hv
			print(checkpoint_path, ": ", metric)
			best_epochs_list.append(best_epoch)
			metric_list.append(metric)

		avg_best_epoch = np.average(best_epochs_list)
		avg_metric = np.average(metric_list)
		all_metrics.append([idx, config_folder_name, avg_best_epoch, avg_metric])

	# ---- find best metric ----
	all_metrics = sorted(all_metrics, key=lambda x: x[3], reverse=True)
	print_nicely(all_metrics)
	best_epoch = all_metrics[0][2]
	best_idx = all_metrics[0][0]
	best_config_path = config_paths[best_idx]
	best_config = json.load(open(best_config_path, "rb"))
	return best_epoch, best_config


def find_best_config_in_folder(root_path, output_path="configs"):
	"""
	wrapper on find_best_config function
	runs find_best_config over all subfolders in the specified root_path

	example root_path = "output_files/grid_search_experiments"

	Note that the function also expects subfolders inside the root_path. \
	ideally these subfolder represent problem names e.g. sin_cos, mo_segmentation
	"""
	output_path = "configs"
	problem_names = os.listdir(root_path)

	for problem_name in problem_names:
		problem_folder = os.path.join(root_path, problem_name)
		experiment_names = os.listdir(problem_folder)

		for experiment_name in experiment_names:
			experiment_path = os.path.join(problem_folder, experiment_name)

			best_epoch, best_config = find_best_config(experiment_path)
			print("Best epoch: ", best_epoch)
			print(json.dumps(best_config))

			output_folder = os.path.join(output_path, problem_name)
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)
			json.dump(best_config, open(os.path.join(output_folder, f"{experiment_name}.json"), "w"))
	return None


def get_experiment_info(experiment_path):
	config_paths = glob.glob(experiment_path + "/*/cfg.json")
	if len(config_paths)==0:
		raise RuntimeError("No config files were found in the experiment_path: {}".format(experiment_path))
	config_paths.sort()
	info_list = []
	for idx, pathname in enumerate(config_paths):
		cfg = json.load(open(pathname, "rb"))
		parent, config_name = os.path.split(pathname)
		# config_idx = parent.split("/")[-1]
		run_paths = glob.glob(parent + "/run*/")
		run_paths.sort()
		info = {"path": parent,
				"config": cfg,
				"run_paths": run_paths
				}
		info_list.append(info)

	return info_list


def patchup_old_to_new_saving(experiment_path):
	"""
	saves torch state dicts from old checkpoints in an experiment folder

	Note the commented section is required only for mtl/mo_seg old saving
	"""
	checkpoints_pathlist = glob.glob(experiment_path + "/*/*/checkpoint.pckl")
	for checkpoint_path in checkpoints_pathlist:
		filepath, filename = os.path.split(checkpoint_path)
		filename, _ = os.path.splitext(filename)
		checkpoint = pickle.load(open(checkpoint_path, "rb"))

		# checkpoint["training_data_indices"] = checkpoint["training_data"]
		# checkpoint["validation_data_indices"] = checkpoint["validation_data"]
		# checkpoint["training_data"] = None
		# checkpoint["validation_data"] = None

		net_list = checkpoint["net_list"]
		state_dicts = [net.state_dict() for net in net_list]
		checkpoint["state_dicts"] = state_dicts
		del checkpoint["net_list"]

		new_filepath = os.path.join(filepath, "{}.pth".format(filename))
		print(checkpoint_path, " --> ", new_filepath)
		torch.save(checkpoint, new_filepath)

	return None


if __name__ == '__main__':
	# config_name = "cfg.json"
	# template_config = json.load(open(config_name))[0]
	# params_dict = {"batch_size": [8, 16, 32],
	# 				("meta_optimizer_params", "lr"): [0.001, 0.0001]}
	# config_list = generate_config_list(template_config, params_dict)

	# for item in config_list:
	# 	print(item)
	# 	print("")
	root_folder = "output_files/grid_search_experiments"
	find_best_config_in_folder(root_folder)
