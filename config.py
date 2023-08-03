import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Tuple
from pydantic import BaseSettings, validator
import torch

from cli import cli_args


class Config(BaseSettings):
	# General
	EXPERIMENT_NAME: str = "prelim_experiment"
	# Folders for logging, Base folders
	OUT_DIR: str = ""
	@validator("OUT_DIR")
	def set_out_dir(cls, v, values):
		"""Dynamically create based on experiment name"""
		t0 = datetime.now()
		t0_str = datetime.strftime(t0, "%d%m%Y_%H%M%S")
		value = f"runs/{values['EXPERIMENT_NAME']}_{t0_str}"
		return value

	# Subdirectories
	FOLDERNAMES: Dict[str, str] = {
		"out_dir_train": "train",
		"out_dir_val": "val",
		"out_dir_weights": "weights",
		"out_dir_results": "results",
	}

	DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
	DEBUG: bool = False

	RANDOM_SEED: int = 20220903
	NRUNS: int = 1
	NFOLDS: Optional[int] = 1

	# problem specific
	PROBLEM_NAME: Literal["modir", "modir3d"] = "modir"
	MO_OPTIMIZER: Literal["higamo_hv", "linear_scalarization", "pareto_mtl"] = "higamo_hv"
	MO_OPTIMIZER_PARAMS: Dict = {"beta_one": 0.9, "obj_space_normalize": True}
	MO_MODE: Literal["mean_loss_over_samples", "loss_per_sample"] = "mean_loss_over_samples"
	N_MO_SOL: int = 5
	@validator("N_MO_SOL")
	def check_n_mo_sol(cls, v, values):
		if v<1:
			raise ValueError(f"N_MO_SOL should be >= 1.")
		return v
	ENSEMBLE_TYPE: Literal["deep", "khead"] = "deep"

	DATASET: str = "MNIST_DIR"
	DATA_PARAMS: Dict = {"root": "/export/scratch2/data/grewal/Data"}
	MODEL_NAME: str = "Net"
	MODEL_PARAMS: Dict = {"width":16, "depth":3}
	@validator("DATA_PARAMS", "MODEL_PARAMS", "MO_OPTIMIZER_PARAMS")
	def convert_to_bool(cls, v, values):
		str_to_bool = {"true": True, "false": False}
		for key, val in v.items():
			if val in ["true", "false"]:
				v[key] = str_to_bool[val]
		return v
	OPTIMIZER: Literal["SGD", "Adam"] = "Adam"
	OPTIMIZER_PARAMS: Dict = {}
	LR: float = 1e-3
	LR_SCHEDULER: Literal["StepLR"] = "StepLR"
	LR_SCHEDULER_PARAMS: Dict = {"step_size": 1,"gamma": 1.0}
	WEIGHT_DECAY: float = 1e-4
	BATCHSIZE: int = 1
	LEARNING_ITERATIONS: int = 20
	LOSS_FUNCTIONS: List = ["NCCLoss", "TransformationLoss"]
	@validator("LOSS_FUNCTIONS")
	def set_n_obj(cls, v, values):
		# assert 1 < len(v) < 4
		values["N_MO_OBJ"] = len(v)
		return v

	REF_POINT: Tuple = (20, 20)
	@validator("REF_POINT")
	def check_ref_point(cls, v, values):
		if values["MO_OPTIMIZER"] == "higamo_hv":
			assert len(v) == len(values["LOSS_FUNCTIONS"])
		return v

	# validation frequency
	VALIDATION_FREQUENCY = 10
	# Where to perform visualization
	VISUALIZE_OUTPUT: Literal["none", "val", "test", "all"] = "none"
	SAVE_MODEL: Literal["none", "best", "final"] = "best"


class TestConfig(BaseSettings):
	EXPERIMENT_DIR: str = "./runs/prelim_experiment"
	VISUALIZE_OUTPUT: Literal["none", "val", "test", "all"] = "none"


def get_config(env_file=cli_args.env_file, test_env_file=cli_args.test_env_file):

	if not env_file and not test_env_file:
		print("No env_file supplied. " "Creating default config")
		return Config()
	else:
		if env_file:
			env_path = Path(env_file).expanduser()
			if env_path.is_file():
				print("Creating config based on file")
				return Config(_env_file=env_file)
			else:
				print(
					"env_file supplied but does not resolve to a file. "
					"Creating default config"
				)
				return Config()
		elif test_env_file:
			env_path = Path(test_env_file).expanduser()
			if env_path.is_file():
				print("Creating config based on file")
				test_settings = TestConfig(_env_file=test_env_file)
			else:
				print(
					"env_file supplied but does not resolve to a file. "
					"Creating default config"
				)
				test_settings = TestConfig()
			
			exp_dir_path = Path(test_settings.EXPERIMENT_DIR).expanduser()
			if exp_dir_path.is_dir():
				print("Loading config from run.")
				config = Config.parse_file(
					os.path.join(exp_dir_path, "run_parameters.json")
				)
			else:
				print(
					f"{test_settings.EXPERIMENT_DIR} not a directory. loading default config"
				)
				config = Config()
			
			# modify config according to test settings
			config.OUT_DIR = test_settings.EXPERIMENT_DIR
			config.VISUALIZE_OUTPUT = test_settings.VISUALIZE_OUTPUT
			return config
		else:
			print("No env_file or out_dir supplied. " "Creating default config.")
			return Config()


config = get_config()
