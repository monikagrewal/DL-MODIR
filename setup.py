import json
import logging
import os
import re
from copy import deepcopy
from typing import Callable, Dict, List, Union, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from config import config
from runtime_cache import RuntimeCache
from mo_optimizers import linear_scalarization, higamo_hv
from net_ensembles import DeepEnsemble, KHeadEnsemble
from training import train
from utilities.logging import define_chart_layout


def load_problem(mode="train") -> Tuple:
    if config.PROBLEM_NAME=="mo_regression":
        logging.debug("Problem name: mo_regression")
        from problems.mo_regression import data, model, losses, inference
    elif config.PROBLEM_NAME=="modir":
        logging.debug("Problem name: modir")
        from problems.modir import data, model, losses, inference
    elif config.PROBLEM_NAME=="modir3d":
        logging.debug("Problem name: modir3d")
        from problems.modir3d import data, model, losses, inference
    elif config.PROBLEM_NAME=="dtlz":
        logging.debug("Problem name: dtlz")
        from problems.DTLZ import data, model, losses, inference
    elif config.PROBLEM_NAME=="genmed":
        logging.debug("Problem name: genmed")
        from problems.genmed import data, model, losses, inference
    elif config.PROBLEM_NAME=="zdt":
        logging.debug("Problem name: zdt")
        from problems.ZDT import data, model, losses, inference
    else:
        raise ValueError(f"PROBLEM_NAME = {config.PROBLEM_NAME} not identified.")
    
    dataset_fn = data.get_dataset
    model_fn = model.get_network
    criterion = losses.Loss(config.LOSS_FUNCTIONS)
    if mode=="train":
        validation_fn = inference.validation
    elif mode=="test":
        validation_fn = inference.testing
    else:
        logging.error("unknown mode.")
    return dataset_fn, model_fn, criterion, validation_fn


def get_datasets(get_dataset: Callable) -> List[Dict[str, Dataset]]:
    """
    Assumption: this function will be used only during training

    nfolds = 0 (all data in train),
            1 (hold out validation set with 80:20 split)
            >=2 (N/nfolds splits, where N is total data)
    """
    # call get_dataset to have two copies of full dataset,
    # useful when train and validation use differetn transform pipelines
    full_dataset_train = get_dataset(config.DATASET, train=True, **config.DATA_PARAMS)
    full_dataset_val = get_dataset(config.DATASET, train=False, **config.DATA_PARAMS)

    N = len(full_dataset_train)
    # No folds, return full dataset
    nfolds = config.NFOLDS
    if nfolds == 0:
        logging.info(f"NFolds = {nfolds}: Full dataset")
        indices = np.arange(N)
        train_dataset = deepcopy(full_dataset_train).partition(indices)
        val_dataset = deepcopy(full_dataset_val).partition([])
        datasets_list = [
            {
                "train": train_dataset,
                "val": val_dataset,
            }
        ]

    # Basic single holdout validation
    elif nfolds == 1:
        logging.info(f"NFolds = {1}: Single holdout")
        indices = np.arange(N)
        np.random.shuffle(indices)
        ntrain = int(N * 0.80)
        train_indices = indices[:ntrain]
        val_indices = indices[ntrain:]

        train_dataset = deepcopy(full_dataset_train).partition(train_indices)
        val_dataset = deepcopy(full_dataset_val).partition(val_indices)

        datasets_list = [{"train": train_dataset, "val": val_dataset}]

    # K-Fold
    elif nfolds >= 2:
        logging.info(f"NFolds = {nfolds}: K-Fold")
        datasets_list = []
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=config.RANDOM_SEED)
        for train_indices, val_indices in kf.split(full_dataset_train):
            train_dataset = deepcopy(full_dataset_train).partition(train_indices)
            val_dataset = deepcopy(full_dataset_val).partition(val_indices)
    
            datasets_list.append({"train": train_dataset, "val": val_dataset})

    return datasets_list


def get_dataloaders(datasets: Dict[str, Dataset]) -> Dict[str, DataLoader]:

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.BATCHSIZE, num_workers=3
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
    }


def get_mo_optimizer() -> object:
    """
    calls init method of respective mo_optimizer class with appropriate args
    """
    opt_name = config.MO_OPTIMIZER
    ref_point = config.REF_POINT
    n_mo_sol = config.N_MO_SOL
    n_mo_obj = config.N_MO_OBJ
    mo_optimizer_params = config.MO_OPTIMIZER_PARAMS

    if opt_name == "linear_scalarization":
        mo_opt = linear_scalarization.LinearScalarization(n_mo_sol, n_mo_obj, **mo_optimizer_params)
    elif opt_name == "higamo_hv":
        # hack for mtl version. remove
        if n_mo_obj==4:
            n_mo_obj = 3
            ref_point = ref_point[:3]
        mo_opt = higamo_hv.HigamoHv(n_mo_sol, n_mo_obj, ref_point, **mo_optimizer_params)
    elif opt_name == "pareto_mtl":
        if config.MO_MODE == 'loss_per_sample':
            raise NotImplementedError('pareto MTL not implemented for mo_mode loss_per_sample')
        mo_opt = pareto_mtl.ParetoMTL(n_mo_sol, n_mo_obj, device=config.DEVICE, **mo_optimizer_params)
    elif opt_name == "epo":
        if config.MO_MODE == 'loss_per_sample':
            raise NotImplementedError('EPO not implemented for mo_mode loss_per_sample')
        # mo_opt = epo.EPO(n_mo_sol, n_mo_obj, self.n_parameters_list, **mo_optimizer_params) # TODO: fix this
        mo_opt = epo.EPO(n_mo_sol, n_mo_obj, **mo_optimizer_params)
    else:
        raise ValueError("unknown opt name")
    return mo_opt


def get_optimizer() -> Callable:
    if config.OPTIMIZER == "SGD":
        optimizer = optim.SGD
    elif config.OPTIMIZER == "Adam":
        optimizer = optim.Adam
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    return optimizer


def get_lr_scheduler() -> Callable:
    scheduler = optim.lr_scheduler
    if config.LR_SCHEDULER == "StepLR":
        scheduler = optim.lr_scheduler.StepLR
    elif config.LR_SCHEDULER == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR
    elif config.LR_SCHEDULER == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR
    else:
        raise ValueError(f"Unknown lr scheduler: {config.LR_SCHEDULER}")
    return scheduler


def setup_train():
    # make experiment dir
    os.makedirs(config.OUT_DIR, exist_ok=True)

    # save config file
    with open(os.path.join(config.OUT_DIR, "run_parameters.json"), "w") as file:
        json.dump(config.dict(), file, indent=4)
    logging.info(f"CONFIG: {config.dict()}")

    # get train procedures
    dataset_fn, model_fn, criterion, validation_fn = load_problem()
    optimizer_fn = get_optimizer()
    lr_scheduler_fn = get_lr_scheduler()
    mo_optimizer = get_mo_optimizer()

    # Set seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

    # Load datasets
    datasets_list = get_datasets(dataset_fn)

    for i_fold, datasets in enumerate(datasets_list):
        # Create fold folder
        fold_dir = os.path.join(config.OUT_DIR, f"fold{i_fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Set seed again for dataloader reproducibility (probably unnecessary)
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        dataloaders = get_dataloaders(datasets)

        for i_run in range(config.NRUNS):
            ntrain, nval = len(datasets["train"]), len(datasets["val"])
            logging.info(f"Run: {i_run}, Fold: {i_fold}")
            logging.info(f"Total train dataset: {ntrain}")
            logging.info(f"Total validation dataset: {nval}")

            # Create run folder and set-up run dir
            run_dir = os.path.join(fold_dir, f"run{i_run}")
            os.makedirs(run_dir, exist_ok=True)

            # Intermediate results storage to pass to other functions to reduce parameters
            cache = RuntimeCache(mode="train")
            #  Create subfolders
            foldernames = config.FOLDERNAMES
            cache.create_subfolders(run_dir, foldernames)
            # Logging of training progress
            writer = SummaryWriter(run_dir)
            layout = define_chart_layout(config.N_MO_SOL, config.N_MO_OBJ, config.LOSS_FUNCTIONS)
            writer.add_custom_scalars(layout)

            # Change seed for each run
            torch.manual_seed(config.RANDOM_SEED + i_run)
            np.random.seed(config.RANDOM_SEED + i_run)

            # Initialize parameters
            if config.ENSEMBLE_TYPE=="deep":
                net_ensemble = DeepEnsemble(config, model_fn, optimizer_fn, lr_scheduler_fn)
            elif config.ENSEMBLE_TYPE=="khead":
                net_ensemble = KHeadEnsemble(config, model_fn, optimizer_fn, lr_scheduler_fn)
            else:
                raise ValueError("unknown ensemble type: {config.ENSEMBLE_TYPE}")

            # Mixed precision training scaler
            scaler = torch.cuda.amp.GradScaler()

            # Training
            train(mo_optimizer,
                net_ensemble,
                criterion,
                validation_fn,
                scaler,
                dataloaders,
                cache,
                writer,
            )

            # Delete cache in the end. Q. is it necessary?
            del cache
            writer.close()


def setup_test():
    from testing import main as test

    # load dataset
    dataset_fn, model_fn, criterion, test_fn = load_problem(mode="test")
    test_dataset = dataset_fn(config.DATASET, train=False, **config.DATA_PARAMS)
    logging.info(f"Total dataset: {len(test_dataset)}")

    dataloader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.BATCHSIZE, num_workers=3
    )

    if config.NFOLDS==0:
        NFOLDS = 1
    else:
        NFOLDS = config.NFOLDS
    for i_fold in range(NFOLDS):
        for i_run in range(config.NRUNS):
            logging.info(f"Run: {i_run}, Fold: {i_fold}")
            # initialize cache and summarywriter
            cache = RuntimeCache(mode="test")
            run_dir = os.path.join(config.OUT_DIR, f"fold{i_fold}", f"run{i_run}")
            cache.set_subfolder_names(run_dir, config.FOLDERNAMES)
            cache.out_dir_test = os.path.join(run_dir, "test")
            if not os.path.exists(cache.out_dir_test):
                os.makedirs(cache.out_dir_test, exist_ok=True)
            writer = SummaryWriter(cache.out_dir_test)

            # Initialize parameters and load weights
            weights_path = os.path.join(
                cache.out_dir_weights, f"checkpoint_{config.SAVE_MODEL}.pth"
            )
            if config.ENSEMBLE_TYPE=="deep":
                net_ensemble = DeepEnsemble(config, model_fn, weights_path=weights_path)
            elif config.ENSEMBLE_TYPE=="khead":
                net_ensemble = KHeadEnsemble(config, model_fn, weights_path=weights_path)
            else:
                raise ValueError("unknown ensemble type: {config.ENSEMBLE_TYPE}")
            logging.info("Model initialized for testing")

            test(test_fn, net_ensemble, dataloader, criterion, cache, writer)

