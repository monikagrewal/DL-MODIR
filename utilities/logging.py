import os
from typing import Dict, List
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def create_subfolders(
    root_folder: str, foldernames: Dict, cache: object = None
) -> None:

    for name, value in foldernames.items():
        folderpath = os.path.join(root_folder, value)
        os.makedirs(folderpath, exist_ok=True)
        if cache is not None:
            cache.__setattr__(name, folderpath)


def log_iteration_metrics(
    metrics: Dict, 
    steps: int,
    writer: SummaryWriter,
    data: str = "train",
    loss_functions: List[str] = []
) -> None:

    """
    assuming arrays to be samples (optional) * n_obj * n_sol
    """
    for metric_name, metric_val in metrics.items():
        if metric_val.ndim == 3: # assuming first axis to be samples
            metric_val = metric_val.mean(axis=0)
        
        if metric_val.ndim==1: # assuming 1 value per sample
            writer.add_scalar(f"{data}/{metric_name}", metric_val.mean(), steps)
        elif metric_val.ndim==2: # assuming n_obj * n_sol
            if len(loss_functions)==0:
                loss_functions = [f"Loss{i}" for i in range(metric_val.shape[0])]
            else:
                assert len(loss_functions)==metric_val.shape[0]
            for i_obj, loss in enumerate(loss_functions):
                for i_sol in range(metric_val.shape[1]):
                    writer.add_scalar(f"{data}/{metric_name}/{loss}/Pred{i_sol}", metric_val[i_obj, i_sol], steps)
        else:
            logging.warning("metric value has more than 3 dimensions."\
                "don't know what that represents.")

    return writer


def save_checkpoint(net_ensemble, cache, filename="checkpoint"):
    state_dicts = [net.state_dict() for net in net_ensemble.net_list]
    checkpoint = {"state_dicts": state_dicts,
                    "epoch": cache.epoch,
                    "iter": cache.iter,
                    "hv": cache.hv,
                    "best_hv": cache.best_hv,
                    "best_iter": cache.best_iter}
    filepath = os.path.join(cache.out_dir_weights, f"{filename}.pth")
    torch.save(checkpoint, filepath)


def define_chart_layout(n_mo_sol: int, n_mo_obj: int, loss_functions: List[str]) -> Dict:
    root_tags = ["train/loss",
                "train/dynamic_weights",
                "val/loss"
                ]
    layout = {}
    for root_tag in root_tags:
        main_chart = {}
        for loss in loss_functions:
            charts_to_gather = []
            for i_sol in range(n_mo_sol):
                charts_to_gather.append(f"{root_tag}/{loss}/Pred{i_sol}")
            main_chart[loss] = ['Multiline', charts_to_gather]
        
        layout[root_tag] = main_chart

    return layout