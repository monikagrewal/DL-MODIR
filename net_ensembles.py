import numpy as np
import torch
from torch import nn, optim
import pdb
import copy
from typing import Callable, Dict, List, Union, Tuple

from torch.utils.data import DataLoader
import logging


class DeepEnsemble():
    def __init__(self, config, model_fn, optimizer_fn=None, lr_scheduler_fn=None, weights_path=None):
        self.name = config.PROBLEM_NAME
        # model = import_module(".{}.model".format(self.name), "problems")
        self.n_mo_sol = config.N_MO_SOL
        self.n_mo_obj = config.N_MO_OBJ
        self.net_list = list()
        self.optimizer_list = list()
        self.lr_scheduler_list = list()

        for i_mo_sol in range(0, config.N_MO_SOL):
            # model
            model = model_fn(config.MODEL_NAME, target_device=config.DEVICE, **config.MODEL_PARAMS)
            self.net_list.append(model)
            self.net_list[i_mo_sol].update_device(config.DEVICE)
            # style transfer needs params to be placed in a list
            if self.name == 'vincent_van_jarjar':
                params = [model.params]
            else:
                params = model.params
            
            if optimizer_fn is not None and lr_scheduler_fn is not None: #None during testing
                # optimizer
                optimizer = optimizer_fn(params,
                                            lr=config.LR,
                                            weight_decay=config.WEIGHT_DECAY,
                                            **config.OPTIMIZER_PARAMS,
                                            )
                self.optimizer_list.append(optimizer)

                # lr_scheduler
                lr_scheduler = lr_scheduler_fn(optimizer,**config.LR_SCHEDULER_PARAMS)
                self.lr_scheduler_list.append(lr_scheduler)


        # count the number of parameters in all networks. If some do not have 'requires_grad' then this breaks (and probably more stuff in this code)
        self.n_parameters_list = list()
        for i_mo_sol in range(0, config.N_MO_SOL):
            self.n_parameters_list.append(int(np.sum([cur_par.numel() for cur_par in self.net_list[i_mo_sol].params])))
        # check that all networks have the same number of parameters
        assert np.all([x == self.n_parameters_list[0] for x in self.n_parameters_list])
        self.n_parameters = self.n_parameters_list[0]

        # load weights
        if weights_path is not None:
            state_dicts = torch.load(
                weights_path,
                map_location=config.DEVICE,
            )["state_dicts"]

            for i, state_dict in enumerate(state_dicts):
                self.net_list[i].load_state_dict(state_dict)
            logging.info("Weights loaded")


class KHeadEnsemble():
    """
    wrapper over khead network for MO training
    """
    def __init__(self, config, model_fn, optimizer_fn=None, lr_scheduler_fn=None, weights_path=None):
        self.name = config.PROBLEM_NAME
        # model = import_module(".{}.model".format(self.name), "problems")
        self.n_mo_sol = config.N_MO_SOL
        self.n_mo_obj = config.N_MO_OBJ

        # model
        self.model = model_fn(config.MODEL_NAME, target_device=config.DEVICE, K=config.N_MO_SOL, **config.MODEL_PARAMS)
        self.model.update_device(config.DEVICE)
        # style transfer needs params to be placed in a list
        if self.name == 'vincent_van_jarjar':
            params = [self.model.params]
        else:
            params = self.model.params
        
        if optimizer_fn is not None and lr_scheduler_fn is not None: #None during testing
            # optimizer
            self.optimizer = optimizer_fn(params,
                                        lr=config.LR,
                                        weight_decay=config.WEIGHT_DECAY,
                                        **config.OPTIMIZER_PARAMS,
                                        )

            # lr_scheduler
            self.lr_scheduler = lr_scheduler_fn(self.optimizer, **config.LR_SCHEDULER_PARAMS)

        # count the number of parameters in all networks. If some do not have 'requires_grad' then this breaks (and probably more stuff in this code)
        self.n_parameters = int(np.sum([cur_par.numel() for cur_par in self.model.params]))

        # load weights
        if weights_path is not None:
            state_dict = torch.load(
                weights_path,
                map_location=config.DEVICE,
            )["state_dicts"]

            self.model.load_state_dict(state_dict)
            logging.info("Weights loaded")