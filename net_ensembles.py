import numpy as np
import torch
from torch import nn, optim
import pdb
import copy
from typing import Callable, Dict, List, Union, Tuple

from torch.utils.data import DataLoader
from importlib import import_module
from functools import partial


class NetEnsemble():
    def __init__(self, config, model_fn, optimizer_fn, lr_scheduler_fn):
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