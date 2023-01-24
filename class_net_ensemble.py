import numpy as np
import torch
import pdb
import copy
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from importlib import import_module
from functools import partial

import utils as main_utils
from mo_optimizers import linear_scalarization, uhv, higamo_hv, pareto_mtl, epo


def sanity_check(val, min_val=0, max_val=np.infty):
    assert (not np.any(np.isnan(val)))
    assert np.all(val >= min_val)
    assert np.all(val < max_val)    
    return None


class NetEnsemble():
    def __init__(self, target_device, cfg):
        self.cfg = cfg
        self.name = cfg["problem_name"]
        self.meta_optimizer_name = cfg["meta_optimizer_name"]
        self.meta_optimizer_scheduler_name = cfg["meta_optimizer_scheduler_name"]
        model = import_module(".{}.model".format(self.name), "problems")
        self.net_list = []
        self.meta_optimizer_list = list()
        self.meta_optimizer_scheduler_list = list()
        self.n_mo_sol = cfg["n_mo_sol"]
        
        self.target_device = target_device
        self.ref_point = np.array(cfg["ref_point"])
        self.n_mo_obj = len(self.ref_point)
        self.net_list = list()
        self.dynamic_weights_list = list()

        self.mo_obj_val_per_sample_list = list()
        # used in the validation after each epoch to check whether the new net_list is better than the current
        self.best_validation_hv = -1 * np.inf # mo_mode: mean_loss_over_samples
        self.best_validation_mean_hv_over_samples = -1 * np.inf # mo_mode: loss_per_sample

        for i_mo_sol in range(0, self.n_mo_sol):
            self.net_list.append(model.get_network(cfg["network"], target_device=self.target_device, **cfg["network_params"]))
            self.net_list[i_mo_sol].update_device(self.target_device)
            # style transfer needs params to be placed in a list
            if self.name == 'vincent_van_jarjar':
                params = [self.net_list[i_mo_sol].params]
            else:
                params = self.net_list[i_mo_sol].params

            if self.meta_optimizer_name == 'Adam':
                meta_optimizer = torch.optim.Adam(params,**cfg['meta_optimizer_params'])
            elif self.meta_optimizer_name == 'SGD':
                meta_optimizer = torch.optim.SGD(params,**cfg['meta_optimizer_params'])
            elif self.meta_optimizer_name == 'LBFGS':
                meta_optimizer = torch.optim.LBFGS(params,**cfg['meta_optimizer_params'])
            else:
                raise ValueError('Unknown meta_optimizer_name.')
            self.meta_optimizer_list.append(meta_optimizer)
            
            if self.meta_optimizer_scheduler_name == 'StepLR':
                meta_optimizer_scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimizer_list[i_mo_sol],**cfg['meta_optimizer_scheduler_params'])
            elif self.meta_optimizer_scheduler_name == 'MultiStepLR':
                meta_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.meta_optimizer_list[i_mo_sol],**cfg['meta_optimizer_scheduler_params'])
            else:
                raise ValueError('Unknown meta_optimizer_scheduler_name.')
            self.meta_optimizer_scheduler_list.append(meta_optimizer_scheduler)


        # count the number of parameters in all networks. If some do not have 'requires_grad' then this breaks (and probably more stuff in this code)
        self.n_parameters_list = list()
        for i_mo_sol in range(0, self.n_mo_sol):
            self.n_parameters_list.append(int(np.sum([cur_par.numel() for cur_par in self.net_list[i_mo_sol].params])))
        # check that all networks have the same number of parameters
        assert np.all([x == self.n_parameters_list[0] for x in self.n_parameters_list])
        self.n_parameters = self.n_parameters_list[0]

        data_module = import_module(".{}.data".format(self.name), "problems")
        self.training_data = data_module.get_dataset(cfg["dataset"], train=True, **cfg["data_params"])
        self.validation_data = data_module.get_dataset(cfg["dataset"], train=False, **cfg["data_params"])
        self.training_data, self.validation_data = main_utils.train_and_val_split(self.training_data, self.validation_data,\
                                     train_ratio=cfg["train_ratio"])

        self.training_dataloader = DataLoader(self.training_data, batch_size=cfg["batch_size"], shuffle=True)
        batchsize = max(1, min(cfg["batch_size"], len(self.validation_data)))
        self.validation_dataloader = DataLoader(self.validation_data, batch_size=batchsize, shuffle=False)
        
        losses = import_module("{}.losses".format(self.name), "problems")
        if self.name == "vincent_van_jarjar":
            # style transfer needs set number
            self.obj_func = losses.Loss(cfg["network_params"]["set_nr"], target_device, cfg["obj_func"])
        else:
            self.obj_func = losses.Loss(target_device, cfg["obj_func"])
        
        utils = import_module("{}.utils".format(self.name), "problems")
        self.postprocess_fn = utils.postprocess
        self.validation_fn = utils.validation

        # be careful if you decide to initialize it not in the end
        self.mo_opt = self.initialize_mo_optimizer(cfg)


    def initialize_mo_optimizer(self, cfg):
        """
        calls init method of respective mo_optimizer class with appropriate args
        """
        opt_name = cfg["mo_optimizer"]
        ref_point = cfg["ref_point"]
        n_mo_sol = self.n_mo_sol
        n_mo_obj = self.n_mo_obj
        mo_optimizer_params = cfg.get("mo_optimizer_params", {})

        if opt_name == "linear_scalarization":
            mo_opt = linear_scalarization.LinearScalarization(n_mo_sol, n_mo_obj, **mo_optimizer_params)
        elif opt_name == "uhv":
            mo_opt = uhv.UHV(n_mo_sol, n_mo_obj, ref_point, **mo_optimizer_params)
        elif opt_name == "higamo_hv":
            mo_opt = higamo_hv.HigamoHv(n_mo_sol, n_mo_obj, ref_point, **mo_optimizer_params)
        elif opt_name == "pareto_mtl":
            if cfg["mo_mode"] == 'loss_per_sample':
                raise NotImplementedError('pareto MTL not implemented for mo_mode loss_per_sample')
            mo_opt = pareto_mtl.ParetoMTL(n_mo_sol, n_mo_obj, device=self.target_device, **mo_optimizer_params)
        elif opt_name == "epo":
            if cfg["mo_mode"] == 'loss_per_sample':
                raise NotImplementedError('EPO not implemented for mo_mode loss_per_sample')
            mo_opt = epo.EPO(n_mo_sol, n_mo_obj, self.n_parameters_list, **mo_optimizer_params)
        else:
            raise ValueError("unknown opt name")
        return mo_opt


    def dynamic_weight_optimization_mean(self, data_batch):
        """
        maximize the average (for each batch) HV over all solutions
        """
        opt_name = self.mo_opt.name
        inputs = data_batch['X']
        labels = data_batch['Y']
        # inputs = data_batch['X'].to(self.target_device)
        # labels = data_batch['Y'].to(self.target_device)

        # dynamic weight calculation in epo and pareto MTL requires gradients, so needs extra forward & backward propagation
        if opt_name in ['pareto_mtl', 'epo']:
            dynamic_weights = self.mo_opt.compute_weights(self.net_list,
                                                        self.meta_optimizer_list,
                                                        self.obj_func,
                                                        inputs,
                                                        labels)

        # forward propagation
        _, _, mo_obj_val_torch, mo_obj_val_mean = self.forward_propagation(inputs, labels)

        # compute dynamic weights
        if opt_name in ['uhv', 'higamo_hv', 'linear_scalarization']:
            dynamic_weights = self.mo_opt.compute_weights(mo_obj_val_mean)

        self.dynamic_weights = dynamic_weights.cpu().numpy()
        self.dynamic_weights_list.append(dynamic_weights)
        dynamic_weights = dynamic_weights.to(self.target_device)

        # backward propagation
        for i_mo_sol in range(0, self.n_mo_sol):
            weighted_objective = torch.sum(dynamic_weights[:, i_mo_sol] * mo_obj_val_torch[i_mo_sol])
            weighted_objective.backward()

        # optimizer step
        self.optimizer_step(inputs, labels)

        return None


    def dynamic_weight_optimization_per_sample(self, data_batch):
        """
        maximize the HV of all samples separately
        """
        opt_name = self.mo_opt.name
        inputs = data_batch['X']
        labels = data_batch['Y']
        n_samples = self.cfg["batch_size"]
        
        # dynamic weight calculation in epo and pareto MTL
        if opt_name in ['pareto_mtl', 'epo']:
            raise ValueError("dynamic weight optimization per sample does not make sense for ParetoMTL and EPO (imo)")

        # forward propagation
        mo_obj_val_torch_per_sample, mo_obj_val_per_sample, _, _ = self.forward_propagation(inputs, labels)

        # compute dynamic weights per sample
        dynamic_weights_per_sample = torch.ones(self.n_mo_sol, self.n_mo_obj, n_samples)
        for i_sample in range(0, n_samples):
            if opt_name in ['uhv', 'higamo_hv', 'linear_scalarization']:
                weights = self.mo_opt.compute_weights(mo_obj_val_per_sample[i_sample,:,:])
            dynamic_weights_per_sample[:, :, i_sample] = weights.permute(1,0)

        dynamic_weights_per_sample = dynamic_weights_per_sample.to(self.target_device)
        dynamic_weights = torch.mean(dynamic_weights_per_sample, dim=2)
        dynamic_weights = dynamic_weights.permute(1, 0)
        self.dynamic_weights = dynamic_weights.cpu().numpy()
        self.dynamic_weights_list.append(dynamic_weights)

        # backward propagation
        for i_mo_sol in range(0, self.n_mo_sol):
            # adding torch.mean, does it have an effect on the step size? with mean, the magnitude of weighted_objective for both mo_modes is the same
            weighted_objective = torch.sum(torch.mean(dynamic_weights_per_sample[i_mo_sol, :, :] * mo_obj_val_torch_per_sample[i_mo_sol], dim=1))
            weighted_objective.backward()

        # optimizer step
        self.optimizer_step(inputs, labels)

        return None


    def forward_propagation(self, inputs, labels):
        """
        compute mo_obj_val per sample and the mo_obj_val_mean (mo_obj_val averaged over all samples)
        """
        n_samples = inputs[0].shape[0]
        mo_obj_val_torch_per_sample = list()
        mo_obj_val_torch = list()
        mo_obj_val_per_sample = np.zeros((n_samples, self.n_mo_obj, self.n_mo_sol))
        mo_obj_val_mean = np.zeros((self.n_mo_obj, self.n_mo_sol))
        for i_mo_sol in range(0, self.n_mo_sol):
            self.meta_optimizer_list[i_mo_sol].zero_grad()
            
            Y_hat = self.net_list[i_mo_sol](inputs)
            loss_per_sample = self.obj_func(Y_hat, labels)
            loss_per_sample = torch.stack(loss_per_sample, dim=0)
            loss_mean = loss_per_sample.mean(dim=1).view(-1)

            mo_obj_val_torch_per_sample.append(loss_per_sample)
            mo_obj_val_torch.append(loss_mean)
            
            mo_obj_val_per_sample[:, :, i_mo_sol] = loss_per_sample.cpu().detach().numpy().T
            mo_obj_val_mean[:, i_mo_sol] = loss_mean.cpu().detach().numpy()

        # check validity of mo_obj_val_mean & mo_obj_val_per_sample
        sanity_check(mo_obj_val_mean)
        sanity_check(mo_obj_val_per_sample)

        self.mo_obj_val = mo_obj_val_mean
        self.mo_obj_val_per_sample = mo_obj_val_per_sample
        self.mo_obj_val_per_sample_list.append(mo_obj_val_per_sample)

        outputs = (mo_obj_val_torch_per_sample, mo_obj_val_per_sample, mo_obj_val_torch, mo_obj_val_mean)
        return outputs


    def optimizer_step(self, input_data_batch, label_batch):
        # treat neural style case separately
        closure_list = None
        if self.meta_optimizer_name == 'LBFGS':
            closure_list = self.create_list_of_closures(input_data_batch, label_batch)
        
        self.do_so_step(closure_list)

        # special case for vincent van jarjar, where the params are image pixels
        if self.name == "vincent_van_jarjar":
            for i_mo_sol in range(self.n_mo_sol):
                self.net_list[i_mo_sol].params.data.clamp_(0, 1)
        return None


    def do_so_step(self, closure_list=None):
        # update weights for each mo-solution
        for i_mo_sol in range(0,self.n_mo_sol):
            if closure_list is None:
                self.meta_optimizer_list[i_mo_sol].step()
            else:
                closure = closure_list[i_mo_sol]
                self.meta_optimizer_list[i_mo_sol].step(closure)
        # weights have changed, so update the weights stored in separate array 'self.weights'
        # self.collect_weights()
        return None


    def create_list_of_closures(self,input_data_batch,label_batch):
        # create list of closures (1 for each neural net; each net depends on different dynamic weights)
        closure_list = list()
        # load closure function
        temp_closure = self.opt_closure
        for i_mo_sol in range(0, self.n_mo_sol):
            # get last known dynamic weights
            cur_dyn_weight = self.dynamic_weights_list[-1][:,i_mo_sol]
            # send to target device
            cur_dyn_weight = cur_dyn_weight.to(self.target_device)
            # fix function inputs to create closure without inputs
            partial_neural_style_closure = partial(temp_closure, net_ensemble=self, i_mo_sol=i_mo_sol, input_data_batch=input_data_batch, label_batch=label_batch, obj_weight=cur_dyn_weight)
            closure_list.append(partial_neural_style_closure)
        return(closure_list)


    def opt_closure(self,i_mo_sol,input_data_batch,label_batch,obj_weight):
        # the closure needs to allow evaluating new mo-solutions; dynamic gradient weights are fixed at the last computed value
        mo_obj_val_torch = list()
        # HACK clamping of input image for style transfer needs to be done before zero_grad, I haven't found a way to fit it into the problem folder
        if self.name == 'vincent_van_jarjar':
            self.net_list[i_mo_sol].params.data.clamp_(0,1)
        # set grad to zero
        self.meta_optimizer_list[i_mo_sol].zero_grad()
        # pass data through network
        Y_hat = self.net_list[i_mo_sol](input_data_batch)
        # compute losses
        loss = self.obj_func(Y_hat, label_batch)
        ## compute HV objective which produces HV gradients
        # iteratively add dynamically weighted objectives
        for i_mo_obj in range(0,self.n_mo_obj):
            if i_mo_obj == 0:
                weighted_objective = obj_weight[i_mo_obj] * loss[i_mo_obj]
            else:
                weighted_objective = weighted_objective + obj_weight[i_mo_obj] * loss[i_mo_obj]
        # backprop
        weighted_objective.backward(retain_graph=True)
        # HACK for style transfer: clip gradients. This needs to be tuned per problem. I haven't found a way to fit it into the problem folder
        if self.name == 'vincent_van_jarjar':
            torch.nn.utils.clip_grad_norm_(self.net_list[i_mo_sol].params,0.1)
        # print gradients for debugging gradient clipping
        # print(torch.norm((net_ensemble.net_list[0].params.grad)))
        return(weighted_objective)

       
    def collect_weights(self):
        # get network weights from all networks and store them in a numpy array
        self.weights = np.zeros((self.n_parameters, self.n_mo_sol))
        for i_mo_sol in range(0,self.n_mo_sol):
            par_list = list()
            for cur_par in self.net_list[i_mo_sol].params:
                par_list.append(cur_par.flatten().detach().cpu().numpy())
            self.weights[:,i_mo_sol] = np.concatenate(par_list)

        return None