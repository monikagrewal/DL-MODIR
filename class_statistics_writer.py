import os
import numpy as np
import csv
import _pickle as pickle
import torch
import pdb

from functions_evaluation import compute_ud_2d, determine_non_dom_mo_sol, compute_hv_in_higher_dimensions

class StatisticsWriter():

    def __init__(self, mo_mode, output_file_name, ref_point, n_mo_sol, max_iter, grid_search_memory_saving, ud_eps=10.0**-5, conv_eps=10.0**-20):
        self.grid_search_memory_saving = grid_search_memory_saving
        self.output_file_name = output_file_name
        self.mo_mode = mo_mode
        # used to compute UD to points slightliy nudged across reference or domination boundary
        self.ud_eps = ud_eps
        # used to check individual mo-solution convergence
        self.conv_eps = conv_eps
        self.ref_point = ref_point
        # number of objectives
        self.n_mo_obj = len(ref_point)
        
        
        self.n_mo_sol = n_mo_sol
        self.max_iter = max_iter

        # USER SETTING
        self.debug_mode = False

        # initialize variables
        self.iter_number = -1
        self.eval_count = 0
        self.best_hv = -np.inf
        self.best_mean_hv_over_samples = -np.inf
        self.best_uhv = -np.inf
        self.best_iter = -np.inf
        self.mo_sol_convergence_count = np.zeros(self.n_mo_sol)

        self.best_validation_hv = -np.inf
        self.best_validation_mean_hv_over_samples = -np.inf
        self.best_validation_uhv = -np.inf
        self.best_validation_iter = -np.inf
        self.cur_iter_is_best_val_iter = False


        # initialize lists
        self.dynamic_weights_list = list()

        # initialize lists
        self.mo_obj_val_list = list()
        self.n_non_dom_list = list()
        self.uhv_list = list()
        self.hv_list = list()
        self.ud_list = list()

        self.mean_hv_over_samples_list = list()
        self.hv_per_sample_list = list()
        self.mo_obj_val_per_sample_list = list()
        self.mo_sol_is_dominated_list = list()
        
        # initialize lists
        self.validation_mo_obj_val_list = list()
        self.validation_n_non_dom_list = list()
        self.validation_uhv_list = list()
        self.validation_hv_list = list()
        self.validation_ud_list = list()
        self.validation_iter_list = list()

        self.validation_mean_hv_over_samples_list = list()
        self.validation_hv_per_sample_list = list()
        self.validation_mo_obj_val_per_sample_list = list()
        self.validation_mo_sol_is_dominated_list = list()

        # create output file with header
        self.create_statistics_file()

    def update_statistics(self, net_ensemble, epoch):
        """
        To be called at every iteration
        """
        # Extract fields from net_ensemble class (2 times redundant)
        self.mo_obj_val = net_ensemble.mo_obj_val
        self.mo_obj_val_per_sample = net_ensemble.mo_obj_val_per_sample
        dynamic_weights = net_ensemble.dynamic_weights

        self.epoch = epoch
        self.iter_number += 1
        self.eval_count += self.n_mo_sol
        self.n_non_dom, self.hv, self.hv_per_sample, self.mean_hv_over_samples, self.ud, self.uhv, self.mo_sol_is_dominated = self.compute_statistics(self.mo_obj_val, self.mo_obj_val_per_sample)
        # update lists
        self.mo_obj_val_list.append(self.mo_obj_val)
        if not self.grid_search_memory_saving:
            self.mo_obj_val_per_sample_list.append(self.mo_obj_val_per_sample) # used for plotting
            self.dynamic_weights_list.append(dynamic_weights)
        self.n_non_dom_list.append(self.n_non_dom)
        self.hv_list.append(self.hv)
        if not self.grid_search_memory_saving:
            self.hv_per_sample_list.append(self.hv_per_sample)
        self.mean_hv_over_samples_list.append(self.mean_hv_over_samples)
        self.ud_list.append(self.ud)
        self.uhv_list.append(self.uhv)
        self.mo_sol_is_dominated_list.append(self.mo_sol_is_dominated)

        self.check_convergence()
        # check whether new solution improves current best solution
        self.record_best()

        if self.iter_number == self.best_iter:
            cur_iter_is_best_iter = True
        else:
            cur_iter_is_best_iter = False

        ## only write results to file in selected iterations
        # the first 100 iters, every ten iters until iter 1000, every 1000 iters, close to max_iter
        log_cond1 = (self.iter_number < 100)
        log_cond2 = ( (self.iter_number < 1000) and (np.mod(self.iter_number,10) == 0) )
        log_cond3 = (np.mod(self.iter_number,1000) == 0)
        log_cond4 = (self.iter_number >= (self.max_iter-1))
        # log_cond5 = (self.iterations_without_best_hv_update_counter >= (self.best_hv_no_improvement_iteration_limit-1))
        if log_cond1 or log_cond2 or log_cond3 or log_cond4:
            self.write_iteration_output()

        if self.debug_mode:
            self.write_debugging_output()

        return(cur_iter_is_best_iter)

    def compute_statistics(self, mo_obj_val, mo_obj_val_per_sample):
        # check that the size and content of the set of solutions is as expected
        assert self.n_mo_obj == mo_obj_val.shape[0]
        assert self.n_mo_sol == mo_obj_val.shape[1]
        assert (not np.any(np.isnan(mo_obj_val)))
        assert np.all(mo_obj_val >= 0)
        assert np.all(mo_obj_val < np.infty)

        # check that the size and content of the losses per sample is as expected
        assert self.n_mo_obj == mo_obj_val_per_sample.shape[1]
        assert self.n_mo_sol == mo_obj_val_per_sample.shape[2]
        assert (not np.any(np.isnan(mo_obj_val_per_sample)))
        assert np.all(mo_obj_val_per_sample >= 0)
        assert np.all(mo_obj_val_per_sample < np.infty)

        ## compute size of non-dominated set
        # determine non-dom set
        _,mo_sol_is_dominated = determine_non_dom_mo_sol(mo_obj_val)
        # compute size
        n_non_dom = int(np.sum(mo_sol_is_dominated == False))
        assert type(n_non_dom) == int

        # compute HV
        hv = compute_hv_in_higher_dimensions(mo_obj_val,self.ref_point)
        assert type(hv) == float

        # compute mean HV over samples
        n_samples = mo_obj_val_per_sample.shape[0]
        hv_per_sample = np.zeros(n_samples)
        for i_sample in range(0,n_samples):
            hv_per_sample[i_sample] = compute_hv_in_higher_dimensions(mo_obj_val_per_sample[i_sample,:,:],self.ref_point)

        mean_hv_over_samples = float(np.mean(hv_per_sample))

        assert type(mean_hv_over_samples) == float

        # compute UD
        if self.n_mo_obj == 2:
            ud, _,_ = compute_ud_2d(mo_obj_val,self.ref_point,self.ud_eps)
        else:
            ud = -99
        ud = float(ud)
        assert type(ud) == float
        
        # compute UHV
        uhv = hv - ud
        assert type(uhv) == float

        return(n_non_dom,hv,hv_per_sample,mean_hv_over_samples,ud,uhv,mo_sol_is_dominated)

    def compute_and_record_validation_statistics(self, validation_mo_obj_val,validation_mo_obj_val_per_sample):

        (self.validation_n_non_dom,self.validation_hv,self.validation_hv_per_sample,self.validation_mean_hv_over_samples,self.validation_ud,self.validation_uhv,self.validation_mo_sol_is_dominated) = self.compute_statistics(validation_mo_obj_val,validation_mo_obj_val_per_sample)
        self.validation_mo_obj_val = validation_mo_obj_val
        self.validation_mo_obj_val_per_sample = validation_mo_obj_val_per_sample
        # update lists
        self.validation_mo_obj_val_list.append(validation_mo_obj_val)
        self.validation_mo_obj_val_per_sample_list.append(validation_mo_obj_val_per_sample)
        self.validation_iter_list.append(self.iter_number)

        self.validation_n_non_dom_list.append(self.validation_n_non_dom)
        self.validation_hv_list.append(self.validation_hv)
        self.validation_hv_per_sample_list.append(self.validation_hv_per_sample)
        self.validation_mean_hv_over_samples_list.append(self.validation_mean_hv_over_samples)
        self.validation_ud_list.append(self.validation_ud)
        self.validation_uhv_list.append(self.validation_uhv)
        self.validation_mo_sol_is_dominated_list.append(self.validation_mo_sol_is_dominated)

        self.record_best_validation()

    def check_convergence(self):
        # check convergence per solution
        for i_mo_sol in range(0,self.n_mo_sol):
            if (self.iter_number > 10) and np.sum(np.abs(self.mo_obj_val[:,i_mo_sol] - self.mo_obj_val_list[-1][:,i_mo_sol])) < self.conv_eps:
                self.mo_sol_convergence_count[i_mo_sol] += 1
            else:
                self.mo_sol_convergence_count[i_mo_sol] = 0

    def record_best(self):
        # select target metric based on mo_mode
        if self.mo_mode == 'loss_per_sample':
            current_target_metric = self.mean_hv_over_samples
            best_target_metric = self.best_mean_hv_over_samples
        elif self.mo_mode == 'mean_loss_over_samples':
            current_target_metric = self.hv
            best_target_metric = self.best_hv
        else:
            raise ValueError('Unknown mo_mode.')
        if current_target_metric >= best_target_metric:
            self.best_uhv = float(self.uhv)
            self.best_hv = float(self.hv)
            self.best_ud = float(self.ud)
            self.best_n_non_dom = self.n_non_dom
            self.best_mo_obj_val = self.mo_obj_val.copy()
            self.best_iter = self.iter_number
            self.best_mean_hv_over_samples = float(self.mean_hv_over_samples)
            self.best_mo_obj_val_per_sample = self.mo_obj_val_per_sample.copy()


    def record_best_validation(self):
        self.cur_iter_is_best_val_iter = False
        # select target metric based on mo_mode
        if self.mo_mode == 'loss_per_sample':
            current_validation_target_metric = self.validation_mean_hv_over_samples
            best_validation_target_metric = self.best_validation_mean_hv_over_samples
        elif self.mo_mode == 'mean_loss_over_samples':
            current_validation_target_metric = self.validation_hv
            best_validation_target_metric = self.best_validation_hv
        else:
            raise ValueError('Unknown mo_mode.')

        if current_validation_target_metric >= best_validation_target_metric:
            self.best_validation_uhv = float(self.validation_uhv)
            self.best_validation_hv = float(self.validation_hv)
            self.best_validation_ud = float(self.validation_ud)
            self.best_validation_n_non_dom = self.validation_n_non_dom
            self.best_validation_mo_obj_val = self.validation_mo_obj_val.copy()
            self.best_validation_iter = self.iter_number
            self.best_validation_mean_hv_over_samples = float(self.validation_mean_hv_over_samples)
            self.best_validation_mo_obj_val_per_sample = self.validation_mo_obj_val_per_sample.copy()
            self.cur_iter_is_best_val_iter = True


    def create_statistics_file(self):
        # create table with same format as in Stef's tables (values that are not computed are replaced by -99)
        header_row = ['Gen', 'Evals', 'Time', 'Best_f', 'Best_constr', 'Current_obj', 'Std_obj', 'Cur_constr', 'Std_constr', 'Best_HV', 'Best_IGD', 'Best_GD', 'size', 'Archive_HV', 'Archive_IGD', 'Archive_GD', 'Archive_size','Average_obj','Avg_constr']
        with open(self.output_file_name,'w') as file_handle:
            file_writer = csv.writer(file_handle, dialect = 'excel-tab')
            # add header
            file_writer.writerow(header_row)

    def write_iteration_output(self):
        # write output per generation
        with open(self.output_file_name,'a') as file_handle:
            file_writer = csv.writer(file_handle, dialect = 'excel-tab')
            cur_row = [self.iter_number, self.eval_count, -99, -99, -99, self.uhv,  -99, -99, -99, self.best_hv, -99, -99, self.best_n_non_dom, -99, -99, -99, -99, -99, -99]
            file_writer.writerow(cur_row)

    def save(self, net_ensemble, output_folder):
        """
        need to be called in sync with validation.
        different saving behavior between sin_cos and mtl/mo_seg;
        sin_cos save data, but mtl/mo_seg save indices
        """
        if self.grid_search_memory_saving:
            state_dicts = []
            training_data = None
            validation_data = None
            training_data_indices = None
            validation_data_indices = None
        elif hasattr(net_ensemble.validation_data, "indices"):
            state_dicts = [net.state_dict() for net in net_ensemble.net_list]
            training_data = None
            validation_data = None
            training_data_indices = net_ensemble.training_data.indices
            validation_data_indices = net_ensemble.validation_data.indices
        else:
            state_dicts = [net.state_dict() for net in net_ensemble.net_list]
            training_data = net_ensemble.training_data
            validation_data = net_ensemble.validation_data
            training_data_indices = None
            validation_data_indices = None
            
        checkpoint = {"stats": self,
                    "state_dicts": state_dicts,
                    "training_data": training_data,
                    "validation_data": validation_data,
                    "training_data_indices": training_data_indices,
                    "validation_data_indices": validation_data_indices}
        filepath = os.path.join(output_folder, 'checkpoint.pth')
        torch.save(checkpoint, filepath)

        if self.cur_iter_is_best_val_iter:
            filepath = os.path.join(output_folder, 'best_checkpoint.pth')
            # torch.save(checkpoint, filepath)