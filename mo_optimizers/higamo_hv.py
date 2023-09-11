import numpy as np
import torch

from functions.functions_evaluation import fastNonDominatedSort
from functions.functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling


class HigamoHv(object):
    """
    Mo optimizer for calculating dynamic weights using higamo style hv maximization
    based on Hao Wang et al.'s HIGA-MO
    uses non-dominated sorting to create multiple fronts, and maximize hypervolume of each
    """
    def __init__(self, n_mo_sol, n_mo_obj, ref_point, beta_one, obj_space_normalize=True, **kwargs):
        super(HigamoHv, self).__init__()
        self.name = 'higamo_hv'
        self.ref_point = np.array(ref_point, dtype=np.float32)
        self.dyn_ref_point = self.ref_point
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj
        self.iter_counter = 0.0 # float because we will take powers and divide (is that only an issue in C?)
        self.adam_eps = 10**(-8.0)
        self.beta_one = beta_one
        self.weight_mean = None
        self.obj_space_normalize = obj_space_normalize
        self.adaptive_constraint = kwargs.get("adaptive_constraint", False)
        self.adaptive_constraint_iter = kwargs.get("adaptive_constraint_iter", 300)
        self.constrained_obj = kwargs.get("constrained_obj", (0))
        self.weighted_hypervolume = kwargs.get("weighted_hypervolume", False)
        
        default_gamma = [0 for i in range(n_mo_obj)]
        gamma = kwargs.get("gamma", default_gamma)
        gamma = np.repeat(gamma, n_mo_sol).reshape(n_mo_obj, n_mo_sol)
        self.gamma = np.array(gamma)



    def compute_weights(self, mo_obj_val):
        n_mo_obj = self.n_mo_obj
        n_mo_sol = self.n_mo_sol
        self.iter_counter = self.iter_counter + 1

        # # compute dynamic ref point for gradients (choose maximum of user-defined ref point and 1.1 times the worst observed loss per objective)
        # dyn_ref_point =  1.1 * np.max(mo_obj_val, axis=1)
        # for i_obj in range(0,n_mo_obj):
        #     dyn_ref_point[i_obj] = np.maximum(self.ref_point[i_obj], dyn_ref_point[i_obj])
        if self.adaptive_constraint and \
            self.iter_counter > self.adaptive_constraint_iter:

            for obj_idx in self.constrained_obj:
                self.dyn_ref_point[obj_idx] = \
                    np.minimum(4*np.min(
                    mo_obj_val[obj_idx, :]), self.dyn_ref_point[obj_idx]
                    )

        # clamp loss values to ref point
        for i_obj in range(0, n_mo_obj):
            mo_obj_val[i_obj, :] = np.clip(mo_obj_val[i_obj, :], a_min=0, a_max=0.99*self.dyn_ref_point[i_obj])
        
        # non-dom sorting to create multiple fronts
        hv_subfront_indices = fastNonDominatedSort(mo_obj_val)
        number_of_fronts = np.max(hv_subfront_indices) + 1 # +1 because of 0 indexing
        obj_space_multifront_hv_gradient = np.zeros((n_mo_obj, n_mo_sol))
        for i_fronts in range(0,number_of_fronts):
            # compute HV gradients for current front
            temp_grad_array = grad_multi_sweep_with_duplicate_handling(mo_obj_val[:, (hv_subfront_indices == i_fronts)], self.dyn_ref_point)
            # if hypervolume gradient is undefined, do x + F(x^(a) - x^(b)) with a and b from the same front, F in[0,2]
            # TODO
            # fill hv_gradient output array with result
            obj_space_multifront_hv_gradient[:, (hv_subfront_indices == i_fronts) ] = temp_grad_array

        # apply momentum on obj_space_multifront_hv_gradient
        if self.iter_counter == 1.0:
            self.weight_mean = obj_space_multifront_hv_gradient
        else:
            self.weight_mean = self.beta_one * self.weight_mean + (1 - self.beta_one) * obj_space_multifront_hv_gradient
        obj_space_multifront_hv_gradient = self.weight_mean

        # normalize the hv_gradient in obj space (||dHV/dY|| == 1)
        normalized_obj_space_multifront_hv_gradient = np.zeros((n_mo_obj,n_mo_sol))
        for i_mo_sol in range(0,n_mo_sol):
            w = np.sqrt(np.sum(obj_space_multifront_hv_gradient[:,i_mo_sol]**2.0))
            # if the length of the gradient is close to 0, leave the search direction un-normalized
            if np.isclose(w,0):
                w = 1
                print('w is 0')
            if self.obj_space_normalize:
                normalized_obj_space_multifront_hv_gradient[:,i_mo_sol] = obj_space_multifront_hv_gradient[:,i_mo_sol]/w
            else:
            # use this to deactivate normalization
                normalized_obj_space_multifront_hv_gradient[:,i_mo_sol] = obj_space_multifront_hv_gradient[:,i_mo_sol]

        # weighted hypervolume
        if self.weighted_hypervolume:
            hv_weights = np.exp(-normalized_obj_space_multifront_hv_gradient * self.gamma) / np.exp(-self.gamma)
            normalized_obj_space_multifront_hv_gradient = hv_weights * normalized_obj_space_multifront_hv_gradient

        dynamic_weights = torch.tensor(normalized_obj_space_multifront_hv_gradient, dtype=torch.float)
        return(dynamic_weights)
