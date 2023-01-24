import numpy as np
import torch    

import sys
sys.path.append("../..")
from functions_hv_grad_3d import grad_multi_sweep_with_duplicate_handling
from functions_evaluation import compute_ud_gradient_2d


class UHV(object):
    """UHV based hypervolume maximization for dynamic weight calculation"""
    def __init__(self, n_mo_sol, n_mo_obj, ref_point, ud_eps=10.0**-5):
        super(UHV, self).__init__()
        self.name = 'uhv'
        self.ref_point = np.array(ref_point)
        self.ud_eps = ud_eps
        self.n_mo_sol = n_mo_sol
        self.n_mo_obj = n_mo_obj

        
    def compute_uhv_weights(self, mo_obj_val):
        n_mo_obj = self.n_mo_obj
        n_mo_sol = self.n_mo_sol
        hv_gradient = -1 * grad_multi_sweep_with_duplicate_handling(mo_obj_val, self.ref_point) # -1 * is needed, didn't check why
        if not (n_mo_obj == 2):
            raise ValueError('UHV cannot handle more than 2 objectives')
        ud_gradient = compute_ud_gradient_2d(mo_obj_val, self.ref_point, self.ud_eps)
        obj_space_uhv_gradient = hv_gradient - ud_gradient

        # normalize the uhv_gradient in obj space (||dUHV/dY|| == 1)
        use_obj_space_normalization = True
        normalized_obj_space_uhv_gradient = np.zeros((n_mo_obj, n_mo_sol))
        for i_mo_sol in range(0,n_mo_sol):
            w = np.sqrt(np.sum(obj_space_uhv_gradient[:,i_mo_sol]**2.0))
            # if the length of the gradient is close to 0, leave the search direction un-normalized
            if np.isclose(w,0):
                w = 1
                print('w is 0')
            if use_obj_space_normalization:
                normalized_obj_space_uhv_gradient[:,i_mo_sol] = obj_space_uhv_gradient[:,i_mo_sol]/w
            else:
            # use this to deactivate normalization
                normalized_obj_space_uhv_gradient[:,i_mo_sol] = obj_space_uhv_gradient[:,i_mo_sol]
                print('turned off normalization')

        dynamic_weights = -1 * torch.tensor(normalized_obj_space_uhv_gradient, dtype=torch.float)
        return(dynamic_weights)