import numpy as np
import torch


class UncertaintyWeighting(object):
    """
    UncertaintyWeighting MO optimizer.

    implementation of paper:
    Kendall, Alex, Yarin Gal, and Roberto Cipolla. 
    "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
    """
    def __init__(self, n_mo_sol, n_obj, weights=None):
        self.name = 'linear_scalarization'
        if weights is None:
            if n_obj==1:
                weights = torch.ones(n_mo_sol).view(1, -1)
            elif n_obj==2:
                if n_mo_sol==1:
                    weights = torch.ones(n_obj, n_mo_sol)
                else:
                    weights = torch.zeros(n_obj, n_mo_sol)
                    for i_mo_sol in range(0, n_mo_sol):
                        weights[0, i_mo_sol] = i_mo_sol/(n_mo_sol-1)
                        weights[1, i_mo_sol] = 1 - weights[0, i_mo_sol]
            else:
                raise ValueError('This is not yet generalized to more than 2 objectives')
        else:
            weights = torch.tensor(weights).float()
            assert weights.shape==(n_obj, n_mo_sol)
        
        self.weights = weights
        print("fixed weights: ", self.weights)


    def compute_weights(self, mo_obj_val):
        return self.weights