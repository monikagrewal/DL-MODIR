import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb

"""
For information on DTLZ problem. Refer to https://pymoo.org/problems/many/dtlz.html
"""

# class Loss(nn.Module):
#     """
#     DTLZ1
#     """
#     def __init__(self, loss_name_list, n_var=2, **kwargs):
#         super().__init__()
#         self.n_obj = len(loss_name_list)
#         self.k = n_var - self.n_obj + 1


#     def g1(self, X_M):
#         return 100 * (self.k + torch.sum(torch.pow(X_M - 0.5, 2) - torch.cos(20 * 3.14 * (X_M - 0.5))))


#     def forward(self, x, y):
#         X_, X_M = x[:self.n_obj - 1], x[self.n_obj - 1:]
#         g = self.g1(X_M)
        
#         f = []
#         for i in range(0, self.n_obj):
#             _f = 0.5 * (1 + g)
#             _f *= torch.prod(X_[:len(X_) - i])
#             if i > 0:
#                 _f *= 1 - X_[len(X_) - i]
#             f.append(_f.view(-1))
#         return f
    

class Loss(nn.Module):
    """
    DTLZ2
    """
    def __init__(self, loss_name_list, n_var=2, **kwargs):
        super().__init__()
        self.n_obj = len(loss_name_list)
        self.k = n_var - self.n_obj + 1


    def g2(self, X_M):
        return torch.sum(torch.pow(X_M - 0.5, 2))
    
    def forward(self, x, y):
        alpha = 1
        X_, X_M = x[:self.n_obj - 1], x[self.n_obj - 1:]
        g = self.g2(X_M)

        f = []
        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= torch.prod(torch.cos(torch.pow(X_[:len(X_) - i], alpha) * 3.14 / 2.0))
            if i > 0:
                _f *= torch.sin(torch.pow(X_[len(X_) - i], alpha) * 3.14 / 2.0)

            f.append(_f.view(-1))
        
        # print(f"{x.data}, {[item.item() for item in f]}, {torch.sum(torch.tensor(f))}")
        
        return f


